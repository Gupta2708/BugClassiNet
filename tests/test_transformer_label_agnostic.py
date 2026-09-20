"""The shared transformer trainer must not assume any task-specific label."""

import numpy as np
import pandas as pd
import pytest

from bugclassinet.data.transformer import stable_sample_fingerprint
from bugclassinet.models import stage3
from bugclassinet.models.transformer_classifier import (
    _class_weight_diagnostics,
    _migrate_stored_manifest,
    _resolve_class_weights,
    identifier_column,
)

STAGE1_COUNTS = {"BUG": 5000, "DOCUMENTATION": 800, "ENHANCEMENT": 3000, "QUESTION": 1200}
STAGE2_COUNTS = {"BOH": 400, "MANDELBUG": 200}
STAGE3_COUNTS = {"ARB": 40, "NAM": 160}


def diagnostics(counts, strategy="balanced"):
    labels = sorted(counts)
    selected = _resolve_class_weights(labels, counts, strategy)
    effective = np.ones(len(labels)) if selected is None else selected
    return labels, selected, _class_weight_diagnostics(labels, counts, effective.tolist())


def test_stage1_weights_are_numerically_unchanged():
    """The frozen Stage-1 power-balanced weighting must survive the refactor."""
    labels = sorted(STAGE1_COUNTS)
    total = sum(STAGE1_COUNTS.values())
    expected = np.power(
        np.asarray([total / (len(labels) * STAGE1_COUNTS[label]) for label in labels]), 0.25
    )
    selected = _resolve_class_weights(labels, STAGE1_COUNTS, "quarter_balanced")
    np.testing.assert_allclose(selected, expected, rtol=0, atol=0)


def test_stage1_reference_is_the_most_frequent_class():
    _, _, report = diagnostics(STAGE1_COUNTS, "quarter_balanced")
    assert report["reference_label"] == "BUG"
    assert report["weight_ratios_to_reference"]["BUG"] == 1.0


@pytest.mark.parametrize(
    ("counts", "reference"),
    [(STAGE1_COUNTS, "BUG"), (STAGE2_COUNTS, "BOH"), (STAGE3_COUNTS, "NAM")],
)
def test_every_stage_resolves_without_a_bug_label(counts, reference):
    """Stage 2 (BOH/MANDELBUG) and Stage 3 (ARB/NAM) carry no BUG class."""
    labels, selected, report = diagnostics(counts)
    assert report["reference_label"] == reference
    assert np.all(np.isfinite(selected)) and np.all(selected > 0)
    assert set(report["class_weights_by_label"]) == set(labels)
    assert report["weight_ratios_to_reference"][reference] == 1.0


def test_stage3_balanced_weights_follow_the_repository_formula():
    labels = sorted(STAGE3_COUNTS)
    total = sum(STAGE3_COUNTS.values())
    expected = [total / (len(labels) * STAGE3_COUNTS[label]) for label in labels]
    np.testing.assert_allclose(_resolve_class_weights(labels, STAGE3_COUNTS, "balanced"), expected)
    # The rarer class must be up-weighted relative to the reference.
    _, _, report = diagnostics(STAGE3_COUNTS)
    assert report["weight_ratios_to_reference"]["ARB"] > 1.0


def test_unweighted_strategy_still_produces_diagnostics():
    """selected_weights=None must not break reference-label logging."""
    labels, selected, report = diagnostics(STAGE3_COUNTS, "none")
    assert selected is None
    assert report["reference_label"] == "NAM"
    assert report["class_weights_by_label"] == dict.fromkeys(labels, 1.0)
    assert report["weight_ratios_to_reference"] == dict.fromkeys(labels, 1.0)


def test_zero_reference_weight_does_not_divide_by_zero():
    report = _class_weight_diagnostics(["ARB", "NAM"], {"ARB": 1, "NAM": 2}, [0.0, 0.0])
    assert report["reference_label"] == "NAM"
    assert report["weight_ratios_to_reference"] is None


def test_reference_selection_is_deterministic_under_ties():
    first = _class_weight_diagnostics(["ARB", "NAM"], {"ARB": 100, "NAM": 100}, [1.0, 1.0])
    second = _class_weight_diagnostics(["NAM", "ARB"], {"ARB": 100, "NAM": 100}, [1.0, 1.0])
    assert first["reference_label"] == second["reference_label"] == "NAM"


def test_identifier_column_accepts_either_stable_identifier():
    assert identifier_column(["text", "canonical_label", "issue_id"]) == "issue_id"
    assert identifier_column(["text", "canonical_label", "issue_key"]) == "issue_key"
    # issue_id wins when both are present, preserving Stage-1 behaviour.
    assert identifier_column(["issue_key", "issue_id"]) == "issue_id"
    assert identifier_column(["text", "canonical_label"]) is None


def test_legacy_manifest_key_is_migrated_so_old_checkpoints_resume():
    migrated = _migrate_stored_manifest({"class_weight_ratios_to_bug": {"BUG": 1.0}})
    assert migrated == {"class_weight_ratios_to_reference": {"BUG": 1.0}}
    # A manifest already using the new field is left untouched.
    current = {"class_weight_ratios_to_reference": {"NAM": 1.0}}
    assert _migrate_stored_manifest(current) == current


class _FakeDataset:
    def __init__(self, columns):
        self.column_names = list(columns)
        self._columns = columns

    def __len__(self):
        return len(next(iter(self._columns.values())))

    def select_columns(self, names):
        return _FakeDataset({name: self._columns[name] for name in names})

    def iter(self, batch_size):
        yield {name: values[:batch_size] for name, values in self._columns.items()}


def test_sample_fingerprint_uses_issue_key_when_issue_id_is_absent():
    keyed = _FakeDataset({"issue_key": ["MySQL:1", "HTTPD:2"], "text": ["a", "b"]})
    content = _FakeDataset({"text": ["a", "b"], "canonical_label": ["ARB", "NAM"]})
    assert stable_sample_fingerprint(keyed) != stable_sample_fingerprint(content)
    changed = _FakeDataset({"issue_key": ["MySQL:1", "HTTPD:3"], "text": ["a", "b"]})
    assert stable_sample_fingerprint(keyed) != stable_sample_fingerprint(changed)


def research_frame():
    return pd.DataFrame(
        {
            "issue_key": ["MySQL:1", "HTTPD:2", "Linux:3"],
            "stage3_label": ["ARB", "NAM", "BOH"],
            "text_initial": ["initial arb", "initial nam", "initial boh"],
            "text_full": ["full arb", "full nam", "full boh"],
            "project": ["MySQL", "HTTPD", "Linux"],
        }
    )


def test_prepare_stage3_accepts_the_research_schema_and_drops_non_stage3_classes():
    frame = research_frame()
    prepared = stage3.prepare_stage3(frame, evidence_mode="initial")
    assert sorted(prepared["canonical_label"]) == ["ARB", "NAM"]
    assert sorted(prepared["text"]) == ["initial arb", "initial nam"]
    assert "issue_key" in prepared.columns and "project" in prepared.columns
    # BOH is dropped, never remapped into a Stage-3 class.
    assert "BOH" not in set(prepared["canonical_label"])
    # The caller's frame is untouched.
    assert "canonical_label" not in frame.columns


def test_prepare_stage3_full_evidence_selects_the_other_text_column():
    prepared = stage3.prepare_stage3(research_frame(), evidence_mode="full")
    assert sorted(prepared["text"]) == ["full arb", "full nam"]


def test_prepare_stage3_still_accepts_the_canonical_schema():
    canonical = pd.DataFrame(
        {
            "canonical_label": ["ARB", "NAM", "BOH"],
            "text": ["a", "b", "c"],
            "project": ["MySQL", "HTTPD", "Linux"],
        }
    )
    prepared = stage3.prepare_stage3(canonical)
    assert sorted(prepared["canonical_label"]) == ["ARB", "NAM"]
    assert sorted(prepared["text"]) == ["a", "b"]


def test_prepare_stage3_fails_loudly_on_missing_fields():
    with pytest.raises(ValueError, match="project"):
        stage3.prepare_stage3(pd.DataFrame({"stage3_label": ["ARB"], "text_initial": ["x"]}))
    with pytest.raises(ValueError, match="stage3_label or canonical_label"):
        stage3.prepare_stage3(pd.DataFrame({"text": ["x"], "project": ["MySQL"]}))
    with pytest.raises(ValueError, match="text_full or text"):
        stage3.prepare_stage3(
            pd.DataFrame({"stage3_label": ["ARB"], "text_initial": ["x"], "project": ["MySQL"]}),
            evidence_mode="full",
        )
    with pytest.raises(ValueError, match="Invalid evidence_mode"):
        stage3.prepare_stage3(research_frame(), evidence_mode="comments")
    with pytest.raises(ValueError, match="at least one ARB or NAM"):
        stage3.prepare_stage3(
            pd.DataFrame({"stage3_label": ["BOH"], "text_initial": ["x"], "project": ["Linux"]})
        )


def _fake_dependencies(created, tmp_path):
    """Minimal HF stand-ins so the trainer runs without downloading a model."""
    from types import SimpleNamespace

    from test_transformer_smoke import FakeDataset

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            input_ids = [[101, len(text), 102] for text in texts]
            return {
                "input_ids": input_ids,
                "attention_mask": [[1] * len(values) for values in input_ids],
            }

        def save_pretrained(self, output):
            return None

    class Trainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def train(self, **kwargs):
            control = SimpleNamespace(should_save=False, should_training_stop=False)
            state = SimpleNamespace(global_step=2, max_steps=2)
            for callback in self.kwargs["callbacks"]:
                for hook in ("on_train_begin", "on_train_end"):
                    if hasattr(callback, hook):
                        getattr(callback, hook)(None, state, control)

        def save_model(self, output):
            return None

        def predict(self, dataset):
            return SimpleNamespace(
                predictions=np.eye(2), metrics={"test_loss": 0.1, "test_runtime": 1.0}
            )

    def make_model(*args, **kwargs):
        created["model_kwargs"] = kwargs
        return SimpleNamespace(float=lambda: None)

    return FakeDataset, (
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            tensor=lambda *args, **kwargs: np.array(args[0]),
            float=float,
        ),
        FakeDataset,
        SimpleNamespace(from_pretrained=make_model),
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer()),
        lambda **kwargs: object(),
        lambda **kwargs: object(),
        Trainer,
        object,
        lambda **kwargs: SimpleNamespace(n_gpu=1, world_size=1, parallel_mode="not_distributed"),
    )


def test_stage3_arb_nam_reaches_training_without_the_bug_reference_error(tmp_path, monkeypatch):
    """Regression: this previously raised a BUG-reference-class ValueError."""
    import json

    from bugclassinet.models import transformer_classifier as module
    from bugclassinet.models.transformer_classifier import (
        TransformerTrainingConfig,
        train_transformer,
    )

    created: dict[str, object] = {}
    FakeDataset, dependencies = _fake_dependencies(created, tmp_path)
    monkeypatch.setattr(module, "_dependencies", lambda: dependencies)

    def inspect_dataset(dataset, split_name, required_columns):
        assert required_columns <= set(dataset.column_names)
        counts: dict[str, int] = {}
        for row in dataset.rows:
            counts[str(row["canonical_label"])] = counts.get(str(row["canonical_label"]), 0) + 1
        return counts

    monkeypatch.setattr(module, "inspect_dataset", inspect_dataset)
    FakeDataset.map_calls.clear()
    # Stage-3 labels only: no BUG class exists anywhere in this run.
    train = FakeDataset(
        [
            {"text": "aging related", "canonical_label": "ARB"},
            {"text": "non aging", "canonical_label": "NAM"},
        ],
        fingerprint="train",
    )
    validation = FakeDataset(
        [
            {"issue_key": "MySQL:1", "project": "MySQL", "text": "aging", "canonical_label": "ARB"},
            {"issue_key": "HTTPD:2", "project": "HTTPD", "text": "non", "canonical_label": "NAM"},
        ],
        fingerprint="validation",
    )

    metrics = train_transformer(
        train,
        validation,
        tmp_path,
        TransformerTrainingConfig(model_name="tiny", epochs=1, class_weight_strategy="balanced"),
    )

    assert metrics["eval_macro_f1"] == 1.0
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    # Generic diagnostics replace the Stage-1-only ratios field.
    assert manifest["class_weight_reference_label"] in {"ARB", "NAM"}
    assert "class_weight_ratios_to_bug" not in manifest
    assert set(manifest["class_weights_by_label"]) == {"ARB", "NAM"}
    # issue_key carried the prediction identity with no adapter column.
    predictions = pd.read_parquet(tmp_path / "validation_predictions.parquet")
    assert predictions.columns.tolist()[0] == "issue_key"
    assert sorted(predictions["issue_key"]) == ["HTTPD:2", "MySQL:1"]


def test_stage2_boh_mandelbug_reaches_training_through_the_same_path(tmp_path, monkeypatch):
    """The fix is generic: Stage 2 has no BUG class either."""
    from bugclassinet.models import transformer_classifier as module
    from bugclassinet.models.transformer_classifier import (
        TransformerTrainingConfig,
        train_transformer,
    )

    created: dict[str, object] = {}
    FakeDataset, dependencies = _fake_dependencies(created, tmp_path)
    monkeypatch.setattr(module, "_dependencies", lambda: dependencies)

    def inspect_dataset(dataset, split_name, required_columns):
        counts: dict[str, int] = {}
        for row in dataset.rows:
            counts[str(row["canonical_label"])] = counts.get(str(row["canonical_label"]), 0) + 1
        return counts

    monkeypatch.setattr(module, "inspect_dataset", inspect_dataset)
    FakeDataset.map_calls.clear()
    train = FakeDataset(
        [
            {"text": "deterministic", "canonical_label": "BOH"},
            {"text": "transient", "canonical_label": "MANDELBUG"},
        ],
        fingerprint="train",
    )
    validation = FakeDataset(
        [
            {"issue_key": "Linux:1", "text": "deterministic", "canonical_label": "BOH"},
            {"issue_key": "AXIS:2", "text": "transient", "canonical_label": "MANDELBUG"},
        ],
        fingerprint="validation",
    )

    metrics = train_transformer(
        train,
        validation,
        tmp_path,
        TransformerTrainingConfig(model_name="tiny", epochs=1, class_weight_strategy="balanced"),
    )
    assert metrics["eval_macro_f1"] == 1.0
