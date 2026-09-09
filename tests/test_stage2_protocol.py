import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, f1_score

from bugclassinet.data.mandelbugs_schema import LABELS, PROJECTS
from bugclassinet.training.stage2_baseline import train_baseline
from bugclassinet.training.stage2_modernbert import EvidenceDataset
from bugclassinet.training.stage2_protocol import (
    bootstrap_ci,
    inner_validation,
    load_stage2,
    lopo_folds,
    metrics,
)


@pytest.fixture
def dataset(tmp_path):
    rows = []
    for project in PROJECTS:
        for i in range(12):
            label = LABELS[i % 2]
            prefix = "deterministic reproducible" if i % 2 == 0 else "intermittent rare"
            text = f"{prefix} failure {project.lower()} unique{i}"
            rows.append(
                {
                    "issue_key": f"{project}:{i}",
                    "project": project,
                    "stage2_label": label,
                    "text_initial": text,
                    "text_full": text + " later comment",
                }
            )
    path = tmp_path / "fixture.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_lopo_and_inner_split_integrity(dataset):
    frame = load_stage2(dataset, "initial")
    visited = []
    for project, train, test, _ in lopo_folds(frame):
        assert project not in set(frame.iloc[train].project)
        assert set(frame.iloc[test].project) == {project}
        fit, valid = inner_validation(frame, train, 42)
        assert set(fit) | set(valid) == set(train)
        assert not set(fit) & set(valid)
        assert not set(frame.iloc[fit].text_group) & set(frame.iloc[valid].text_group)
        assert set(frame.iloc[fit].stage2_label) == set(LABELS)
        assert np.array_equal(fit, inner_validation(frame, train, 42)[0])
        visited.extend(test)
    assert sorted(visited) == list(range(len(frame)))


def test_cross_project_text_removed_only_from_train(dataset):
    frame = load_stage2(dataset, "initial")
    first = frame.index[frame.project.eq("Linux")][0]
    other = frame.index[frame.project.eq("AXIS")][0]
    frame.loc[other, "text_group"] = frame.loc[first, "text_group"]
    project, train, test, purged = lopo_folds(frame)[0]
    assert project == "Linux" and first in test and other not in train
    assert frame.loc[other, "issue_key"] in purged
    assert len(test) == 12


def test_metrics_orientation_and_undefined_auc():
    truth = ["BOH", "BOH", "MANDELBUG", "MANDELBUG"]
    predicted = ["BOH", "MANDELBUG", "MANDELBUG", "MANDELBUG"]
    values = metrics(truth, predicted, [0.1, 0.8, 0.7, 0.9])
    assert values["confusion_matrix"] == [[1, 1], [0, 2]]
    assert values["macro_f1"] == pytest.approx(
        f1_score(truth, predicted, labels=LABELS, average="macro")
    )
    assert values["pr_auc_mandelbug"] == pytest.approx(
        average_precision_score([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9])
    )
    assert metrics(["BOH"], ["BOH"], [0.0])["pr_auc_mandelbug"] is None


def test_tiny_baseline_deterministic_artifacts_and_fold_only_vocabulary(dataset, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("bootstrap_iterations: 5\nword_min_df: 1\n")
    runs = [tmp_path / "run1", tmp_path / "run2"]
    for out in runs:
        train_baseline(str(dataset), "tfidf_svm", "initial", str(out), str(config))
    one, two = [pd.read_parquet(out / "predictions.parquet") for out in runs]
    pd.testing.assert_frame_equal(one, two)
    assert len(one) == 48 and one.issue_key.nunique() == 48
    assert len(pd.read_parquet(runs[0] / "majority/predictions.parquet")) == 48
    summary = json.loads((runs[0] / "summary.json").read_text())
    assert summary["primary_metric"] == "macro_f1"
    for project in PROJECTS:
        pipeline = joblib.load(runs[0] / "folds" / project / "model.joblib")
        assert project.lower() not in pipeline.named_steps["features"].vocabulary_
        manifest = json.loads((runs[0] / "folds" / project / "run_manifest.json").read_text())
        assert project not in manifest["train_projects"] and manifest["test_projects"] == [project]
    assert bootstrap_ci(one, 42, 5) == bootstrap_ci(two, 42, 5)
    with pytest.raises(ValueError, match="nonempty"):
        train_baseline(str(dataset), "tfidf_svm", "initial", str(runs[0]), str(config))


def test_neural_tokenization_is_lazy_and_metadata_free():
    calls = []

    def tokenizer(text, **kwargs):
        calls.append((text, kwargs))
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    dataset = EvidenceDataset(["Issue evidence"], ["MANDELBUG"], tokenizer, 256)
    assert not calls
    assert dataset[0]["labels"] == 1
    assert calls == [("Issue evidence", {"truncation": True, "max_length": 256, "padding": False})]


@pytest.mark.parametrize("model", ["sbert_logreg", "frozen_stage1_encoder_transfer"])
def test_embedding_heads_share_outer_folds_without_encoder_training(
    dataset, tmp_path, monkeypatch, model
):
    from bugclassinet.training import stage2_embeddings

    calls = []

    def fixture_embeddings(texts, config, cache_dir, stage1_model):
        calls.append(stage1_model)
        # Stub only the expensive frozen encoder; fit the real fold-wise linear heads.
        vectors = np.array([[float("intermittent" in t), 1.0] for t in texts], dtype=np.float32)
        return vectors, {"encoder_training_invoked": False, "fixture_only": True}

    monkeypatch.setattr(stage2_embeddings, "frozen_embeddings", fixture_embeddings)
    config = tmp_path / "config.yaml"
    config.write_text("bootstrap_iterations: 3\n")
    out = tmp_path / "embedding_run"
    train_baseline(
        str(dataset), model, "initial", str(out), str(config), stage1_model="frozen_fixture"
    )
    assert calls == ["frozen_fixture" if model == "frozen_stage1_encoder_transfer" else None]
    result = pd.read_parquet(out / "predictions.parquet")
    assert result.issue_key.nunique() == len(result) == 48
    assert set(result.project) == set(PROJECTS)
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["encoder"]["encoder_training_invoked"] is False
    for fold in manifest["folds"]:
        assert fold["held_out_project"] not in fold["train_projects"]


def test_weighted_loss_and_frozen_lower_layers():
    torch = pytest.importorskip("torch")
    from bugclassinet.training.stage2_modernbert import freeze_lower_layers, weighted_loss

    logits = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
    labels, weights = torch.tensor([0, 1]), torch.tensor([0.7, 1.3])
    assert weighted_loss(logits, labels, weights).item() == pytest.approx(
        torch.nn.functional.cross_entropy(logits, labels, weight=weights).item()
    )
    from types import SimpleNamespace

    base = SimpleNamespace(
        embeddings=torch.nn.Embedding(5, 2),
        layers=torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]),
    )
    model = SimpleNamespace(base_model=base)
    freeze_lower_layers(model, 1)
    assert not any(p.requires_grad for p in base.layers[0].parameters())
    assert all(p.requires_grad for p in base.layers[1].parameters())
