import json
import shutil
from types import SimpleNamespace

import pandas as pd
import pytest

from bugclassinet.cli import build_parser
from bugclassinet.models import transformer_classifier as module


def _stable_manifest(train_identity, validation_identity) -> dict[str, object]:
    label_mapping = {"BUG": 0, "DOCUMENTATION": 1}
    return {
        "dataset_path": train_identity["path"],
        "dataset_fingerprint": "hf-session-a",
        "train_hf_fingerprint": "hf-session-a",
        "train_source_sha256": train_identity["sha256"],
        "validation_path": validation_identity["path"],
        "validation_fingerprint": "hf-validation-session-a",
        "validation_hf_fingerprint": "hf-validation-session-a",
        "validation_source_sha256": validation_identity["sha256"],
        "training_rows": 2,
        "train_rows": 2,
        "validation_rows": 2,
        "class_counts": {"BUG": 1, "DOCUMENTATION": 1},
        "train_class_counts": {"BUG": 1, "DOCUMENTATION": 1},
        "validation_class_counts": {"BUG": 1, "DOCUMENTATION": 1},
        "train_schema_columns": train_identity["schema_columns"],
        "validation_schema_columns": validation_identity["schema_columns"],
        "train_required_columns": ["text", "canonical_label"],
        "validation_required_columns": ["issue_id", "text", "canonical_label"],
        "train_checksum_provenance": train_identity["checksum_provenance"],
        "validation_checksum_provenance": validation_identity["checksum_provenance"],
        "preprocessing_version": module._PREPROCESSING_VERSION,
        "seed": 42,
        "model_name": "model",
        "model_revision": "revision",
        "config_hash": "config-a",
        "label_mapping": label_mapping,
        "label_mapping_hash": module._json_hash(label_mapping),
        "class_weights": [1.0, 1.0],
        "training_settings": {"epochs": 1},
        "expected_total_optimizer_steps": 10,
        "ending_global_step": 4,
    }


def _parquet_identities(tmp_path):
    frame = pd.DataFrame(
        {
            "issue_id": ["1", "2"],
            "text": ["bug text", "docs text"],
            "canonical_label": ["BUG", "DOCUMENTATION"],
        }
    )
    train = tmp_path / "train_clean.parquet"
    validation = tmp_path / "validation_clean.parquet"
    frame.to_parquet(train, index=False)
    shutil.copy2(train, validation)
    (tmp_path / "checksums.json").write_text(
        json.dumps(
            {
                train.name: module.sha256_file(train),
                validation.name: module.sha256_file(validation),
            }
        ),
        encoding="utf-8",
    )
    return (
        module._source_file_identity(train, ["text", "canonical_label"]),
        module._source_file_identity(validation, ["issue_id", "text", "canonical_label"]),
    )


def _simulate_segment(
    directory,
    *,
    starting_step: int,
    planned_steps: int,
    scheduler_step: int,
    stop_after_steps: int | None,
    resumed_expected_steps: int | None = None,
):
    manifest = {"status": "initialized"}
    callback = module._make_execution_callback(
        object,
        manifest,
        directory,
        stop_after_steps,
        resumed_expected_steps,
    )
    state = SimpleNamespace(global_step=starting_step, max_steps=planned_steps)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    callback.on_train_begin(None, state, control)
    observed_scheduler_steps = []
    while state.global_step < state.max_steps:
        state.global_step += 1
        scheduler_step += 1
        observed_scheduler_steps.append(scheduler_step)
        control.should_save = False
        control.should_training_stop = False
        callback.on_step_end(None, state, control)
        if control.should_save:
            checkpoint = directory / f"checkpoint-{state.global_step}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            for name in (
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
                "model.safetensors",
            ):
                (checkpoint / name).write_text("fixture", encoding="utf-8")
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"global_step": state.global_step}), encoding="utf-8"
            )
            callback.on_save(None, state, control)
        if control.should_training_stop:
            break
    callback.on_train_end(None, state, control)
    return state, scheduler_step, observed_scheduler_steps, manifest, callback


def test_split_resume_matches_uninterrupted_global_and_scheduler_steps(tmp_path) -> None:
    uninterrupted = _simulate_segment(
        tmp_path / "uninterrupted",
        starting_step=0,
        planned_steps=10,
        scheduler_step=0,
        stop_after_steps=None,
    )
    first = _simulate_segment(
        tmp_path / "segment-1",
        starting_step=0,
        planned_steps=10,
        scheduler_step=0,
        stop_after_steps=4,
    )
    second = _simulate_segment(
        tmp_path / "segment-2",
        starting_step=first[0].global_step,
        planned_steps=10,
        scheduler_step=first[1],
        stop_after_steps=None,
        resumed_expected_steps=first[3]["expected_total_optimizer_steps"],
    )

    assert uninterrupted[0].global_step == second[0].global_step == 10
    assert uninterrupted[1] == second[1] == 10
    assert first[0].global_step == 4
    assert first[2] == [1, 2, 3, 4]
    assert second[2] == [5, 6, 7, 8, 9, 10]
    assert second[3]["starting_global_step"] == 4
    assert first[3]["expected_total_optimizer_steps"] == 10
    assert first[0].max_steps == 10
    assert first[4].stopped_early is True
    assert first[4].final_checkpoint.endswith("checkpoint-4")


def test_different_hf_fingerprints_resume_when_source_sha_matches(tmp_path) -> None:
    train_identity, validation_identity = _parquet_identities(tmp_path)
    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 4}), encoding="utf-8")
    stored = _stable_manifest(train_identity, validation_identity)
    current = dict(stored)
    current["dataset_fingerprint"] = "hf-session-b"
    current["validation_fingerprint"] = "hf-validation-session-b"
    current["train_hf_fingerprint"] = "hf-session-b"
    current["validation_hf_fingerprint"] = "hf-validation-session-b"

    module._validate_resume_manifest(stored, current, checkpoint)


def test_different_parquet_content_is_rejected(tmp_path) -> None:
    train_identity, validation_identity = _parquet_identities(tmp_path)
    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 4}), encoding="utf-8")
    stored = _stable_manifest(train_identity, validation_identity)
    changed = tmp_path / "changed.parquet"
    pd.DataFrame(
        {
            "issue_id": ["1", "2"],
            "text": ["changed bug text", "docs text"],
            "canonical_label": ["BUG", "DOCUMENTATION"],
        }
    ).to_parquet(changed, index=False)
    changed_identity = module._source_file_identity(changed, ["text", "canonical_label"])
    current = dict(stored)
    current["train_source_sha256"] = changed_identity["sha256"]

    with pytest.raises(ValueError, match="train_source_sha256"):
        module._validate_resume_manifest(stored, current, checkpoint)


@pytest.mark.parametrize(
    "key",
    ["train_rows", "config_hash", "seed", "label_mapping", "label_mapping_hash"],
)
def test_stable_resume_manifest_mismatch_is_rejected(tmp_path, key: str) -> None:
    train_identity, validation_identity = _parquet_identities(tmp_path)
    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 4}), encoding="utf-8")
    stored = _stable_manifest(train_identity, validation_identity)
    current = dict(stored)
    current[key] = "different"

    with pytest.raises(ValueError, match=key):
        module._validate_resume_manifest(stored, current, checkpoint)


def test_legacy_manifest_allows_hf_fingerprint_change_only_with_persisted_metadata(
    tmp_path,
) -> None:
    train_identity, validation_identity = _parquet_identities(tmp_path)
    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 4}), encoding="utf-8")
    current = _stable_manifest(train_identity, validation_identity)
    legacy = {
        key: value
        for key, value in current.items()
        if key
        in {
            "dataset_path",
            "dataset_fingerprint",
            "validation_path",
            "validation_fingerprint",
            "training_rows",
            "validation_rows",
            "seed",
            "model_name",
            "model_revision",
            "config_hash",
            "label_mapping",
            "class_counts",
            "class_weights",
            "training_settings",
            "expected_total_optimizer_steps",
            "ending_global_step",
        }
    }
    current["dataset_fingerprint"] = "different-hf-fingerprint"
    current["validation_fingerprint"] = "different-validation-hf-fingerprint"

    module._validate_resume_manifest(legacy, current, checkpoint)


def test_published_checksum_rejects_changed_parquet(tmp_path) -> None:
    train_identity, _ = _parquet_identities(tmp_path)
    train = tmp_path / "train_clean.parquet"
    pd.DataFrame(
        {
            "issue_id": ["1", "2"],
            "text": ["changed", "docs text"],
            "canonical_label": ["BUG", "DOCUMENTATION"],
        }
    ).to_parquet(train, index=False)

    assert train_identity["sha256"] != module.sha256_file(train)
    with pytest.raises(ValueError, match="does not match published metadata"):
        module._source_file_identity(train, ["text", "canonical_label"])


def test_resume_rejects_different_expected_total_steps(tmp_path) -> None:
    manifest = {"status": "initialized"}
    callback = module._make_execution_callback(
        object,
        manifest,
        tmp_path,
        stop_after_steps=None,
        resumed_expected_steps=11,
    )
    state = SimpleNamespace(global_step=4, max_steps=10)
    control = SimpleNamespace(should_save=False, should_training_stop=False)

    with pytest.raises(ValueError, match="planned total steps differ"):
        callback.on_train_begin(None, state, control)


def test_resume_checkpoint_requires_full_trainer_state(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    for name in (
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "rng_state.pth",
        "model.safetensors",
        "run_manifest.json",
    ):
        (checkpoint / name).write_text("{}", encoding="utf-8")

    normalized, manifest = module._require_resume_checkpoint(checkpoint, use_fp16=False)
    assert normalized == checkpoint.resolve()
    assert manifest == {}

    with pytest.raises(ValueError, match="scaler.pt"):
        module._require_resume_checkpoint(checkpoint, use_fp16=True)


def test_stage1_cli_exposes_segmented_resume_options() -> None:
    args = build_parser().parse_args(
        [
            "train-stage1",
            "--resume-from-checkpoint",
            "/input/checkpoint-18000",
            "--stop-after-steps",
            "18000",
            "--skip-final-evaluation",
        ]
    )

    assert args.resume_from_checkpoint == "/input/checkpoint-18000"
    assert args.stop_after_steps == 18000
    assert args.skip_final_evaluation is True
