import re

import pytest

from bugclassinet.evaluation.stage1 import (
    _assert_parameters_unchanged,
    _build_evaluation_manifest,
    _checkpoint_model_sha256,
    _evaluation_only_trainer_class,
    _validate_complete_checkpoint,
)


def test_checkpoint_hash_identifies_persisted_model_bytes(tmp_path) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"first frozen model")
    first = _checkpoint_model_sha256(tmp_path)

    assert len(first) == 64
    weights.write_bytes(b"different frozen model")
    assert _checkpoint_model_sha256(tmp_path) != first


def test_checkpoint_hash_requires_model_weights(tmp_path) -> None:
    with pytest.raises(ValueError, match="No persisted model weights"):
        _checkpoint_model_sha256(tmp_path)


def test_evaluation_only_trainer_rejects_training() -> None:
    class TrainerBase:
        pass

    trainer = _evaluation_only_trainer_class(TrainerBase)()
    with pytest.raises(RuntimeError, match="Training is disabled"):
        trainer.train()


def test_parameter_integrity_mismatch_fails() -> None:
    _assert_parameters_unchanged("same", "same")
    with pytest.raises(RuntimeError, match="parameters changed"):
        _assert_parameters_unchanged("before", "after")


def test_evaluation_manifest_records_required_frozen_identities(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-68106"
    test_data = tmp_path / "test.parquet"
    result = _build_evaluation_manifest(
        checkpoint_path=checkpoint,
        checkpoint_hash="a" * 64,
        parameter_hash_before="b" * 64,
        parameter_hash_after="b" * 64,
        model_name="microsoft/deberta-v3-small",
        model_revision="revision",
        config_hash="c" * 64,
        test_path=test_data,
        test_hash="d" * 64,
        label_mapping={
            "BUG": 0,
            "DOCUMENTATION": 1,
            "ENHANCEMENT": 2,
            "QUESTION": 3,
        },
        max_length=256,
        preprocessing_version="v1",
        tokenizer_source="/model/tokenizer",
        ending_global_step=68_106,
        rows=142_320,
    )

    assert result["final_checkpoint_path"] == str(checkpoint)
    assert result["final_checkpoint_sha256"] == "a" * 64
    assert result["test_dataset_sha256"] == "d" * 64
    assert result["rows"] == 142_320
    assert result["model_parameters_unchanged"] is True
    assert result["training_invoked"] is False
    assert result["class_weights_used_for_inference"] is False
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T.*Z", result["timestamp"])


def test_official_evaluation_rejects_incomplete_checkpoint(tmp_path) -> None:
    manifest = {"ending_global_step": 44_000, "expected_total_optimizer_steps": 68_106}
    with pytest.raises(ValueError, match="completed final checkpoint"):
        _validate_complete_checkpoint(manifest, tmp_path)


def test_complete_checkpoint_step_matches_trainer_state(tmp_path) -> None:
    manifest = {"ending_global_step": 68_106, "expected_total_optimizer_steps": 68_106}
    (tmp_path / "trainer_state.json").write_text('{"global_step": 68106}', encoding="utf-8")

    assert _validate_complete_checkpoint(manifest, tmp_path) == 68_106
