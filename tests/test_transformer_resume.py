import json
from types import SimpleNamespace

import pytest

from bugclassinet.cli import build_parser
from bugclassinet.models import transformer_classifier as module


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


@pytest.mark.parametrize("key", ["dataset_fingerprint", "config_hash"])
def test_resume_manifest_mismatch_is_rejected(tmp_path, key: str) -> None:
    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 4}), encoding="utf-8")
    stored = {
        "dataset_fingerprint": "train-a",
        "validation_fingerprint": "validation-a",
        "training_rows": 10,
        "validation_rows": 4,
        "seed": 42,
        "model_name": "model",
        "model_revision": "revision",
        "config_hash": "config-a",
        "label_mapping": {"BUG": 0},
        "class_counts": {"BUG": 10},
        "class_weights": [1.0],
        "training_settings": {"epochs": 1},
        "ending_global_step": 4,
    }
    current = dict(stored)
    current[key] = "different"

    with pytest.raises(ValueError, match=key):
        module._validate_resume_manifest(stored, current, checkpoint)


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
