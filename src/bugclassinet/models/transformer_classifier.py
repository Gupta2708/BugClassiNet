"""Reusable Hugging Face sequence-classification trainer."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from bugclassinet.data.transformer import (
    inspect_dataset,
    load_parquet_dataset,
    stratified_limit,
)
from bugclassinet.evaluation.metrics import classification_metrics
from bugclassinet.features.text import require_model_text
from bugclassinet.utils.checksums import sha256_file
from bugclassinet.utils.memory import log_memory
from bugclassinet.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

_PREPROCESSING_VERSION = "nlbse2023-clean-parquet-v1"

_CHECKPOINT_MODEL_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unavailable"


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_path(value: Any) -> str | None:
    return str(Path(value).resolve()) if isinstance(value, (str, Path)) else None


def _dataset_fingerprint(dataset: Any) -> str:
    return str(getattr(dataset, "_fingerprint", f"rows-{len(dataset)}"))


def _checksum_claims(path: Path) -> tuple[list[str], list[str]]:
    claims: list[str] = []
    sources: list[str] = []
    checksum_path = path.parent / "checksums.json"
    if checksum_path.is_file():
        checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
        value = checksums.get(path.name)
        if value is not None:
            claims.append(str(value).lower())
            sources.append("checksums.json")

    manifest_path = path.parent / "data_manifest.json"
    if manifest_path.is_file():
        data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = data_manifest.get("processed_files", {}).get(path.name)
        if value is not None:
            claims.append(str(value).lower())
            sources.append("data_manifest.json")
    return claims, sources


def _source_file_identity(path: str | Path, required_columns: list[str]) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Training source does not exist: {source}")
    schema_columns = pq.read_schema(source).names
    missing = sorted(set(required_columns) - set(schema_columns))
    if missing:
        raise ValueError(f"{source.name} is missing required columns: {missing}")

    claims, provenance = _checksum_claims(source)
    if len(set(claims)) > 1:
        raise ValueError(
            f"Checksum metadata disagrees for {source.name}: "
            f"{dict(zip(provenance, claims, strict=True))}"
        )
    actual = sha256_file(source)
    if claims and claims[0] != actual:
        raise ValueError(
            f"Source SHA-256 does not match published metadata for {source.name}: "
            f"metadata={claims[0]}, actual={actual}"
        )
    LOGGER.info(
        "Verified source identity file=%s sha256=%s checksum_provenance=%s",
        source,
        actual,
        provenance or ["computed"],
    )
    return {
        "path": str(source),
        "sha256": actual,
        "schema_columns": schema_columns,
        "required_columns": required_columns,
        "checksum_provenance": provenance or ["computed"],
    }


def _model_revision(tokenizer: Any) -> str:
    options = getattr(tokenizer, "init_kwargs", {})
    return str(
        options.get("_commit_hash") or getattr(tokenizer, "_commit_hash", None) or "unresolved"
    )


def _write_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _require_resume_checkpoint(path: str | Path, use_fp16: bool) -> tuple[Path, dict[str, Any]]:
    checkpoint = Path(path).resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"Resume checkpoint does not exist: {checkpoint}")
    required = ["optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth"]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if not any((checkpoint / name).is_file() for name in _CHECKPOINT_MODEL_FILES):
        missing.append("model weights")
    if use_fp16 and not (checkpoint / "scaler.pt").is_file():
        missing.append("scaler.pt")
    if missing:
        raise ValueError(
            f"Resume checkpoint is incomplete ({checkpoint}); missing: {', '.join(missing)}"
        )
    manifest_path = checkpoint / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Resume checkpoint has no run_manifest.json: {checkpoint}")
    return checkpoint, json.loads(manifest_path.read_text(encoding="utf-8"))


def _validate_resume_manifest(
    stored: dict[str, Any], current: dict[str, Any], checkpoint: Path
) -> None:
    stable_identity_available = bool(
        stored.get("train_source_sha256") and stored.get("validation_source_sha256")
    )
    compared = (
        "train_source_sha256",
        "validation_source_sha256",
        "train_rows",
        "validation_rows",
        "train_class_counts",
        "validation_class_counts",
        "train_schema_columns",
        "validation_schema_columns",
        "train_required_columns",
        "validation_required_columns",
        "label_mapping_hash",
        "preprocessing_version",
        "seed",
        "model_name",
        "model_revision",
        "config_hash",
        "label_mapping",
        "class_counts",
        "class_weights",
        "training_settings",
    )
    if not stable_identity_available:
        compared = (
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
        )
        path_pairs = (
            ("dataset_path", "train_checksum_provenance"),
            ("validation_path", "validation_checksum_provenance"),
        )
        legacy_failures = []
        for path_key, provenance_key in path_pairs:
            if not stored.get(path_key) or stored.get(path_key) != current.get(path_key):
                legacy_failures.append(f"{path_key} must exactly match the legacy checkpoint")
            provenance = current.get(provenance_key, [])
            if not provenance or provenance == ["computed"]:
                legacy_failures.append(
                    f"{path_key} requires checksums.json or data_manifest.json for legacy resume"
                )
        if legacy_failures:
            raise ValueError(
                "Legacy checkpoint cannot be verified against persistent source metadata: "
                + "; ".join(legacy_failures)
            )
        LOGGER.warning(
            "Accepting legacy checkpoint manifest without source SHA-256 because source paths "
            "match exactly and current Parquet files were verified against persisted "
            "checksum metadata"
        )

    mismatches = [key for key in compared if stored.get(key) != current.get(key)]
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={stored.get(key)!r}, current={current.get(key)!r}"
            for key in mismatches
        )
        raise ValueError(f"Checkpoint run is incompatible with the current run: {details}")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    checkpoint_step = int(state.get("global_step", -1))
    if checkpoint_step < 0 or checkpoint_step != stored.get("ending_global_step"):
        raise ValueError(
            "Checkpoint global step disagrees with run_manifest.json: "
            f"trainer_state={checkpoint_step}, manifest={stored.get('ending_global_step')}"
        )


def _make_execution_callback(
    callback_base: type,
    manifest: dict[str, Any],
    destination: Path,
    stop_after_steps: int | None,
    resumed_expected_steps: int | None,
    force_final_checkpoint: bool = False,
) -> Any:
    class ExecutionCallback(callback_base):
        """Persist run identity and stop only after requesting a full checkpoint."""

        stopped_early = False
        final_checkpoint: str | None = None

        def _persist(self, state: Any, checkpoint: bool = False) -> None:
            manifest["ending_global_step"] = int(state.global_step)
            _write_run_manifest(destination / "run_manifest.json", manifest)
            if checkpoint:
                checkpoint_dir = destination / f"checkpoint-{state.global_step}"
                if not checkpoint_dir.is_dir():
                    raise RuntimeError(
                        f"Trainer did not create expected checkpoint: {checkpoint_dir}"
                    )
                _write_run_manifest(checkpoint_dir / "run_manifest.json", manifest)
                _require_resume_checkpoint(
                    checkpoint_dir,
                    use_fp16=bool(manifest.get("training_settings", {}).get("fp16")),
                )
                self.final_checkpoint = str(checkpoint_dir.resolve())

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            expected_steps = int(state.max_steps)
            starting_step = int(state.global_step)
            if resumed_expected_steps is not None and expected_steps != resumed_expected_steps:
                raise ValueError(
                    "Checkpoint planned total steps differ from the current Trainer plan: "
                    f"checkpoint={resumed_expected_steps}, current={expected_steps}"
                )
            if (
                stop_after_steps is not None
                and not starting_step < stop_after_steps <= expected_steps
            ):
                raise ValueError(
                    "stop_after_steps must be greater than the starting global step and no greater "
                    f"than the planned total ({starting_step} < N <= {expected_steps})"
                )
            manifest["expected_total_optimizer_steps"] = expected_steps
            manifest["starting_global_step"] = starting_step
            manifest["ending_global_step"] = starting_step
            manifest["status"] = "running"
            _write_run_manifest(destination / "run_manifest.json", manifest)
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            at_planned_end = int(state.global_step) >= int(state.max_steps)
            at_segment_end = (
                stop_after_steps is not None and int(state.global_step) >= stop_after_steps
            )
            if (force_final_checkpoint and at_planned_end) or at_segment_end:
                control.should_save = True
            if at_segment_end and not at_planned_end:
                manifest["status"] = "stopped"
                self.stopped_early = True
                control.should_training_stop = True
            return control

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            self._persist(state, checkpoint=True)
            return control

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if not self.stopped_early:
                manifest["status"] = "training_complete"
            self._persist(state)
            return control

    return ExecutionCallback()


@dataclass(frozen=True)
class TransformerTrainingConfig:
    """Configurable, reproducible sequence-classification settings."""

    model_name: str
    max_length: int = 256
    batch_size: int = 8
    learning_rate: float = 2e-5
    epochs: int = 5
    gradient_accumulation_steps: int = 1
    seed: int = 42
    early_stopping_patience: int = 2
    max_steps: int = -1
    tokenization_batch_size: int = 512
    tokenization_writer_batch_size: int = 512
    dataloader_num_workers: int = 0
    dataset_cache_dir: str | None = None
    checkpoint_steps: int | None = None
    save_total_limit: int | None = None


def _dependencies() -> tuple[Any, ...]:
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as error:
        raise ImportError(
            "Transformer training requires `pip install -e .[transformers]` and `datasets`."
        ) from error
    return (
        torch,
        Dataset,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )


def _tokenize_batch(
    batch: dict[str, list[Any]],
    tokenizer: Any,
    label_to_id: dict[str, int],
    max_length: int,
) -> dict[str, list[Any]]:
    """Tokenize one bounded batch without global padding."""
    text = batch["text"]
    labels = batch["canonical_label"]
    if any(not isinstance(value, str) or not value.strip() for value in text):
        raise ValueError("Model input contains missing or blank text")
    encoded = tokenizer(text, truncation=True, max_length=max_length, padding=False)
    encoded["labels"] = [label_to_id[label] for label in labels]
    return encoded


def _tokenization_fingerprint(
    dataset: Any,
    split_name: str,
    model_name: str,
    tokenizer: Any,
    max_length: int,
    label_to_id: dict[str, int],
) -> str:
    tokenizer_options = getattr(tokenizer, "init_kwargs", {})
    payload = {
        "source": getattr(dataset, "_fingerprint", f"rows-{len(dataset)}"),
        "split": split_name,
        "model": model_name,
        "tokenizer": {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "name_or_path": getattr(tokenizer, "name_or_path", model_name),
            "commit": tokenizer_options.get("_commit_hash"),
            "vocab_size": getattr(tokenizer, "vocab_size", None),
        },
        "transformers_version": _package_version("transformers"),
        "max_length": max_length,
        "labels": label_to_id,
        "padding": "dynamic",
        "version": 1,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _tokenize_dataset(
    dataset: Any,
    tokenizer: Any,
    label_to_id: dict[str, int],
    config: TransformerTrainingConfig,
    cache_dir: Path,
    split_name: str,
) -> Any:
    """Build or reuse a disk-backed, variable-length Arrow token cache."""
    fingerprint = _tokenization_fingerprint(
        dataset,
        split_name,
        config.model_name,
        tokenizer,
        config.max_length,
        label_to_id,
    )
    token_cache = cache_dir / f"{split_name}-tokens-{fingerprint}.arrow"
    return dataset.map(
        _tokenize_batch,
        batched=True,
        batch_size=config.tokenization_batch_size,
        writer_batch_size=config.tokenization_writer_batch_size,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label_to_id": label_to_id,
            "max_length": config.max_length,
        },
        remove_columns=list(dataset.column_names),
        keep_in_memory=False,
        load_from_cache_file=True,
        cache_file_name=str(token_cache),
        new_fingerprint=fingerprint,
        desc=f"Tokenizing {split_name}",
    )


def _dataframe_dataset(frame: pd.DataFrame, retain_metadata: bool) -> Any:
    """Adapt the smaller Stage-2/3 pandas inputs without changing their APIs."""
    _, Dataset, *_ = _dependencies()
    require_model_text(frame)
    if "canonical_label" not in frame:
        raise ValueError("Transformer input is missing canonical_label")
    columns = list(frame.columns) if retain_metadata else ["text", "canonical_label"]
    return Dataset.from_pandas(frame.loc[:, columns], preserve_index=False)


def _load_inputs(
    train: Any,
    validation: Any,
    cache_dir: Path,
    max_train_samples: int | None,
    seed: int,
) -> tuple[Any, Any, bool]:
    """Load projected, disk-backed Stage-1 data or adapt existing in-memory inputs."""
    if isinstance(train, (str, Path)) and isinstance(validation, (str, Path)):
        log_memory(LOGGER, "before dataset load")
        train_dataset = load_parquet_dataset(
            Path(train), ["text", "canonical_label"], cache_dir / "raw-train"
        )
        validation_dataset = load_parquet_dataset(
            Path(validation),
            ["issue_id", "text", "canonical_label"],
            cache_dir / "raw-validation",
        )
        log_memory(
            LOGGER,
            "after HF Dataset construction",
            train_rows=len(train_dataset),
            validation_rows=len(validation_dataset),
        )
        train_dataset = stratified_limit(train_dataset, max_train_samples, seed, cache_dir, "train")
        log_memory(
            LOGGER,
            "after sampling",
            train_rows=len(train_dataset),
            validation_rows=len(validation_dataset),
        )
        return train_dataset, validation_dataset, False

    if isinstance(train, pd.DataFrame) and isinstance(validation, pd.DataFrame):
        log_memory(LOGGER, "before Dataset construction")
        train_dataset = _dataframe_dataset(train, retain_metadata=False)
        validation_dataset = _dataframe_dataset(validation, retain_metadata=True)
        log_memory(
            LOGGER,
            "after HF Dataset construction",
            train_rows=len(train_dataset),
            validation_rows=len(validation_dataset),
        )
        return train_dataset, validation_dataset, True

    return train, validation, False


def _balanced_weights(labels: list[str], counts: dict[str, int]) -> np.ndarray:
    total = sum(counts.values())
    return np.asarray([total / (len(labels) * counts[label]) for label in labels], dtype=np.float64)


def train_transformer(
    train: Any,
    validation: Any,
    output_dir: str | Path,
    config: TransformerTrainingConfig,
    resume_from_checkpoint: str | None = None,
    max_train_samples: int | None = None,
    stop_after_steps: int | None = None,
    skip_final_evaluation: bool = False,
) -> dict[str, Any]:
    """Train and save a weighted-loss Transformer, selecting best macro-F1 checkpoint."""
    (
        torch,
        _,
        AutoModel,
        AutoTokenizer,
        DataCollator,
        EarlyStopping,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    ) = _dependencies()
    if config.tokenization_batch_size <= 0 or config.tokenization_writer_batch_size <= 0:
        raise ValueError("Tokenization batch sizes must be positive")
    if config.dataloader_num_workers < 0:
        raise ValueError("dataloader_num_workers cannot be negative")
    if config.checkpoint_steps is not None and config.checkpoint_steps <= 0:
        raise ValueError("checkpoint_steps must be positive when set")
    if config.save_total_limit is not None and config.save_total_limit <= 0:
        raise ValueError("save_total_limit must be positive when set")
    if stop_after_steps is not None and stop_after_steps <= 0:
        raise ValueError("stop_after_steps must be positive")
    if stop_after_steps is not None and config.checkpoint_steps is None:
        raise ValueError("stop_after_steps requires checkpoint_steps in the training config")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        Path(config.dataset_cache_dir)
        if config.dataset_cache_dir
        else destination.parent / ".bugclassinet-cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    train_path = _source_path(train)
    validation_path = _source_path(validation)
    train_source_identity = (
        _source_file_identity(train_path, ["text", "canonical_label"]) if train_path else None
    )
    validation_source_identity = (
        _source_file_identity(validation_path, ["issue_id", "text", "canonical_label"])
        if validation_path
        else None
    )
    train, validation, preserve_validation_text = _load_inputs(
        train,
        validation,
        cache_dir,
        max_train_samples,
        config.seed,
    )
    train_counts = inspect_dataset(train, "train", required_columns={"text", "canonical_label"})
    validation_counts = inspect_dataset(
        validation, "validation", required_columns={"text", "canonical_label"}
    )
    labels = sorted(train_counts)
    if set(validation_counts) - set(labels):
        raise ValueError("Validation has labels not represented in training")
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    train_fingerprint = _dataset_fingerprint(train)
    validation_fingerprint = _dataset_fingerprint(validation)
    train_schema_columns = list(train.column_names)
    validation_schema_columns = list(validation.column_names)
    LOGGER.info("Stage-1 training class counts=%s", train_counts)
    LOGGER.info("Stage-1 validation class counts=%s", validation_counts)

    # DeBERTa-v3 ships a SentencePiece model. Avoid fast-tokenizer conversion,
    # which can incorrectly attempt to parse it as a tiktoken BPE file.
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False)
    revision = _model_revision(tokenizer)
    tokenized_train = _tokenize_dataset(train, tokenizer, label_to_id, config, cache_dir, "train")
    tokenized_validation = _tokenize_dataset(
        validation, tokenizer, label_to_id, config, cache_dir, "validation"
    )
    log_memory(
        LOGGER,
        "after tokenization/setup",
        train_rows=len(tokenized_train),
        validation_rows=len(tokenized_validation),
    )

    # Path-backed Stage 1 no longer needs raw text: both token tables are
    # memory-mapped. The pandas adapter preserves Stage-2/3 prediction schemas.
    prediction_columns = [
        column for column in validation.column_names if preserve_validation_text or column != "text"
    ]
    validation_metadata = validation.select_columns(prediction_columns)
    del train, validation
    gc.collect()

    use_fp16 = bool(torch.cuda.is_available())
    model = AutoModel.from_pretrained(
        config.model_name, num_labels=len(labels), label2id=label_to_id, id2label=id_to_label
    )
    if use_fp16:
        # AMP's GradScaler requires trainable master parameters and gradients in
        # FP32. The Microsoft checkpoint itself may contain FP16 tensors.
        model = model.float()
    balanced_weights = _balanced_weights(labels, train_counts)
    weights_tensor = torch.tensor(balanced_weights, dtype=torch.float)
    class_weights = balanced_weights.tolist()

    class WeightedTrainer(Trainer):
        def compute_loss(
            self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **_: Any
        ) -> Any:
            labels_tensor = inputs.pop("labels")
            outputs = model(**inputs)
            class_weights = weights_tensor.to(
                device=outputs.logits.device, dtype=outputs.logits.dtype
            )
            loss = torch.nn.CrossEntropyLoss(weight=class_weights)(outputs.logits, labels_tensor)
            return (loss, outputs) if return_outputs else loss

        def _save_checkpoint(self, model: Any, trial: Any) -> None:
            super()._save_checkpoint(model, trial)
            scaler = getattr(self.accelerator, "scaler", None)
            if scaler is not None and self.args.should_save:
                checkpoint_dir = destination / f"checkpoint-{self.state.global_step}"
                torch.save(scaler.state_dict(), checkpoint_dir / "scaler.pt")

        def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
            super()._load_optimizer_and_scheduler(checkpoint)
            scaler = getattr(self.accelerator, "scaler", None)
            if scaler is not None and checkpoint is not None:
                scaler_state = torch.load(Path(checkpoint) / "scaler.pt", map_location="cpu")
                scaler.load_state_dict(scaler_state)

    step_checkpointing = config.checkpoint_steps is not None
    arguments = TrainingArguments(
        output_dir=str(destination),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_strategy="no" if step_checkpointing else "epoch",
        save_strategy="steps" if step_checkpointing else "epoch",
        save_steps=config.checkpoint_steps or 500,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=not step_checkpointing,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_fp16,
        logging_strategy="epoch",
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
        max_steps=config.max_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_persistent_workers=False,
        ignore_data_skip=False,
    )

    config_for_hash = asdict(config)
    config_for_hash.pop("dataset_cache_dir", None)
    training_settings = {
        "max_length": config.max_length,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_steps": config.max_steps,
        "fp16": use_fp16,
        "n_gpu": int(getattr(arguments, "n_gpu", 0)),
        "world_size": int(getattr(arguments, "world_size", 1)),
        "data_seed": config.seed,
        "ignore_data_skip": False,
        "torch_version": str(getattr(torch, "__version__", "unavailable")),
        "transformers_version": _package_version("transformers"),
        "datasets_version": _package_version("datasets"),
    }
    manifest: dict[str, Any] = {
        "dataset_path": train_path,
        "dataset_fingerprint": train_fingerprint,
        "train_hf_fingerprint": train_fingerprint,
        "train_source_sha256": (train_source_identity["sha256"] if train_source_identity else None),
        "validation_path": validation_path,
        "validation_fingerprint": validation_fingerprint,
        "validation_hf_fingerprint": validation_fingerprint,
        "validation_source_sha256": (
            validation_source_identity["sha256"] if validation_source_identity else None
        ),
        "training_rows": len(tokenized_train),
        "train_rows": len(tokenized_train),
        "validation_rows": len(tokenized_validation),
        "train_class_counts": train_counts,
        "validation_class_counts": validation_counts,
        "class_counts": train_counts,
        "train_schema_columns": (
            train_source_identity["schema_columns"]
            if train_source_identity
            else train_schema_columns
        ),
        "validation_schema_columns": (
            validation_source_identity["schema_columns"]
            if validation_source_identity
            else validation_schema_columns
        ),
        "train_required_columns": ["text", "canonical_label"],
        "validation_required_columns": ["issue_id", "text", "canonical_label"],
        "train_checksum_provenance": (
            train_source_identity["checksum_provenance"] if train_source_identity else []
        ),
        "validation_checksum_provenance": (
            validation_source_identity["checksum_provenance"] if validation_source_identity else []
        ),
        "preprocessing_version": _PREPROCESSING_VERSION,
        "seed": config.seed,
        "model_name": config.model_name,
        "model_revision": revision,
        "config_hash": _json_hash(config_for_hash),
        "label_mapping": label_to_id,
        "label_mapping_hash": _json_hash(label_to_id),
        "class_weights": class_weights,
        "training_settings": training_settings,
        "expected_total_optimizer_steps": None,
        "starting_global_step": 0,
        "ending_global_step": 0,
        "resumed_checkpoint_path": None,
        "status": "initialized",
    }
    resumed_expected_steps = None
    normalized_checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint, stored_manifest = _require_resume_checkpoint(resume_from_checkpoint, use_fp16)
        _validate_resume_manifest(stored_manifest, manifest, checkpoint)
        resumed_expected_steps = stored_manifest.get("expected_total_optimizer_steps")
        if not isinstance(resumed_expected_steps, int) or resumed_expected_steps <= 0:
            raise ValueError("Checkpoint manifest has no valid expected_total_optimizer_steps")
        normalized_checkpoint = str(checkpoint)
        manifest["resumed_checkpoint_path"] = normalized_checkpoint

    (destination / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )

    def compute_metrics(prediction: Any) -> dict[str, float]:
        predicted = np.argmax(prediction.predictions, axis=1)
        truth = [id_to_label[int(value)] for value in prediction.label_ids]
        output = [id_to_label[int(value)] for value in predicted]
        metrics = classification_metrics(truth, output)
        return {
            key: float(value)
            for key, value in metrics.items()
            if key.endswith("_f1") or key == "mcc"
        }

    data_collator = DataCollator(tokenizer=tokenizer, padding=True, return_tensors="pt")
    execution_callback = _make_execution_callback(
        TrainerCallback,
        manifest,
        destination,
        stop_after_steps,
        resumed_expected_steps,
        force_final_checkpoint=step_checkpointing,
    )
    callbacks = [execution_callback]
    if not step_checkpointing:
        callbacks.append(EarlyStopping(early_stopping_patience=config.early_stopping_patience))
    else:
        LOGGER.info(
            "Full-state checkpoints enabled every %s optimizer steps (save_total_limit=%s)",
            config.checkpoint_steps,
            config.save_total_limit,
        )
    trainer = WeightedTrainer(
        model=model,
        args=arguments,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    n_gpu = int(getattr(arguments, "n_gpu", 0))
    LOGGER.info(
        "Trainer devices: n_gpu=%s world_size=%s parallel_mode=%s effective_batch_size=%s",
        n_gpu,
        getattr(arguments, "world_size", 1),
        getattr(arguments, "parallel_mode", "unknown"),
        config.batch_size * max(n_gpu, 1) * config.gradient_accumulation_steps,
    )
    log_memory(LOGGER, "immediately before Trainer.train()")
    trainer.train(resume_from_checkpoint=normalized_checkpoint)
    if execution_callback.stopped_early and execution_callback.final_checkpoint is None:
        raise RuntimeError("Segment stopped without producing a resumable checkpoint")

    if execution_callback.stopped_early or skip_final_evaluation:
        return {
            "status": manifest["status"],
            "starting_global_step": manifest["starting_global_step"],
            "ending_global_step": manifest["ending_global_step"],
            "expected_total_optimizer_steps": manifest["expected_total_optimizer_steps"],
            "checkpoint": execution_callback.final_checkpoint,
        }

    # Drop direct references we control before the full validation pass. Trainer
    # may retain its prepared loader, but its token table is disk-backed rather
    # than a Python-list or in-memory Arrow duplicate.
    trainer.train_dataset = None
    del tokenized_train
    gc.collect()
    log_memory(LOGGER, "before evaluation", validation_rows=len(tokenized_validation))
    metrics = trainer.evaluate()
    trainer.save_model(destination)
    tokenizer.save_pretrained(destination)
    (destination / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float) + "\n", encoding="utf-8"
    )

    # Reuse the exact evaluation token table; do not tokenize validation twice.
    prediction = trainer.predict(tokenized_validation)
    predicted_labels = [
        id_to_label[int(value)] for value in np.argmax(prediction.predictions, axis=1)
    ]
    prediction_rows = validation_metadata.add_column("prediction", predicted_labels)
    prediction_rows.to_parquet(str(destination / "validation_predictions.parquet"))
    manifest["status"] = "completed"
    _write_run_manifest(destination / "run_manifest.json", manifest)
    return metrics
