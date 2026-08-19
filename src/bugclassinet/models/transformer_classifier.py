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

from bugclassinet.data.transformer import (
    inspect_dataset,
    load_parquet_dataset,
    stratified_limit,
)
from bugclassinet.evaluation.metrics import classification_metrics
from bugclassinet.features.text import require_model_text
from bugclassinet.utils.memory import log_memory
from bugclassinet.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unavailable"


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
        TrainingArguments,
    ) = _dependencies()
    if config.tokenization_batch_size <= 0 or config.tokenization_writer_batch_size <= 0:
        raise ValueError("Tokenization batch sizes must be positive")
    if config.dataloader_num_workers < 0:
        raise ValueError("dataloader_num_workers cannot be negative")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        Path(config.dataset_cache_dir)
        if config.dataset_cache_dir
        else destination.parent / ".bugclassinet-cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
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
    LOGGER.info("Stage-1 training class counts=%s", train_counts)
    LOGGER.info("Stage-1 validation class counts=%s", validation_counts)

    # DeBERTa-v3 ships a SentencePiece model. Avoid fast-tokenizer conversion,
    # which can incorrectly attempt to parse it as a tiktoken BPE file.
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False)
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
    weights_tensor = torch.tensor(_balanced_weights(labels, train_counts), dtype=torch.float)

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

    arguments = TrainingArguments(
        output_dir=str(destination),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_fp16,
        logging_strategy="epoch",
        report_to=[],
        seed=config.seed,
        max_steps=config.max_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_persistent_workers=False,
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
    trainer = WeightedTrainer(
        model=model,
        args=arguments,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStopping(early_stopping_patience=config.early_stopping_patience)],
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
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

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
    (destination / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
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
    return metrics
