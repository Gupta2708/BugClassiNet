"""Standalone evaluation for saved Stage-1 Transformer models."""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from bugclassinet.data.transformer import inspect_dataset, load_parquet_dataset
from bugclassinet.evaluation.reporting import STAGE1_REPORT_LABELS, write_stage1_evaluation
from bugclassinet.utils.memory import log_memory

LOGGER = logging.getLogger(__name__)


def _load_manifest(model_dir: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for candidate in (model_dir / "run_manifest.json", model_dir.parent / "run_manifest.json"):
        if candidate.is_file():
            manifest = json.loads(candidate.read_text(encoding="utf-8"))
            break
    for candidate in (
        model_dir / "training_config.json",
        model_dir.parent / "training_config.json",
    ):
        if candidate.is_file():
            training_config = json.loads(candidate.read_text(encoding="utf-8"))
            for key in ("model_name", "max_length", "batch_size", "seed"):
                if key in training_config:
                    manifest.setdefault(key, training_config[key])
            break
    return manifest


def _label_mapping(model_config: Any, manifest: dict[str, Any]) -> dict[str, int]:
    stored = manifest.get("label_mapping")
    raw = stored if isinstance(stored, dict) else getattr(model_config, "label2id", {})
    mapping = {str(label): int(index) for label, index in raw.items()}
    if set(mapping) != set(STAGE1_REPORT_LABELS):
        raise ValueError(
            "Saved Stage-1 model has an incompatible label mapping: "
            f"expected={list(STAGE1_REPORT_LABELS)}, actual={mapping}"
        )
    if sorted(mapping.values()) != list(range(len(STAGE1_REPORT_LABELS))):
        raise ValueError(f"Saved Stage-1 label IDs must be contiguous: {mapping}")
    return mapping


def evaluate_saved_stage1(
    model_dir: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Load a trained Stage-1 model and create reports using one prediction pass."""
    try:
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise ImportError(
            "Stage-1 evaluation requires `pip install -e .[transformers]`."
        ) from error

    from bugclassinet.models.transformer_classifier import (
        TransformerTrainingConfig,
        _restore_checkpoint_model_state,
        _tokenize_dataset,
    )

    source = Path(model_dir).resolve()
    data = Path(data_path).resolve()
    destination = Path(output_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Stage-1 model directory does not exist: {source}")
    if not data.is_file():
        raise FileNotFoundError(f"Stage-1 evaluation data does not exist: {data}")
    destination.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(source)
    model_config = AutoConfig.from_pretrained(source)
    label_to_id = _label_mapping(model_config, manifest)
    id_to_label = {index: label for label, index in label_to_id.items()}
    model = AutoModelForSequenceClassification.from_config(model_config)
    _restore_checkpoint_model_state(model, source, torch)

    tokenizer_source: str | Path = source
    if not (source / "tokenizer_config.json").is_file():
        parent = source.parent
        tokenizer_source = (
            parent
            if (parent / "tokenizer_config.json").is_file()
            else manifest.get("model_name", getattr(model_config, "_name_or_path", source))
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)

    schema = pq.read_schema(data).names
    projected_columns = ["text", "canonical_label"]
    if "issue_id" in schema:
        projected_columns.insert(0, "issue_id")
    cache_dir = destination.parent / ".bugclassinet-cache" / "standalone-evaluation"
    log_memory(LOGGER, "before evaluation dataset load")
    validation = load_parquet_dataset(data, projected_columns, cache_dir / "raw")
    counts = inspect_dataset(
        validation,
        "validation",
        required_columns={"text", "canonical_label"},
    )
    if set(counts) - set(label_to_id):
        raise ValueError(f"Evaluation contains labels absent from the model: {counts}")

    max_length = int(
        manifest.get("max_length") or manifest.get("training_settings", {}).get("max_length") or 256
    )
    training_config = TransformerTrainingConfig(
        model_name=str(manifest.get("model_name", getattr(model_config, "_name_or_path", source))),
        max_length=max_length,
    )
    metadata = validation.select_columns(
        [column for column in ("issue_id", "canonical_label") if column in validation.column_names]
    )
    tokenized = _tokenize_dataset(
        validation,
        tokenizer,
        label_to_id,
        training_config,
        cache_dir / "tokens",
        "standalone-validation",
    )
    del validation
    gc.collect()
    log_memory(LOGGER, "after standalone evaluation tokenization", validation_rows=len(tokenized))

    if "cross_entropy_weighted" in manifest:
        weighted = bool(manifest["cross_entropy_weighted"])
        stored_weights = manifest.get("cross_entropy_class_weights")
    elif "class_weights" in manifest:
        # Manifests predating the explicit strategy field always used balanced
        # cross-entropy and stored its ordered values as class_weights.
        weighted = True
        stored_weights = manifest.get("class_weights")
    else:
        weighted = False
        stored_weights = None
        LOGGER.warning(
            "Saved model has no loss-weight manifest; standalone eval_loss will use "
            "unweighted cross-entropy. Prediction-based metrics are unaffected."
        )
    if weighted and stored_weights is None:
        raise ValueError("Saved weighted-loss model manifest has no class weights")
    weights_tensor = (
        torch.tensor(stored_weights, dtype=torch.float)
        if weighted and stored_weights is not None
        else None
    )

    class EvaluationTrainer(Trainer):
        def compute_loss(
            self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **_: Any
        ) -> Any:
            labels_tensor = inputs.pop("labels")
            outputs = model(**inputs)
            loss_weights = (
                None
                if weights_tensor is None
                else weights_tensor.to(outputs.logits.device, dtype=outputs.logits.dtype)
            )
            loss = torch.nn.CrossEntropyLoss(weight=loss_weights)(outputs.logits, labels_tensor)
            return (loss, outputs) if return_outputs else loss

    settings = manifest.get("training_settings", {})
    use_fp16 = bool(settings.get("fp16", False) and torch.cuda.is_available())
    arguments = TrainingArguments(
        output_dir=str(destination / ".trainer-evaluation"),
        per_device_eval_batch_size=int(settings.get("batch_size", manifest.get("batch_size", 8))),
        fp16=use_fp16,
        report_to=[],
        seed=int(manifest.get("seed", 42)),
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
    )
    trainer = EvaluationTrainer(
        model=model,
        args=arguments,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        ),
    )
    log_memory(LOGGER, "before standalone Trainer.predict", validation_rows=len(tokenized))
    prediction = trainer.predict(tokenized)
    logits = np.asarray(prediction.predictions)
    score_labels = [id_to_label[index] for index in range(len(id_to_label))]
    if logits.shape != (len(tokenized), len(score_labels)):
        raise ValueError(
            "Unexpected Stage-1 prediction shape: "
            f"expected={(len(tokenized), len(score_labels))}, actual={logits.shape}"
        )
    truth = list(metadata["canonical_label"])
    predicted = [id_to_label[int(index)] for index in np.argmax(logits, axis=1)]
    prediction_metrics = dict(getattr(prediction, "metrics", {}) or {})
    eval_loss = prediction_metrics.get("test_loss")
    return write_stage1_evaluation(
        destination / "evaluation",
        truth,
        predicted,
        label_to_id,
        eval_loss=float(eval_loss) if eval_loss is not None else None,
        logits=logits,
        score_label_order=score_labels,
        issue_ids=list(metadata["issue_id"]) if "issue_id" in metadata.column_names else None,
    )
