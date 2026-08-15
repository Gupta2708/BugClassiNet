"""Reusable Hugging Face sequence-classification trainer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from bugclassinet.evaluation.metrics import classification_metrics
from bugclassinet.features.text import require_model_text
from bugclassinet.utils.seed import set_seed


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


def _dependencies() -> tuple[Any, ...]:
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
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
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )


def _dataset(
    frame: pd.DataFrame, tokenizer: Any, label_to_id: dict[str, int], max_length: int
) -> Any:
    _, Dataset, *_ = _dependencies()
    text = require_model_text(frame).tolist()
    encoded = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
    encoded["labels"] = [label_to_id[label] for label in frame["canonical_label"]]
    return Dataset.from_dict(encoded)


def train_transformer(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: str | Path,
    config: TransformerTrainingConfig,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Train and save a weighted-loss Transformer, selecting best macro-F1 checkpoint."""
    torch, _, AutoModel, AutoTokenizer, EarlyStopping, Trainer, TrainingArguments = _dependencies()
    for frame_name, frame in (("train", train), ("validation", validation)):
        if "canonical_label" not in frame:
            raise ValueError(f"{frame_name} is missing canonical_label")
        require_model_text(frame)
    set_seed(config.seed)
    labels = sorted(train["canonical_label"].unique())
    if set(validation["canonical_label"]) - set(labels):
        raise ValueError("Validation has labels not represented in training")
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    # DeBERTa-v3 ships a SentencePiece model.  Avoid fast-tokenizer conversion,
    # which can incorrectly attempt to parse it as a tiktoken BPE file.

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False)
    model = AutoModel.from_pretrained(
        config.model_name, num_labels=len(labels), label2id=label_to_id, id2label=id_to_label
    )
    weights = compute_class_weight("balanced", classes=np.array(labels), y=train["canonical_label"])
    weights_tensor = torch.tensor(weights, dtype=torch.float)

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

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    use_fp16 = bool(torch.cuda.is_available())
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

    trainer = WeightedTrainer(
        model=model,
        args=arguments,
        train_dataset=_dataset(train, tokenizer, label_to_id, config.max_length),
        eval_dataset=_dataset(validation, tokenizer, label_to_id, config.max_length),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStopping(early_stopping_patience=config.early_stopping_patience)],
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = trainer.evaluate()
    trainer.save_model(destination)
    tokenizer.save_pretrained(destination)
    (destination / "training_config.json").write_text(
        json.dumps(config.__dict__, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float) + "\n", encoding="utf-8"
    )
    prediction = trainer.predict(_dataset(validation, tokenizer, label_to_id, config.max_length))
    rows = validation.copy()
    rows["prediction"] = [
        id_to_label[int(value)] for value in np.argmax(prediction.predictions, axis=1)
    ]
    rows.to_parquet(destination / "validation_predictions.parquet", index=False)
    return metrics
