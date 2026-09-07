"""Reusable, training-disabled inference runtime for frozen Stage-1 models."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bugclassinet.evaluation.reporting import _probabilities
from bugclassinet.evaluation.stage1 import (
    _assert_parameters_unchanged,
    _checkpoint_model_sha256,
    _evaluation_only_trainer_class,
    _label_mapping,
    _load_manifest,
    _model_parameter_sha256,
    _validate_complete_checkpoint,
)
from bugclassinet.utils.memory import log_memory


@dataclass
class FrozenPredictions:
    """Predictions and preserved row metadata from one inference pass."""

    metadata: pd.DataFrame
    true_labels: list[str]
    predicted_labels: list[str]
    logits: np.ndarray
    probabilities: np.ndarray
    eval_loss: float | None


class FrozenStage1Evaluator:
    """Load one complete checkpoint and expose prediction-only operations."""

    def __init__(
        self,
        model_dir: str | Path,
        work_dir: str | Path,
        batch_size: int | None = None,
    ) -> None:
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
                "Frozen Stage-1 evaluation requires `pip install -e .[transformers]`."
            ) from error

        from bugclassinet.models.transformer_classifier import (
            TransformerTrainingConfig,
            _restore_checkpoint_model_state,
        )

        self.torch = torch
        self.source = Path(model_dir).resolve()
        self.work_dir = Path(work_dir).resolve()
        if not self.source.is_dir():
            raise FileNotFoundError(f"Stage-1 model directory does not exist: {self.source}")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = _load_manifest(self.source)
        self.ending_global_step = _validate_complete_checkpoint(self.manifest, self.source)
        self.model_revision = self.manifest.get("model_revision")
        self.config_hash = self.manifest.get("config_hash")
        if not self.model_revision or not self.config_hash:
            raise ValueError("Frozen model manifest requires model_revision and config_hash")

        model_config = AutoConfig.from_pretrained(self.source)
        self.label_to_id = _label_mapping(model_config, self.manifest)
        self.id_to_label = {index: label for label, index in self.label_to_id.items()}
        self.score_labels = [self.id_to_label[index] for index in range(len(self.id_to_label))]
        self.model = AutoModelForSequenceClassification.from_config(model_config)
        _restore_checkpoint_model_state(self.model, self.source, torch)
        self.parameter_hash_before = _model_parameter_sha256(self.model, torch)
        self.checkpoint_hash = _checkpoint_model_sha256(self.source)

        tokenizer_source: str | Path = self.source
        if not (self.source / "tokenizer_config.json").is_file():
            parent = self.source.parent
            tokenizer_source = (
                parent
                if (parent / "tokenizer_config.json").is_file()
                else self.manifest.get(
                    "model_name", getattr(model_config, "_name_or_path", self.source)
                )
            )
        self.tokenizer_source = str(tokenizer_source)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
        stored_max_length = self.manifest.get("max_length") or self.manifest.get(
            "training_settings", {}
        ).get("max_length")
        if stored_max_length is None:
            raise ValueError("Frozen model manifest does not record max_length")
        self.max_length = int(stored_max_length)
        self.tokenization_config = TransformerTrainingConfig(
            model_name=str(
                self.manifest.get("model_name", getattr(model_config, "_name_or_path", self.source))
            ),
            max_length=self.max_length,
        )

        settings = self.manifest.get("training_settings", {})
        resolved_batch_size = int(batch_size or settings.get("batch_size", 8))
        if resolved_batch_size <= 0:
            raise ValueError("Evaluation batch size must be positive")
        self.batch_size = resolved_batch_size
        use_fp16 = bool(settings.get("fp16", False) and torch.cuda.is_available())
        arguments = TrainingArguments(
            output_dir=str(self.work_dir / ".trainer"),
            per_device_eval_batch_size=self.batch_size,
            fp16=use_fp16,
            report_to=[],
            seed=int(self.manifest.get("seed", 42)),
            dataloader_num_workers=0,
            dataloader_persistent_workers=False,
        )
        EvaluationOnlyTrainer = _evaluation_only_trainer_class(Trainer)
        self.trainer = EvaluationOnlyTrainer(
            model=self.model,
            args=arguments,
            data_collator=DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True,
                return_tensors="pt",
            ),
        )

    def predict(
        self,
        dataset: Any,
        split_name: str,
        metadata_columns: list[str],
    ) -> FrozenPredictions:
        """Run exactly one no-gradient prediction pass over a labeled dataset."""
        from bugclassinet.data.transformer import inspect_dataset
        from bugclassinet.models.transformer_classifier import _tokenize_dataset

        inspect_dataset(dataset, split_name, required_columns={"text", "canonical_label"})
        missing_metadata = sorted(set(metadata_columns) - set(dataset.column_names))
        if missing_metadata:
            raise ValueError(f"{split_name} is missing metadata columns: {missing_metadata}")
        metadata = dataset.select_columns(metadata_columns).to_pandas()
        tokenized = _tokenize_dataset(
            dataset,
            self.tokenizer,
            self.label_to_id,
            self.tokenization_config,
            self.work_dir / ".token-cache",
            split_name,
        )
        del dataset
        gc.collect()
        log_memory(
            __import__("logging").getLogger(__name__),
            f"before frozen Stage-1 prediction ({split_name})",
            rows=len(tokenized),
        )
        with self.torch.inference_mode():
            output = self.trainer.predict(tokenized)
        logits = np.asarray(output.predictions, dtype=np.float32)
        expected_shape = (len(tokenized), len(self.score_labels))
        if logits.shape != expected_shape:
            raise ValueError(
                "Unexpected Stage-1 prediction shape: "
                f"expected={expected_shape}, actual={logits.shape}"
            )
        truth = metadata["canonical_label"].astype(str).tolist()
        predicted = [self.id_to_label[int(index)] for index in np.argmax(logits, axis=1)]
        prediction_metrics = dict(getattr(output, "metrics", {}) or {})
        eval_loss = prediction_metrics.get("test_loss")
        del tokenized, output
        gc.collect()
        return FrozenPredictions(
            metadata=metadata,
            true_labels=truth,
            predicted_labels=predicted,
            logits=logits,
            probabilities=_probabilities(logits),
            eval_loss=float(eval_loss) if eval_loss is not None else None,
        )

    def verify_unchanged(self) -> str:
        """Assert that inference left every named model parameter untouched."""
        after = _model_parameter_sha256(self.model, self.torch)
        _assert_parameters_unchanged(self.parameter_hash_before, after)
        return after

    def identity(self) -> dict[str, Any]:
        """Return reproducibility and immutability fields shared by manifests."""
        return {
            "model_path": str(self.source),
            "checkpoint_sha256": self.checkpoint_hash,
            "model_name": self.manifest.get("model_name"),
            "model_revision": self.model_revision,
            "config_hash": self.config_hash,
            "label_mapping": self.label_to_id,
            "preprocessing_version": self.manifest.get("preprocessing_version"),
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "ending_global_step": self.ending_global_step,
            "model_parameter_sha256_before": self.parameter_hash_before,
            "training_invoked": False,
        }
