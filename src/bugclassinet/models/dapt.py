"""Domain-adaptive masked-language-model pretraining."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from bugclassinet.utils.seed import set_seed


def train_dapt(
    corpus: pd.DataFrame,
    output_dir: str | Path,
    model_name: str = "answerdotai/ModernBERT-base",
    max_length: int = 256,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    epochs: int = 3,
    seed: int = 42,
    held_out_projects: set[str] | None = None,
    strict_exclude_eval_projects: bool = False,
) -> dict[str, Any]:
    """DAPT on unlabelled text only, retaining a held-out domain validation set."""
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForMaskedLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise ImportError(
            "DAPT requires `pip install -e .[transformers]` and `datasets`."
        ) from error
    if "text" not in corpus:
        raise ValueError("DAPT corpus requires an unlabelled text column")
    if strict_exclude_eval_projects:
        if "project" not in corpus or held_out_projects is None:
            raise ValueError("Strict DAPT requires project column and held_out_projects")
        corpus = corpus.loc[~corpus["project"].isin(held_out_projects)].copy()
    if corpus.empty:
        raise ValueError("No DAPT rows remain after exclusions")
    set_seed(seed)
    train_text, validation_text = train_test_split(
        corpus["text"].astype(str).tolist(), test_size=0.1, random_state=seed
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(texts: list[str]) -> Any:
        return Dataset.from_dict(
            tokenizer(texts, truncation=True, max_length=max_length, padding="max_length")
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(destination),
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=[],
        fp16=torch.cuda.is_available(),
        seed=seed,
    )
    trainer = Trainer(
        model=AutoModelForMaskedLM.from_pretrained(model_name),
        args=arguments,
        train_dataset=tokenize(train_text),
        eval_dataset=tokenize(validation_text),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15),
    )
    trainer.train()
    trainer.save_model(destination)
    tokenizer.save_pretrained(destination)
    manifest = {
        "base_model": model_name,
        "train_rows": len(train_text),
        "validation_rows": len(validation_text),
        "strict_exclude_eval_projects": strict_exclude_eval_projects,
        "excluded_projects": sorted(held_out_projects or []),
    }
    (destination / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
