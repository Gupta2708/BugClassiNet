"""DAPT package entry point."""

from __future__ import annotations

import pandas as pd

from bugclassinet.models.dapt import train_dapt
from bugclassinet.training.common import training_config


def run(train: str | None, output_dir: str, config_path: str | None) -> dict[str, object]:
    """Read an explicit unlabelled corpus Parquet and run DAPT."""
    if not train:
        raise ValueError("train-dapt requires --train unlabelled-corpus.parquet")
    corpus = pd.read_parquet(train)
    config = training_config(config_path, {})
    return train_dapt(corpus, output_dir, **config)
