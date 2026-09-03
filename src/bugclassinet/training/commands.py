"""Adapters between argparse namespaces and reusable trainers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bugclassinet.evaluation.evaluate import evaluate_classifier
from bugclassinet.evaluation.hierarchy import evaluate_hierarchy as hierarchy_metrics
from bugclassinet.evaluation.stage1 import evaluate_saved_stage1
from bugclassinet.models.tfidf_classifier import load_tfidf
from bugclassinet.training import train_dapt as dapt
from bugclassinet.training import train_stage1 as stage1
from bugclassinet.training import train_stage2 as stage2
from bugclassinet.training import train_stage3 as stage3
from bugclassinet.training import train_tfidf as tfidf
from bugclassinet.utils.io import write_json


def _print(result: Any) -> None:
    print(json.dumps(result, indent=2, default=str))


def train_tfidf(args: Any) -> None:
    if args.max_eval_samples is not None:
        raise ValueError("TF-IDF evaluation always uses the complete validation split")
    _print(
        tfidf.run(
            args.data_dir,
            args.train,
            args.validation,
            args.output_dir,
            args.config,
            args.max_train_samples,
        )
    )


def train_stage1(args: Any) -> None:
    if args.checkpoint and args.resume_from_checkpoint:
        raise ValueError("Use only --resume-from-checkpoint; do not also pass --checkpoint")
    resume_from_checkpoint = args.resume_from_checkpoint or args.checkpoint
    _print(
        stage1.run(
            args.data_dir,
            args.train,
            args.validation,
            args.output_dir,
            args.config,
            resume_from_checkpoint,
            args.max_train_samples,
            args.max_eval_samples,
            args.max_steps,
            args.stop_after_steps,
            args.skip_final_evaluation,
        )
    )


def train_stage2(args: Any) -> None:
    _print(
        stage2.run(
            args.data_dir,
            args.train,
            args.validation,
            args.output_dir,
            args.config,
            args.checkpoint,
        )
    )


def train_stage3(args: Any) -> None:
    _print(
        stage3.run(
            args.data_dir,
            args.train,
            args.validation,
            args.output_dir,
            args.config,
            args.checkpoint,
        )
    )


def train_dapt(args: Any) -> None:
    _print(dapt.run(args.train, args.output_dir, args.config))


def evaluate_stage1(args: Any) -> None:
    if not args.model_path or not args.test:
        raise ValueError("evaluate-stage1 requires --model/--model-path and --data/--test")
    model_path = Path(args.model_path)
    if model_path.is_file():
        result = evaluate_classifier(
            load_tfidf(model_path), pd.read_parquet(args.test), args.output_dir
        )
    else:
        result = evaluate_saved_stage1(model_path, args.test, args.output_dir)
    _print(result)


def evaluate_hierarchy(args: Any) -> None:
    if not args.test:
        raise ValueError("evaluate-hierarchy requires --test prediction parquet")
    metrics = hierarchy_metrics(pd.read_parquet(args.test))
    write_json(Path(args.output_dir) / "hierarchy_metrics.json", metrics)
    _print(metrics)
