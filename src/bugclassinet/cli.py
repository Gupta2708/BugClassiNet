"""Command-line interface for reproducible BugClassiNet workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from bugclassinet.data.archive import inspect_archive
from bugclassinet.data.nlbse import prepare_nlbse
from bugclassinet.data.validate import validate_nlbse_variants, validate_parquet
from bugclassinet.utils.logging import configure_logging


def _inspect(args: argparse.Namespace) -> None:
    print(json.dumps(inspect_archive(args.archive), indent=2))


def _prepare(args: argparse.Namespace) -> None:
    result = prepare_nlbse(
        args.train_archive,
        args.test_archive,
        args.output_dir,
        args.sample_dir,
        args.validation_fraction,
        args.seed,
        args.sample_size,
    )
    print(json.dumps(result, indent=2, default=str))


def _validate(args: argparse.Namespace) -> None:
    source = Path(args.path)
    if source.is_dir():
        print(json.dumps(validate_nlbse_variants(source), indent=2))
    else:
        frame = validate_parquet(source)
        print(f"Validated {source}: {len(frame)} rows")


def _sample(args: argparse.Namespace) -> None:
    from bugclassinet.data.nlbse import _balanced_sample

    frame = validate_parquet(args.input)
    sample = _balanced_sample(frame, args.size, args.seed)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(target, index=False)
    print(f"Wrote {len(sample)} sampled rows to {target}")


def _deferred(name: str) -> Callable[[argparse.Namespace], None]:
    def run(args: argparse.Namespace) -> None:
        from bugclassinet.training import commands

        getattr(commands, name)(args)

    return run


def build_parser() -> argparse.ArgumentParser:
    """Build the documented CLI parser."""
    parser = argparse.ArgumentParser(prog="python -m bugclassinet.cli")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-archive")
    inspect.add_argument("archive")
    inspect.set_defaults(func=_inspect)
    prepare = commands.add_parser("prepare-nlbse")
    prepare.add_argument("--train-archive", required=True)
    prepare.add_argument("--test-archive", required=True)
    prepare.add_argument("--output-dir", default="data/processed/nlbse2023")
    prepare.add_argument("--sample-dir", default="data/samples")
    prepare.add_argument("--validation-fraction", type=float, default=0.15)
    prepare.add_argument("--sample-size", type=int, default=20_000)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(func=_prepare)
    validate = commands.add_parser("validate-data")
    validate.add_argument("path")
    validate.set_defaults(func=_validate)
    sample = commands.add_parser("create-sample")
    sample.add_argument("--input", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--size", type=int, default=20_000)
    sample.add_argument("--seed", type=int, default=42)
    sample.set_defaults(func=_sample)
    for command, handler in {
        "train-tfidf": "train_tfidf",
        "train-stage1": "train_stage1",
        "evaluate-stage1": "evaluate_stage1",
        "train-stage2": "train_stage2",
        "train-stage3": "train_stage3",
        "train-dapt": "train_dapt",
        "evaluate-hierarchy": "evaluate_hierarchy",
    }.items():
        sub = commands.add_parser(command)
        sub.add_argument("--config")
        sub.add_argument("--data-dir")
        sub.add_argument("--train")
        sub.add_argument("--validation")
        sub.add_argument("--test")
        sub.add_argument("--model-path")
        sub.add_argument("--output-dir", default="outputs")
        sub.add_argument("--checkpoint")
        sub.add_argument("--threshold", type=float, default=0.0)
        sub.add_argument("--max-train-samples", type=int)
        sub.add_argument("--max-eval-samples", type=int)
        sub.add_argument("--max-steps", type=int)
        sub.set_defaults(func=_deferred(handler))
    return parser


def main(argv: list[str] | None = None) -> None:
    """Execute a CLI command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
