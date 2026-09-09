"""CLI registration isolated from frozen Stage-1 commands."""

from __future__ import annotations

import argparse
import json


def dispatch(args: argparse.Namespace) -> None:
    if args.command == "mandelbugs-audit":
        from bugclassinet.data.mandelbugs import audit_mandelbugs

        result = audit_mandelbugs(args.raw_dir, args.output_dir)
        result = {k: v for k, v in result.items() if k != "warnings"}
    elif args.command == "mandelbugs-enrich":
        from bugclassinet.data.mandelbugs_enrich import enrich_mandelbugs

        result = enrich_mandelbugs(
            args.labels,
            args.output_dir,
            args.cache_dir,
            args.sleep_seconds,
            args.manual_dir,
            args.retry_failures,
            args.offline,
            args.prior_reports,
            args.only_issue,
            args.only_project,
            args.allow_web_archive,
        )
    elif args.command == "mandelbugs-prepare":
        from bugclassinet.data.mandelbugs_prepare import prepare_mandelbugs

        result = prepare_mandelbugs(args.labels, args.reports, args.output_dir)
    elif args.command == "train-stage2-baseline":
        from bugclassinet.training.stage2_baseline import train_baseline

        result = train_baseline(
            args.data,
            args.model,
            args.evidence_mode,
            args.output_dir,
            args.config,
            args.cache_dir,
            args.stage1_model,
        )
    elif args.command == "train-stage2-modernbert":
        from bugclassinet.training.stage2_modernbert import train_modernbert

        result = train_modernbert(
            args.data, args.evidence_mode, args.output_dir, args.seeds, args.config
        )
    else:
        raise ValueError(f"Unknown Stage-2 command {args.command}")
    print(json.dumps(result, indent=2, default=str))


def register_commands(commands: argparse._SubParsersAction) -> None:
    for name in (
        "mandelbugs-audit",
        "mandelbugs-enrich",
        "mandelbugs-prepare",
        "train-stage2-baseline",
        "train-stage2-modernbert",
    ):
        sub = commands.add_parser(name)
        sub.set_defaults(func=dispatch)
        sub.add_argument("--output-dir", required=True)
        if name == "mandelbugs-audit":
            sub.add_argument("--raw-dir", required=True)
        elif name == "mandelbugs-enrich":
            sub.add_argument("--labels", required=True)
            sub.add_argument("--cache-dir", required=True)
            sub.add_argument("--sleep-seconds", type=float, default=1.0)
            sub.add_argument("--manual-dir", default="data/manual_enrichment")
            sub.add_argument("--retry-failures", action="store_true")
            sub.add_argument("--offline", action="store_true")
            sub.add_argument(
                "--allow-web-archive",
                action="store_true",
                help=(
                    "Read public Internet Archive snapshots for trackers that refuse "
                    "automated clients (MySQL). The blocked origin is never contacted."
                ),
            )
            sub.add_argument(
                "--prior-reports",
                action="append",
                help="Existing reports.parquet to reuse; may be specified more than once",
            )
            selector = sub.add_mutually_exclusive_group()
            selector.add_argument(
                "--only-issue", help="Retrieve one official identity, e.g. MySQL:21704"
            )
            selector.add_argument(
                "--only-project",
                choices=["Linux", "MySQL", "HTTPD", "AXIS"],
                help="Retrieve only one official project (useful for portable local runs)",
            )
        elif name == "mandelbugs-prepare":
            sub.add_argument("--labels", required=True)
            sub.add_argument("--reports", required=True)
        else:
            sub.add_argument("--data", required=True)
            sub.add_argument("--evidence-mode", choices=["initial", "full"], required=True)
            sub.add_argument("--config")
            if name == "train-stage2-baseline":
                sub.add_argument(
                    "--model",
                    choices=["tfidf_svm", "sbert_logreg", "frozen_stage1_encoder_transfer"],
                    required=True,
                )
                sub.add_argument("--cache-dir")
                sub.add_argument("--stage1-model")
            else:
                sub.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 97])
