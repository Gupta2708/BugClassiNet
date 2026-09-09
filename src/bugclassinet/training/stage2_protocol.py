"""Shared Stage-2 project-held-out folds, reporting, and confidence intervals."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from bugclassinet.data.harmonize import normalised_text_hash
from bugclassinet.data.mandelbugs_schema import LABELS, PROJECTS, TEXT_VERSION
from bugclassinet.evaluation.metrics import classification_metrics
from bugclassinet.utils.checksums import sha256_file
from bugclassinet.utils.io import write_json
from bugclassinet.utils.reproducibility import git_commit, utc_timestamp


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def load_stage2(path: str | Path, evidence_mode: str) -> pd.DataFrame:
    if evidence_mode not in {"initial", "full"}:
        raise ValueError("evidence_mode must be initial or full")
    frame = pd.read_parquet(path)
    required = {"issue_key", "project", "stage2_label", "text_initial", "text_full"}
    if missing := required - set(frame):
        raise ValueError(f"Stage-2 dataset missing {sorted(missing)}")
    if frame.empty or frame.issue_key.isna().any() or frame.issue_key.duplicated().any():
        raise ValueError("Stage-2 requires nonempty, unique issue identities")
    if set(frame.project) != set(PROJECTS):
        raise ValueError(
            f"LOPO requires all four projects {PROJECTS}; observed={sorted(frame.project.unique())}"
        )
    if set(frame.stage2_label) != set(LABELS):
        raise ValueError(f"Stage-2 requires labels {LABELS}")
    frame = frame.sort_values("issue_key", kind="stable").reset_index(drop=True)
    frame["text"] = frame[f"text_{evidence_mode}"]
    if frame.text.isna().any() or frame.text.str.strip().eq("").any():
        raise ValueError("Stage-2 evidence is missing or blank")
    frame["text_group"] = frame.text.map(normalised_text_hash)
    if (frame.groupby("text_group").stage2_label.nunique() > 1).any():
        raise ValueError("Conflicting text labels must be quarantined by mandelbugs-prepare")
    return frame


def lopo_folds(frame: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray, list[str]]]:
    folds = []
    for project in PROJECTS:
        test = np.flatnonzero(frame.project.eq(project).to_numpy())
        other = frame.project.ne(project)
        overlap = frame.text_group.isin(set(frame.iloc[test].text_group))
        train = np.flatnonzero((other & ~overlap).to_numpy())
        if not len(test) or set(frame.iloc[train].stage2_label) != set(LABELS):
            raise ValueError(
                f"Fold {project} has empty test or single-class training after duplicate purge"
            )
        folds.append((project, train, test, frame.loc[other & overlap, "issue_key"].tolist()))
    return folds


def inner_validation(
    frame: pd.DataFrame, outer_train: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a deterministic stratified, text-group-disjoint inner fold."""
    subset = frame.iloc[outer_train]
    counts = subset.groupby("stage2_label").text_group.nunique()
    n_splits = min(5, int(counts.min()))
    if n_splits < 2:
        raise ValueError("Too few independent text groups for inner validation")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit, valid in splitter.split(subset, subset.stage2_label, subset.text_group):
        if set(subset.iloc[fit].stage2_label) == set(LABELS) and set(
            subset.iloc[valid].stage2_label
        ) == set(LABELS):
            return outer_train[fit], outer_train[valid]
    raise ValueError("No viable two-class inner validation partition")


def metrics(truth: list[str], predicted: list[str], scores: list[float]) -> dict[str, Any]:
    result = classification_metrics(truth, predicted, LABELS)
    result["pr_auc_mandelbug"] = (
        float(average_precision_score([v == "MANDELBUG" for v in truth], scores))
        if set(truth) == set(LABELS)
        else None
    )
    result["pr_auc_definition"] = "average precision; decision scores or probabilities as recorded"
    return result


def bootstrap_ci(predictions: pd.DataFrame, seed: int, iterations: int = 1000) -> dict:
    """Paired text-group bootstrap within projects; keep seeds together per issue."""
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    rng = np.random.default_rng(seed)
    by_seed = [group.set_index("issue_key") for _, group in predictions.groupby("seed", sort=True)]
    first = by_seed[0]
    groups = [
        [group.index.to_list() for _, group in project.groupby("text_group", sort=True)]
        for _, project in first.groupby("project", sort=True)
    ]
    aggregate, project_means = [], []
    for _ in range(iterations):
        ids = [
            key
            for blocks in groups
            for i in rng.integers(0, len(blocks), len(blocks))
            for key in blocks[i]
        ]
        seed_scores, repo_scores = [], []
        for table in by_seed:
            sample = table.loc[ids]
            seed_scores.append(
                f1_score(
                    sample.true_label,
                    sample.predicted_label,
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                )
            )
            repo_scores.append(
                np.mean(
                    [
                        f1_score(
                            g.true_label,
                            g.predicted_label,
                            labels=LABELS,
                            average="macro",
                            zero_division=0,
                        )
                        for _, g in sample.groupby("project")
                    ]
                )
            )
        aggregate.append(np.mean(seed_scores))
        project_means.append(np.mean(repo_scores))
    return {
        "method": "percentile 95%; text groups resampled within observed projects; seeds paired",
        "limitation": (
            "Conditional on these four projects; not uncertainty over unseen project populations"
        ),
        "seed": seed,
        "iterations": iterations,
        "aggregate_macro_f1": np.percentile(aggregate, [2.5, 97.5]).tolist(),
        "mean_project_macro_f1": np.percentile(project_means, [2.5, 97.5]).tolist(),
    }


def fold_manifest(
    frame: pd.DataFrame, train: np.ndarray, test: np.ndarray, valid: np.ndarray | None = None
) -> dict:
    result = {}
    for name, indices in (("train", train), ("validation", valid), ("test", test)):
        part = frame.iloc[indices] if indices is not None else frame.iloc[:0]
        result[f"{name}_projects"] = sorted(part.project.unique())
        result[f"{name}_issue_keys"] = part.issue_key.tolist()
        result[f"{name}_class_counts"] = part.stage2_label.value_counts().to_dict()
    return result


def run_identity(data: str | Path, config: dict, model: str) -> dict:
    packages = {}
    for name in (
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "transformers",
        "sentence-transformers",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "dataset_path": str(Path(data).resolve()),
        "dataset_sha256": sha256_file(data),
        "model": model,
        "config": config,
        "config_hash": object_hash(config),
        "packages": packages,
        "git_commit": git_commit(),
        "timestamp": utc_timestamp(),
        "training_invoked": True,
        "text_version": TEXT_VERSION,
        "label_mapping": {label: i for i, label in enumerate(LABELS)},
        "protocol": "LOPO",
    }


def new_output(path: str | Path) -> Path:
    out = Path(path)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is nonempty; choose a new experiment directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_results(
    out: Path, predictions: pd.DataFrame, manifest: dict, bootstrap_iterations: int = 1000
) -> dict:
    predictions.to_parquet(out / "predictions.parquet", index=False)
    per_fold, per_seed, reports, matrices = [], {}, [], []
    for seed, rows in predictions.groupby("seed", sort=True):
        overall = metrics(
            rows.true_label.tolist(), rows.predicted_label.tolist(), rows.score_mandelbug.tolist()
        )
        per_seed[str(seed)] = overall
        for label in LABELS:
            reports.append({"seed": seed, "class": label, **overall["per_class"][label]})
        for i, label in enumerate(LABELS):
            matrices.append(
                {
                    "seed": seed,
                    "true_label": label,
                    **{
                        f"PRED_{p}": overall["confusion_matrix"][i][j] for j, p in enumerate(LABELS)
                    },
                }
            )
        for project, part in rows.groupby("project", sort=True):
            values = metrics(
                part.true_label.tolist(),
                part.predicted_label.tolist(),
                part.score_mandelbug.tolist(),
            )
            flat = {
                key: values[key]
                for key in ("macro_f1", "accuracy", "balanced_accuracy", "mcc", "pr_auc_mandelbug")
            }
            for label in LABELS:
                flat.update({f"{label}_{k}": v for k, v in values["per_class"][label].items()})
            per_fold.append({"seed": seed, "project": project, "rows": len(part), **flat})
    fold_table = pd.DataFrame(per_fold)
    fold_table.to_csv(out / "per_fold_metrics.csv", index=False)
    pd.DataFrame(reports).to_csv(out / "classification_report.csv", index=False)
    pd.DataFrame(matrices).to_csv(out / "confusion_matrix.csv", index=False)
    write_json(
        out / "aggregate_metrics.json",
        per_seed if len(per_seed) > 1 else next(iter(per_seed.values())),
    )
    aggregate_f1 = [m["macro_f1"] for m in per_seed.values()]
    per_project_f1 = fold_table.groupby("seed").macro_f1.mean().tolist()
    summary = {
        "primary_metric": "macro_f1",
        "seeds": list(per_seed),
        "unique_issues": predictions.issue_key.nunique(),
        "aggregate_macro_f1_mean": float(np.mean(aggregate_f1)),
        "aggregate_macro_f1_std": float(np.std(aggregate_f1, ddof=1))
        if len(aggregate_f1) > 1
        else 0.0,
        "mean_project_macro_f1": float(np.mean(per_project_f1)),
        "mean_project_macro_f1_std": float(np.std(per_project_f1, ddof=1))
        if len(per_project_f1) > 1
        else 0.0,
        "class_counts": predictions.drop_duplicates("issue_key")
        .true_label.value_counts()
        .to_dict(),
        "training_invoked": True,
    }
    summary["aggregate_seed_statistics"] = {}
    for key in (
        "macro_f1",
        "micro_f1",
        "weighted_f1",
        "accuracy",
        "mcc",
        "balanced_accuracy",
        "pr_auc_mandelbug",
    ):
        values = [m[key] for m in per_seed.values() if m[key] is not None]
        summary["aggregate_seed_statistics"][key] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None),
        }
    project_summary = fold_table.groupby("project").macro_f1.agg(["mean", "std", "count"])
    project_summary["std"] = project_summary["std"].fillna(0.0)
    project_summary.to_csv(out / "per_project_seed_summary.csv")
    write_json(out / "summary.json", summary)
    write_json(out / "run_manifest.json", manifest)
    write_json(out / "bootstrap_ci.json", bootstrap_ci(predictions, 42, bootstrap_iterations))
    return summary
