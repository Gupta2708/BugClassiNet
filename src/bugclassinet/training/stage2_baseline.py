"""Deterministic majority, TF-IDF, and frozen-embedding Stage-2 baselines."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from bugclassinet.features.tfidf import SparseMatrixLogger, TfidfFeatureConfig, make_tfidf_features
from bugclassinet.settings import load_yaml
from bugclassinet.training.stage2_protocol import (
    fold_manifest,
    load_stage2,
    lopo_folds,
    new_output,
    run_identity,
    write_results,
)
from bugclassinet.utils.checksums import sha256_file
from bugclassinet.utils.io import write_json

DEFAULTS = {
    "seed": 42,
    "word_min_df": 2,
    "word_max_df": 1.0,
    "word_max_features": 50000,
    "C": 1.0,
    "max_iter": 5000,
    "bootstrap_iterations": 1000,
    "encoder_name": "sentence-transformers/all-mpnet-base-v2",
    "encoder_revision": None,
    "embedding_batch_size": 16,
    "embedding_max_length": 384,
}


def prediction_rows(
    frame: pd.DataFrame, test: np.ndarray, predicted: list, scores: list, seed: int, score_type: str
) -> pd.DataFrame:
    rows = frame.iloc[test][["issue_key", "project", "text_group"]].copy()
    rows["true_label"] = frame.iloc[test].stage2_label.tolist()
    rows["predicted_label"] = predicted
    rows["score_mandelbug"] = scores
    rows["score_type"] = score_type
    rows["seed"] = seed
    return rows


def majority_predictions(
    frame: pd.DataFrame, train: np.ndarray, test: np.ndarray, seed: int
) -> pd.DataFrame:
    counts = frame.iloc[train].stage2_label.value_counts()
    majority = "MANDELBUG" if counts.get("MANDELBUG", 0) > counts.get("BOH", 0) else "BOH"
    score = float(counts.get("MANDELBUG", 0) / len(train))
    return prediction_rows(
        frame,
        test,
        [majority] * len(test),
        [score] * len(test),
        seed,
        "training prevalence; majority decision; tie -> BOH",
    )


def train_baseline(
    data: str,
    model: str,
    evidence_mode: str,
    output_dir: str,
    config_path: str | None = None,
    cache_dir: str | None = None,
    stage1_model: str | None = None,
) -> dict:
    allowed = {"tfidf_svm", "sbert_logreg", "frozen_stage1_encoder_transfer"}
    if model not in allowed:
        raise ValueError(f"Unsupported baseline {model}; expected {sorted(allowed)}")
    config = {**DEFAULTS, **(load_yaml(config_path) if config_path else {})}
    if set(config) - set(DEFAULTS):
        raise ValueError(f"Unknown baseline config keys: {sorted(set(config) - set(DEFAULTS))}")
    if model == "frozen_stage1_encoder_transfer" and not stage1_model:
        raise ValueError(
            "Frozen transfer requires --stage1-model pointing to the completed checkpoint"
        )
    frame = load_stage2(data, evidence_mode)
    folds = lopo_folds(frame)
    out = new_output(output_dir)
    seed = int(config["seed"])
    identity = run_identity(data, {**config, "evidence_mode": evidence_mode}, model)
    embeddings = None
    if model != "tfidf_svm":
        from bugclassinet.training.stage2_embeddings import frozen_embeddings

        embeddings, encoder_identity = frozen_embeddings(
            frame.text.tolist(),
            config,
            Path(cache_dir or out.parent / ".embedding-cache"),
            stage1_model if model == "frozen_stage1_encoder_transfer" else None,
        )
        identity["encoder"] = encoder_identity
    predictions, majority, fold_identities = [], [], []
    for project, train, test, purged in folds:
        if model == "tfidf_svm":
            features = make_tfidf_features(
                TfidfFeatureConfig(
                    feature_mode="word",
                    word_ngram_range=(1, 2),
                    word_min_df=config["word_min_df"],
                    word_max_df=config["word_max_df"],
                    word_max_features=config["word_max_features"],
                )
            )
            classifier = Pipeline(
                [
                    ("features", features),
                    ("sparse_log", SparseMatrixLogger()),
                    (
                        "classifier",
                        LinearSVC(
                            C=config["C"],
                            class_weight="balanced",
                            random_state=seed,
                            max_iter=config["max_iter"],
                            dual=True,
                        ),
                    ),
                ]
            )
            classifier.fit(frame.iloc[train].text, frame.iloc[train].stage2_label)
            predicted = classifier.predict(frame.iloc[test].text)
            scores = classifier.decision_function(frame.iloc[test].text)
            classes = classifier.named_steps["classifier"].classes_
            if classes[1] != "MANDELBUG":
                scores = -scores
            score_type = "LinearSVC decision_function; positive -> MANDELBUG; not probability"
        else:
            classifier = LogisticRegression(
                C=config["C"],
                class_weight="balanced",
                solver="liblinear",
                random_state=seed,
                max_iter=config["max_iter"],
            )
            classifier.fit(embeddings[train], frame.iloc[train].stage2_label)
            predicted = classifier.predict(embeddings[test])
            scores = classifier.predict_proba(embeddings[test])[
                :, list(classifier.classes_).index("MANDELBUG")
            ]
            score_type = "LogisticRegression probability"
        fold_dir = out / "folds" / project
        fold_dir.mkdir(parents=True)
        joblib.dump(classifier, fold_dir / "model.joblib")
        info = {
            "held_out_project": project,
            "seed": seed,
            "purged_train_issue_keys": purged,
            **fold_manifest(frame, train, test),
            "model_sha256": sha256_file(fold_dir / "model.joblib"),
        }
        write_json(fold_dir / "run_manifest.json", {**identity, **info})
        fold_identities.append(info)
        prediction = prediction_rows(frame, test, predicted, scores, seed, score_type)
        prediction.to_parquet(fold_dir / "predictions.parquet", index=False)
        predictions.append(prediction)
        majority.append(majority_predictions(frame, train, test, seed))
    identity["folds"] = fold_identities
    majority_dir = out / "majority"
    majority_dir.mkdir()
    write_results(
        majority_dir,
        pd.concat(majority, ignore_index=True),
        {**identity, "model": "majority"},
        config["bootstrap_iterations"],
    )
    return write_results(
        out, pd.concat(predictions, ignore_index=True), identity, config["bootstrap_iterations"]
    )
