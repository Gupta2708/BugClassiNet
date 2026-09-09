"""Low-data pretrained ModernBERT with inner early stopping and outer LOPO."""

from __future__ import annotations

import gc
import os
from typing import Any

import numpy as np
import pandas as pd

from bugclassinet.data.mandelbugs_schema import LABELS
from bugclassinet.evaluation.metrics import classification_metrics
from bugclassinet.evaluation.stage1 import _model_parameter_sha256
from bugclassinet.settings import load_yaml
from bugclassinet.training.stage2_baseline import majority_predictions, prediction_rows
from bugclassinet.training.stage2_protocol import (
    fold_manifest,
    inner_validation,
    load_stage2,
    lopo_folds,
    new_output,
    object_hash,
    run_identity,
    write_results,
)
from bugclassinet.utils.checksums import sha256_file
from bugclassinet.utils.io import write_json

DEFAULTS = {
    "model_name": "answerdotai/ModernBERT-base",
    "model_revision": None,
    "max_length": 256,
    "batch_size": 8,
    "learning_rate": 1e-5,
    "epochs": 10,
    "patience": 2,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "freeze_lower_layers": 0,
    "bootstrap_iterations": 1000,
}


class EvidenceDataset:
    def __init__(self, texts: list[str], labels: list[str], tokenizer: Any, max_length: int):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        values = self.tokenizer(
            self.texts[index], truncation=True, max_length=self.max_length, padding=False
        )
        values["labels"] = LABELS.index(self.labels[index])
        return values


def weighted_loss(logits: Any, labels: Any, weights: Any) -> Any:
    import torch

    return torch.nn.functional.cross_entropy(logits, labels, weight=weights)


def freeze_lower_layers(model: Any, count: int) -> None:
    if count < 0:
        raise ValueError("freeze_lower_layers must be nonnegative")
    if count == 0:
        return
    base = model.base_model
    layers = getattr(base, "layers", None)
    if layers is None or count > len(layers):
        raise ValueError("Cannot resolve requested ModernBERT lower layers")
    for parameter in base.embeddings.parameters():
        parameter.requires_grad_(False)
    for layer in layers[:count]:
        for parameter in layer.parameters():
            parameter.requires_grad_(False)


def _predict(model: Any, loader: Any, device: Any, torch: Any) -> tuple[list[str], list[float]]:
    model.eval()
    predicted, scores = [], []
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels")
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            probabilities = torch.softmax(logits.float(), dim=-1)
            predicted.extend(LABELS[i] for i in probabilities.argmax(dim=-1).cpu().tolist())
            scores.extend(probabilities[:, LABELS.index("MANDELBUG")].cpu().tolist())
    return predicted, scores


def train_modernbert(
    data: str,
    evidence_mode: str,
    output_dir: str,
    seeds: list[int] | None = None,
    config_path: str | None = None,
) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        set_seed,
    )

    seeds = seeds or [13, 42, 97]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be distinct")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2 ModernBERT requires a GPU; enable a Kaggle accelerator")
    config = {**DEFAULTS, **(load_yaml(config_path) if config_path else {})}
    if set(config) - set(DEFAULTS):
        raise ValueError(f"Unknown ModernBERT config keys: {set(config) - set(DEFAULTS)}")
    for key in ("epochs", "patience", "batch_size", "max_length", "learning_rate", "gradient_clip"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    frame = load_stage2(data, evidence_mode)
    folds = lopo_folds(frame)
    # Resolve every split before downloading/training; no held-out labels used by splitter.
    partitions = {
        (seed, project): inner_validation(frame, train, seed)
        for seed in seeds
        for project, train, _, _ in folds
    }
    out = new_output(output_dir)
    # Resolve one pretrained revision for tokenizer and every fold/seed before loading weights.
    pretrained_config = AutoConfig.from_pretrained(
        config["model_name"], revision=config["model_revision"]
    )
    config["model_revision"] = (
        getattr(pretrained_config, "_commit_hash", None) or config["model_revision"]
    )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    identity = run_identity(
        data, {**config, "seeds": seeds, "evidence_mode": evidence_mode}, "modernbert"
    )
    identity.update(
        precision="float32",
        strategy=(
            "full_finetuning"
            if not config["freeze_lower_layers"]
            else f"freeze_lower_{config['freeze_lower_layers']}_layers"
        ),
        device="cuda:0",
        distributed_training=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], revision=config["model_revision"]
    )
    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    device = torch.device("cuda:0")
    all_predictions, majority, seed_results, fold_info = [], [], [], []
    for seed in seeds:
        for project, outer_train, test, purged in folds:
            set_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            train, valid = partitions[(seed, project)]
            fold_dir = out / f"seed-{seed}" / project
            fold_dir.mkdir(parents=True)
            model = (
                AutoModelForSequenceClassification.from_pretrained(
                    config["model_name"],
                    revision=config["model_revision"],
                    num_labels=2,
                    label2id={label: i for i, label in enumerate(LABELS)},
                    id2label={i: label for i, label in enumerate(LABELS)},
                    attn_implementation="eager",
                    reference_compile=False,
                )
                .float()
                .to(device)
            )
            freeze_lower_layers(model, config["freeze_lower_layers"])
            resolved_revision = getattr(model.config, "_commit_hash", None)
            info = {
                **identity,
                **fold_manifest(frame, train, test, valid),
                "model_revision": resolved_revision,
                "model_config_hash": object_hash(model.config.to_dict()),
                "seed": seed,
                "held_out_project": project,
                "purged_train_issue_keys": purged,
                "status": "running",
                "epoch_history": [],
            }
            write_json(fold_dir / "run_manifest.json", info)

            def loader(indices: np.ndarray, shuffle: bool = False, seed: int = seed) -> Any:
                part = frame.iloc[indices]
                dataset = EvidenceDataset(
                    part.text.tolist(), part.stage2_label.tolist(), tokenizer, config["max_length"]
                )
                return DataLoader(
                    dataset,
                    batch_size=config["batch_size"],
                    shuffle=shuffle,
                    collate_fn=collator,
                    num_workers=0,
                    pin_memory=True,
                    generator=torch.Generator().manual_seed(seed),
                )

            train_loader, valid_loader = loader(train, True), loader(valid)
            counts = frame.iloc[train].stage2_label.value_counts()
            weights = torch.tensor(
                [len(train) / (2 * counts[label]) for label in LABELS],
                dtype=torch.float32,
                device=device,
            )
            info["class_weights"] = dict(zip(LABELS, weights.cpu().tolist(), strict=True))
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )
            best, stale, best_epoch = -1.0, 0, 0
            best_path = fold_dir / "best_model.pt"
            for epoch in range(1, config["epochs"] + 1):
                model.train()
                total_loss = 0.0
                for batch in train_loader:
                    targets = batch.pop("labels").to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
                    loss = weighted_loss(logits, targets, weights)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite training loss in {project}, seed={seed}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                    optimizer.step()
                    total_loss += loss.item()
                predicted, _ = _predict(model, valid_loader, device, torch)
                score = classification_metrics(
                    frame.iloc[valid].stage2_label.tolist(), predicted, LABELS
                )["macro_f1"]
                info["epoch_history"].append(
                    {
                        "epoch": epoch,
                        "inner_macro_f1": score,
                        "training_mean_batch_loss": total_loss / len(train_loader),
                    }
                )
                print(f"seed={seed} held_out={project} epoch={epoch} inner_macro_f1={score:.6f}")
                if score > best:
                    best, stale, best_epoch = score, 0, epoch
                    torch.save(model.state_dict(), best_path)
                else:
                    stale += 1
                write_json(fold_dir / "run_manifest.json", info)
                if stale >= config["patience"]:
                    break
            model.load_state_dict(
                torch.load(best_path, map_location="cpu", weights_only=True), strict=True
            )
            # First access to the outer test loader happens only after early stopping.
            predictions, scores = _predict(model, loader(test), device, torch)
            rows = prediction_rows(frame, test, predictions, scores, seed, "ModernBERT softmax")
            rows.to_parquet(fold_dir / "predictions.parquet", index=False)
            model.save_pretrained(fold_dir / "model")
            tokenizer.save_pretrained(fold_dir / "model")
            info.update(
                status="complete",
                best_epoch=best_epoch,
                best_inner_macro_f1=best,
                model_parameter_sha256=_model_parameter_sha256(model, torch),
                best_model_sha256=sha256_file(best_path),
            )
            write_json(fold_dir / "run_manifest.json", info)
            fold_info.append(info)
            all_predictions.append(rows)
            majority.append(majority_predictions(frame, outer_train, test, seed))
            del model, optimizer, train_loader, valid_loader
            gc.collect()
            torch.cuda.empty_cache()
        seed_dir = out / f"seed-{seed}"
        seed_rows = pd.concat(
            [p for p in all_predictions if int(p.seed.iloc[0]) == seed], ignore_index=True
        )
        seed_results.append(
            write_results(
                seed_dir, seed_rows, {**identity, "seed": seed}, config["bootstrap_iterations"]
            )
        )
    majority_dir = out / "majority"
    majority_dir.mkdir()
    write_results(
        majority_dir,
        pd.concat(majority, ignore_index=True),
        {**identity, "model": "majority"},
        config["bootstrap_iterations"],
    )
    return write_results(
        out,
        pd.concat(all_predictions, ignore_index=True),
        {**identity, "folds": fold_info, "seed_results": seed_results},
        config["bootstrap_iterations"],
    )
