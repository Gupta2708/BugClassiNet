"""Frozen encoders with content-addressed, checksum-verified embedding caches."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from bugclassinet.evaluation.stage1 import _model_parameter_sha256
from bugclassinet.training.stage2_protocol import object_hash
from bugclassinet.utils.checksums import sha256_file
from bugclassinet.utils.io import write_json


def frozen_embeddings(
    texts: list[str], config: dict[str, Any], cache_dir: Path, stage1_model: str | None = None
) -> tuple[np.ndarray, dict]:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if stage1_model:
        from bugclassinet.evaluation.frozen_stage1 import FrozenStage1Evaluator

        evaluator = FrozenStage1Evaluator(stage1_model, cache_dir / "stage1-runtime")
        encoder = evaluator.model.base_model.to(device).eval()
        tokenizer = evaluator.tokenizer
        maximum = evaluator.max_length
        identity = {
            **evaluator.identity(),
            "pooling": "attention-masked mean of final hidden states",
            "experiment": "frozen_stage1_encoder_transfer",
        }
        before = _model_parameter_sha256(evaluator.model, torch)
    else:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError("SBERT requires pip install -e .[stage2]") from error
        encoder = SentenceTransformer(
            config["encoder_name"], revision=config.get("encoder_revision"), device=device
        )
        encoder.eval()
        encoder.max_seq_length = config["embedding_max_length"]
        before = _model_parameter_sha256(encoder, torch)
        identity = {
            "model_name": config["encoder_name"],
            "model_revision": getattr(encoder[0].auto_model.config, "_commit_hash", None),
            "max_length": encoder.max_seq_length,
            "model_parameter_sha256_before": before,
            "pooling": "saved SentenceTransformer pooling, L2 normalized",
        }
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    identity["torch_version"] = torch.__version__
    identity["packages"] = {}
    for package in ("transformers", "tokenizers", "sentence-transformers"):
        try:
            identity["packages"][package] = version(package)
        except PackageNotFoundError:
            identity["packages"][package] = None
    identity["ordered_text_sha256"] = object_hash(texts)
    key = object_hash(identity)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path, meta = cache_dir / f"{key}.npy", cache_dir / f"{key}.json"
    if path.exists() and meta.exists():
        info = json.loads(meta.read_text(encoding="utf-8"))
        if sha256_file(path) != info["embedding_sha256"]:
            raise ValueError(f"Corrupt embedding cache: {path}")
        values = np.load(path, allow_pickle=False)
    else:
        with torch.inference_mode():
            if not stage1_model:
                values = encoder.encode(
                    texts,
                    batch_size=config["embedding_batch_size"],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                ).astype(np.float32)
            else:
                parts = []
                for start in range(0, len(texts), config["embedding_batch_size"]):
                    batch = tokenizer(
                        texts[start : start + config["embedding_batch_size"]],
                        truncation=True,
                        max_length=maximum,
                        padding=True,
                        return_tensors="pt",
                    ).to(device)
                    hidden = encoder(**batch).last_hidden_state
                    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                    parts.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
                values = np.concatenate(parts).astype(np.float32)
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        temporary.replace(path)
        write_json(meta, {**identity, "embedding_sha256": sha256_file(path)})
    after = _model_parameter_sha256(evaluator.model if stage1_model else encoder, torch)
    if before != after:
        raise RuntimeError("Frozen encoder parameters changed during embedding extraction")
    if values.ndim != 2 or len(values) != len(texts) or not np.isfinite(values).all():
        raise ValueError("Invalid cached embedding shape/content")
    return values, {
        **identity,
        "model_parameter_sha256_after": after,
        "encoder_training_invoked": False,
        "embedding_sha256": sha256_file(path),
    }
