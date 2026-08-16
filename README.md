# BugClassiNet-Next

Reproducible, hierarchical classification of software issues. Stage 1 labels an
issue as `BUG`, `ENHANCEMENT`, `QUESTION`, or `DOCUMENTATION`; bug reports then
route through BOH/MAN and ARB/NAM classifiers.

## Local quick start

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
python -m bugclassinet.cli inspect-archive data/raw/nlbse2023/train.tar.gz
python -m bugclassinet.cli prepare-nlbse --train-archive TRAIN.tar.gz --test-archive TEST.tar.gz
python -m bugclassinet.cli train-tfidf --data-dir data/processed/nlbse2023 --output-dir outputs/models/tfidf
python -m bugclassinet.cli evaluate-stage1 --model-path outputs/models/tfidf/model.joblib --test data/processed/nlbse2023/test.parquet
```

Install Transformer dependencies only in environments that train DeBERTa,
ModernBERT, or DAPT: `pip install -r requirements-transformers.txt`.

`prepare-nlbse` detects source columns, preserves them, produces Parquet files,
and refuses ambiguous schemas. It never edits an official test set. It writes
`train_benchmark.parquet`/`validation.parquet` plus test-isolated
`train_clean.parquet`/`validation_clean.parquet`; package training defaults to
the clean variants. Pass paths by arguments or YAML config; no platform-specific
paths are embedded in code.

## Kaggle

Attach a dataset containing the source archives, install the project package in a
notebook, set `--config configs/paths/kaggle.yaml`, and write artifacts to
`/kaggle/working/outputs`. The provided notebooks are deliberately thin wrappers
around `bugclassinet` package functions.

## Quality

```powershell
python -m ruff check .
python -m ruff format --check .
pytest
```

Large raw data, processed data, checkpoints, and outputs are ignored by Git.

## Full-scale TF-IDF

For million-row NLBSE training under constrained RAM, start with
`configs/models/tfidf_stage1_word_only.yaml`. It uses at most 200,000 sparse
`float32` word features. The `tfidf_stage1.yaml` preset adds at most 150,000
character features for a second, more memory-intensive benchmark. Both presets
retain the complete validation split and log sparse matrix shape, dtype, nonzero
count, and estimated storage before fitting `LinearSVC`.

For a memory-safe scale-up, run the combined preset in separate processes with
`--max-train-samples 200000`, then `--max-train-samples 500000`, and finally
omit the option for all clean training rows. The bounded runs use an exact,
deterministic, class-stratified subset (seed 42). Each run still evaluates the
complete validation split; TF-IDF training rejects `--max-eval-samples`.
