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
python -m bugclassinet.cli evaluate-stage1 --model outputs/models/deberta_stage1 --data data/processed/nlbse2023/validation_clean.parquet --output-dir outputs/evaluations/deberta_stage1
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

## Full-scale DeBERTa Stage 1

Stage 1 projects only the model columns from Parquet into a memory-mapped Arrow
dataset. Bounded runs use an exact class-stratified subset (seed 42), and
tokenization is written to a deterministic disk cache in bounded batches.
Sequences remain variable length until `DataCollatorWithPadding` pads each
training batch; truncation remains fixed at 256 tokens. DataLoader workers
default to zero to avoid process-local dataset copies on Kaggle.

Use `configs/models/deberta_stage1_kaggle_1epoch.yaml` for the 1-epoch Kaggle
scaling experiment. Do not pass `--max-eval-samples`: every 50K/200K/500K/full
training run must use the complete `validation_clean.parquet` split. This
Kaggle preset writes a normal, fully resumable Trainer checkpoint every 2,000
optimizer steps and retains the latest two. `--stop-after-steps N` requests a
full checkpoint and graceful stop without changing the one-epoch Trainer plan
or learning-rate schedule. Intermediate segments skip validation with
`--skip-final-evaluation`; the resumed final segment runs the complete
validation split. Resume with `--resume-from-checkpoint /path/to/checkpoint-N`.
The checkpoint manifest identifies persisted Parquet inputs by verified SHA-256,
row/class counts, schema, preprocessing version, and label-mapping hash. Hugging
Face/Arrow fingerprints are retained only for diagnostics because they can vary
between sessions. Resume still rejects changes to the dataset, label mapping,
model revision, seed, device count, precision, optimizer-step horizon, or training
configuration, and Trainer data skipping remains enabled so completed batches
are not replayed.

Legacy DeBERTa Stage-1 checkpoints whose fine-tuned LayerNorm tensors use
`.gamma`/`.beta` are remapped in memory to `.weight`/`.bias` during resume. The
original checkpoint is never rewritten, and training cannot begin unless the
resulting sequence-classification state is strictly complete, including the
classifier and pooler.

Stage-1 class weighting is controlled by `class_weight_strategy`: `balanced`
(the backward-compatible default), `sqrt_balanced`, `quarter_balanced`, `none`,
or `custom` with an explicit `class_weights` mapping. These correspond to
balanced-weight exponents 1, 0.5, 0.25, and 0 respectively (custom weights have
no exponent). The 200K/256-token ablation presets share all other hyperparameters
and must be run with `--max-train-samples 200000`; each manifest records the
ordered sample fingerprint, resolved weights, and exponent. The final validation
uses one `Trainer.predict` pass to produce aggregate and per-class scores, a
fixed-order confusion matrix, true/predicted counts, a full classification
report, and row-level logits/probabilities under `evaluation/`. Existing final
models and checkpoints can be reported without retraining via `evaluate-stage1
--model MODEL_DIR --data validation_clean.parquet --output-dir OUTPUT_DIR`.

## Full-scale TF-IDF

For million-row NLBSE training under constrained RAM, start with
`configs/models/tfidf_stage1_word_only.yaml`. It uses at most 200,000 sparse
`float32` word features. The `tfidf_stage1.yaml` preset adds at most 150,000
character features for a second, more memory-intensive benchmark. Both presets
retain the complete validation split and log sparse matrix shape, dtype, nonzero
count, and estimated storage before fitting `LinearSVC`.

`configs/models/tfidf_stage1_word_only_500k.yaml` is a higher-capacity
word-only benchmark with at most 500,000 unigram and bigram features. Compare it
with the 200,000-feature word-only baseline on the same complete validation set;
the larger vocabulary is not assumed to improve accuracy.

For a memory-safe scale-up, run the combined preset in separate processes with
`--max-train-samples 200000`, then `--max-train-samples 500000`, and finally
omit the option for all clean training rows. The bounded runs use an exact,
deterministic, class-stratified subset (seed 42). Each run still evaluates the
complete validation split; TF-IDF training rejects `--max-eval-samples`.
