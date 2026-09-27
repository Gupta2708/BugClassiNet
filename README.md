# BugClassiNet-Next

**Reproducible, imbalance-aware, hierarchical classification of software issues — from raw GitHub issue text to fine-grained bug taxonomy.**

BugClassiNet-Next is a three-stage cascade. Stage 1 classifies an issue as
`BUG` / `ENHANCEMENT` / `QUESTION` / `DOCUMENTATION` at million-row scale.
Confirmed bugs route to Stage 2, which separates deterministic **Bohrbugs
(BOH)** from **Mandelbugs**; Mandelbugs route to Stage 3, which separates
**Non-Aging Mandelbugs (NAM)** from **Aging-Related Bugs (ARB)**. All stages
are complete, frozen, and evaluated end-to-end.

```
GitHub issue
     │
     ▼
Stage 1 — DeBERTa-v3-small (1.09M train rows, NLBSE 2023)
     ├── DOCUMENTATION / ENHANCEMENT / QUESTION ──▶ stop
     └── BUG
          │
          ▼
Stage 2 — SBERT + LogReg, nested project-aware thresholds (806 issues, 4 projects)
     ├── BOH ──▶ stop
     └── MANDELBUG
          │
          ▼
Stage 3 — SBERT + LogReg, nested class-weight + threshold selection (262 issues)
     └──▶ NAM or ARB
```

## 🏁 Headline results

| Stage | Task | Final model | Key metrics |
|---|---|---|---|
| **1** | BUG / DOC / ENH / QUESTION (NLBSE 2023, 142,320 official test) | DeBERTa-v3-small, 256 tokens, 2 epochs, power-balanced loss (α = 0.25) | **Macro-F1 0.7934** · Micro-F1 0.8848 · MCC 0.8016 |
| **2** | BOH vs MANDELBUG, cross-project LOPO (806 issues: AXIS, HTTPD, Linux, MySQL) | SBERT `all-mpnet-base-v2` @512 + logistic regression + nested project-aware threshold calibration | **Macro-F1 0.7626** · Acc 0.7816 · MCC 0.5332 |
| **3** | NAM vs ARB, cross-project LOPO (262 Mandelbugs) | SBERT @512 + logistic regression + nested class-weight & ARB-threshold selection | **Macro-F1 0.7322** · Acc 0.8397 · ARB F1 0.5625 |
| **Cascade** | Full Stage 1→2→3 routing on 806 known-BUG reports | All three frozen stages, argmax Stage-1 gate | **Accuracy 0.6799** · Macro-F1 0.6126 (oracle-routing ceiling: 0.9479) |

Full write-ups with protocols, ablations, confusion matrices, and error
analyses live in [docs/Reports/](docs/Reports/):

- [Stage 1 — semantic issue classification at scale](docs/Reports/BugClassiNet_Research_Report_Stage1.docx)
- [Stage 2 — Bohrbug vs Mandelbug](docs/Reports/BugClassiNet_Stage2_Research_Report_Updated_Final.docx)
- [Stage 3 — NAM vs ARB](docs/Reports/BugClassiNet_Stage3_Research_Report_Final.docx)
- [Frozen hierarchical routing evaluation](docs/Reports/BugClassiNet_Frozen_Hierarchical_Routing_Evaluation.docx)

## 🔬 What the research found

**Stage 1 (NLBSE 2023, 1,089,694 clean training rows).**
- A leakage-aware preprocessing pipeline (dedup, source-ID disambiguation,
  official-test overlap removal, SHA-256 source identity) feeds both a
  classical baseline and the transformer path.
- A 2M-feature TF-IDF baseline reached Macro-F1 0.7643; the final DeBERTa
  model beat it by ~2.8 Macro-F1 points.
- Conventional inverse-frequency class weighting *over-corrects* minority
  classes (high recall, poor precision). A mild power-balanced loss
  (α = 0.25) gave the best Macro-F1 trade-off.
- Two epochs > one epoch; 256 tokens beat 384 on compute-adjusted Macro-F1.
- Validation (0.7927) and official test (0.7934) Macro-F1 are nearly
  identical — stable generalization, not validation-only gains.
- Engineering: host-RAM pressure cut from ~30 GiB to single-digit GiB via
  disk-backed Arrow, batched tokenization, and dynamic padding; robust
  cross-session Kaggle checkpoint/resume with strict state verification.

**Stage 2 (806 issues across AXIS, HTTPD, Linux, MySQL).**
- Four-project leave-one-project-out (LOPO) evaluation throughout — the
  test project is never seen in training or threshold selection.
- Fixed SBERT sentence embeddings + logistic regression clearly beat both
  TF-IDF (0.5956) and end-to-end ModernBERT fine-tuning (0.6308 ± 0.0291).
- Leakage-free *nested* threshold calibration (inner LOPO on training
  projects only) lifted Macro-F1 from 0.7468 to **0.7626** by cutting
  BOH→MANDELBUG false positives from 139 to 115.
- Triage-time "initial evidence" (title + description + environment)
  outperformed full comment threads — the deployable condition wins.

**Stage 3 (262 Mandelbugs: 207 NAM / 55 ARB).**
- The gain came from *decision selection*, not a bigger encoder: nested
  class-weight + ARB-threshold selection raised Macro-F1 from 0.6149 to
  **0.7322** and ARB recall from 0.20 to 0.49.
- ModernBERT performed near chance (0.4996 Macro-F1) and was rejected;
  full-thread evidence again underperformed initial evidence.

**Routing evaluation (frozen cascade on 806 known-BUG reports).**
- Oracle Stage-2 routing ceiling: 0.9479 accuracy. Predicted routing:
  0.7382. Full cascade with the Stage-1 gate: **0.6799**.
- Stage 2 routing is the dominant bottleneck (65.5 % of final errors);
  Stage 1 rejection adds 20.9 %; Stage 3 subtype errors only 13.6 %.
- Stage 1 retains 93.3 % of known bugs overall and 98.85 % of Mandelbugs,
  but retention is project-skewed (AXIS 84.3 %, MySQL 99.5 %).

## 🚀 Local quick start

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
the clean variants. Pass paths by arguments or YAML config; no
platform-specific paths are embedded in code.

## 🗂️ Repository layout

| Path | Contents |
|---|---|
| `src/bugclassinet/` | All reusable pipeline, training, and evaluation code (CLI: `python -m bugclassinet.cli`) |
| `configs/` | YAML presets for models, paths, and ablations |
| `notebooks/kaggle/` | Thin Kaggle wrappers around package APIs |
| `docs/Reports/` | The four final research reports |
| `docs/stage2_mandelbugs.md` | Stage-2/3 protocol and command reference |
| `docs/shared_transformer_training.md` | Shared label-agnostic transformer trainer notes |
| `tests/` | Pytest suite |

## 🧪 Stage 2 / Stage 3 workflow

The Mandelbugs workflow provides audited ARFF labels, cached/manual issue
enrichment, four-project LOPO baselines, optional frozen Stage-1 encoder
transfer, and a low-data ModernBERT trainer. See
[docs/stage2_mandelbugs.md](docs/stage2_mandelbugs.md) and
[the Kaggle notebook](notebooks/kaggle/12_stage2_mandelbugs_lopo.ipynb).

Key commands: `mandelbugs-audit`, `mandelbugs-enrich`, `mandelbugs-prepare`,
`train-stage2-baseline`, `train-stage2-modernbert`.

## ⚙️ Full-scale Stage 1 training notes

Stage 1 projects only the model columns from Parquet into a memory-mapped
Arrow dataset. Bounded runs use an exact class-stratified subset (seed 42);
tokenization is written to a deterministic disk cache in bounded batches and
padded dynamically per batch (truncation fixed at 256 tokens).

Use `configs/models/deberta_stage1_kaggle_1epoch.yaml` for Kaggle scaling
runs. Every scaling run evaluates the complete `validation_clean.parquet`
split (never pass `--max-eval-samples`). Checkpoints are fully resumable
(`--resume-from-checkpoint`), identified by verified SHA-256 of persisted
Parquet inputs, and reject any change to dataset, label mapping, model
revision, seed, precision, or training configuration. Legacy
`.gamma`/`.beta` LayerNorm checkpoints are remapped in memory on resume.

Class weighting is controlled by `class_weight_strategy`: `balanced`,
`sqrt_balanced`, `quarter_balanced` (the final α = 0.25 choice), `none`, or
`custom`. The standalone evaluator (`evaluate-stage1`) disables training,
verifies model-parameter hashes before/after, and writes full reports plus an
`evaluation_manifest.json`.

```powershell
python -m bugclassinet.cli evaluate-stage1 --model FINAL_CHECKPOINT --data data/processed/nlbse2023/test.parquet --output-dir outputs/evaluation/stage1_final_test
```

### Stage-1 closeout evaluations

`evaluate-stage1-binary` derives BUG vs NON_BUG decisions from the frozen
four-class logits (threshold selected on validation only). `evaluate-nlbse2024`
reports frozen NLBSE 2023 → NLBSE 2024 cross-dataset transfer (`feature` maps
to `ENHANCEMENT`; NLBSE 2024 training data is used for schema audit only,
never fitting). This is not the official NLBSE 2024 competition protocol.

```powershell
python -m bugclassinet.cli evaluate-stage1-binary --model FINAL_CHECKPOINT --validation data/processed/nlbse2023/validation_clean.parquet --test data/processed/nlbse2023/test.parquet --output-dir outputs/evaluation/stage1_binary
python -m bugclassinet.cli evaluate-nlbse2024 --model FINAL_CHECKPOINT --data data/raw/nlbse2024/issues_test.csv --output-dir outputs/evaluation/nlbse2024_transfer
```

### Full-scale TF-IDF baseline

Start with `configs/models/tfidf_stage1_word_only.yaml` (200K word features);
`tfidf_stage1_word_only_500k.yaml` scales to 500K, and `tfidf_stage1.yaml`
adds character features for a more memory-intensive benchmark. Scale up in
separate processes with `--max-train-samples 200000`, then `500000`, then the
full corpus; every run evaluates the complete validation split.

## ☁️ Kaggle

Attach a dataset containing the source archives, install the project package
in a notebook, set `--config configs/paths/kaggle.yaml`, and write artifacts
to `/kaggle/working/outputs`. The provided notebooks are deliberately thin
wrappers around `bugclassinet` package functions.

## ✅ Quality

```powershell
python -m ruff check .
python -m ruff format --check .
pytest
```

Large raw data, processed data, checkpoints, and outputs are ignored by Git.
Official test sets are never edited, and labels/statuses/resolutions never
enter model input text.

## ⚠️ Known limitations

- Stage 2/3 datasets are small (806 / 262 issues) and class-imbalanced;
  per-project estimates have high variance.
- Only four projects (AXIS, HTTPD, Linux, MySQL) back the cross-project
  claims; no independent fifth project was available for external validation.
- The cascade result measures subtype recovery among *known* bugs — it is
  not a global six-class production accuracy.
- Stage 1 development splits are stratified, not repository-aware (the
  NLBSE archive exposed no repository metadata).
