# Stage-2 Mandelbugs research workflow

Stage 1 is **complete and frozen**. Its model, configurations, checkpoint resume
semantics, official-test outputs, and existing commands are unchanged. This work
adds a separate Stage-2 experiment suite, not a retraining of Stage 1.

Stage 2 predicts `BOH` versus `MANDELBUG` on known bug reports. `BOH` maps to BOH;
`NAM` and `ARB` map to MANDELBUG. `UNK` is retained separately. Stage 3 (NAM versus
ARB) is planned; its labels are preserved but no Stage-3 training is implemented
by this workflow. Existing legacy `train-stage2` remains available unchanged;
use the new commands below for the BOH/MANDELBUG LOPO research protocol.

No Stage-2 research performance is claimed by this implementation. Tests train
only a tiny synthetic TF-IDF fixture. Frozen Stage-1 transfer is explicitly an
optional baseline, not ModernBERT and not an end-to-end hierarchy evaluation.

## Source and observed audit

Roberto Natella, *Mandelbugs in Open-Source Software* (dataset, 2017),
[DOI: 10.5281/zenodo.581660](https://doi.org/10.5281/zenodo.581660).
The [dataset repository](https://openscience.us/repo/software-aging/mandelbugs.html)
identifies the original paper:

Domenico Cotroneo, Michael Grottke, Roberto Natella, Roberto Pietrantuono, and
Kishor S. Trivedi. **Fault Triggers in Open-Source Software: An Experience
Report.** 24th IEEE International Symposium on Software Reliability Engineering
(ISSRE), 2013, pp. 178–187.

The local 12-file audit observed the following annotations. These numbers are
reported, **not asserted or hard-coded** by the parser:

| Project | BOH | NAM | ARB | UNK | Total |
|---|---:|---:|---:|---:|---:|
| Linux | 122 | 121 | 24 | 22 | 289 |
| MySQL | 125 | 67 | 17 | 12 | 221 |
| HTTPD | 116 | 15 | 10 | 2 | 143 |
| AXIS | 184 | 7 | 8 | 0 | 199 |
| Total | 547 | 210 | 59 | 36 | 852 |

There are 847 unique project/bug IDs. Eight rows belong to three duplicate-ID
groups, all within individual MySQL files, not across files:

- `49324`: BOH versus NAM conflict; quarantine all its annotations.
- `48993`: repeated ARB plus NAM. Although the binary label agrees, the original
  classification conflicts; conservatively quarantine this ID as well.
- `56709`: three identical ARB annotations; retain one supervised issue.

All 852 rows parsed; no missing/invalid IDs. The AXIS file declares numeric IDs
but stores `AXIS-…` identifiers. The parser preserves those identifiers and
records this declaration anomaly. Literal `N/A` is a valid original subclass,
not a missing-value marker. Missing optional subclasses are null.

Before retrieval losses or text conflicts, the conservative annotation policy
allows at most 809 supervised issues (546 BOH, 263 MANDELBUG). The final training
row count must come from `stage2_dataset_audit.json`, not this upper bound.

The parser handles the downloaded dense ARFF dialect, quoted nominal fields,
comments, and optional subclasses. Unsupported sparse/multiline records and
malformed fields are written to `parse_failures.csv` and fail the audit. It does
not silently skip rows or pretend to support arbitrary ARFF schemas.

## Install and prepare

Use an isolated environment. Install retrieval-only dependencies for CPU data
work without downloading model packages; install neural extras only when needed:

```bash
python -m pip install -e . requests beautifulsoup4
# For SBERT / ModernBERT / optional frozen Stage-1 transfer:
python -m pip install -e ".[stage2,transformers]"
```

```bash
python -m bugclassinet.cli mandelbugs-audit --raw-dir data/raw/mandelbugs --output-dir data/interim/mandelbugs_labels
python -m bugclassinet.cli mandelbugs-enrich --labels data/interim/mandelbugs_labels/labels.parquet --output-dir data/interim/mandelbugs_reports --cache-dir data/interim/mandelbugs_cache --sleep-seconds 1.0 --manual-dir data/manual_enrichment
```

Inspect `retrieval_by_project.csv`, `retrieval_failures.csv`, and several actual
reports **before** proceeding. Check coverage by project and original class:
retrieval failures may introduce selection bias. If blocked, use documented
[manual imports](../data/manual_enrichment/README.md); rerun with `--offline` to
prohibit network access. Successful cached/prior records are kept. Failures are
cached too; `--retry-failures` retries only failed records after legitimate access
is restored. Offline mode overrides retries. Keep raw response caches and their
hashes with your private reproducibility snapshot.

Read-only probes on 2026-09-09 (one issue per tracker, not a full coverage test):

| Tracker/sample | Observed result | Action |
|---|---|---|
| Linux 1436 | HTTP 200; description parsed | Verify more records during enrichment |
| MySQL 21704 | HTTP 403 | Manual/authorized export required from this environment |
| HTTPD 7441 | HTTP 200 anti-bot challenge | Manual/authorized export required; no bypass |
| AXIS AXIS-1 | HTTP 200; description parsed | Verify more records during enrichment |

These are environment- and time-specific results, not guarantees about all IDs.
An unknown layout produces PARSE_ERROR, an empty description EMPTY_CONTENT.
401/403/429 or a recognized anti-bot page opens a host circuit for that run.
Redirects are recorded as UNSUPPORTED for manual verification, not blindly
followed. Requests use a descriptive User-Agent, serial delay, bounded retries
and exponential backoff for connection/5xx errors. `Retry-After` is recorded;
rate-limited hosts are not retried in the same run. Responses and metadata use
atomic cache writes; raw caches are checksum-checked when reused.

```bash
python -m bugclassinet.cli mandelbugs-prepare --labels data/interim/mandelbugs_labels/labels.parquet --reports data/interim/mandelbugs_reports/reports.parquet --output-dir data/processed/mandelbugs_stage2
```

Preparation re-derives labels from the official table and rebuilds input text
from allowlisted raw fields, not a supplied `text` column. It writes:

- `stage2.parquet`: successful, nonblank, unambiguous classified issues.
- `stage2_unknown.parquet`: **all** UNK annotations, including failed retrievals.
- `conflicting_ids.csv`, `conflicting_labels.csv`, `duplicate_annotations.csv`,
  `duplicate_text.csv`, `unenriched_classified.csv`, and a hashed dataset audit.

Text is `[TITLE]`, `[DESCRIPTION]`, `[ENVIRONMENT]`; full evidence adds
`[COMMENTS]`. Project/subsystem/class/subclass/status/resolution metadata is never
inserted into model input. Natural issue prose can itself mention projects or
fixes; no claim of semantic anonymization is made. Full comments can reveal
post-resolution knowledge, so report `initial` and `full` as separate evidence
conditions. Neither is guaranteed to reconstruct the exact historical report
at issue creation; publish retrieval dates and this limitation.

## Evaluation protocol and baselines

The prepared dataset must include all four projects and both binary classes.
There are four **leave-one-project-out** outer folds, always in Linux, MySQL,
HTTPD, AXIS order. Every issue is tested exactly once per seed. Matching
normalized held-out text is removed only from that fold's training portion;
the held-out project is never reduced. These removals are listed in manifests.

Neural validation is a deterministic, stratified, text-group-disjoint inner
partition of the other three projects. With enough groups, approximately 20%
is held out; fewer independent groups use fewer folds. The split must have both
classes, or it fails. It never uses outer test labels for early stopping.
Fixed classical baselines fit on all eligible outer training rows. They share
exactly the same outer folds with ModernBERT; neural fit rows exclude inner
validation, which is explicitly recorded.

```bash
python -m bugclassinet.cli train-stage2-baseline --data data/processed/mandelbugs_stage2/stage2.parquet --model tfidf_svm --evidence-mode initial --config configs/models/stage2_tfidf_svm.yaml --output-dir outputs/stage2/tfidf_initial
python -m bugclassinet.cli train-stage2-baseline --data data/processed/mandelbugs_stage2/stage2.parquet --model sbert_logreg --evidence-mode initial --config configs/models/stage2_sbert_logreg.yaml --cache-dir data/interim/stage2_embeddings --output-dir outputs/stage2/sbert_initial
```

Repeat with `--evidence-mode full` and **new output directories** for the full
evidence condition. Nonempty experiment destinations are rejected to protect
existing results. No large hyperparameter sweep or outer-project tuning is
implemented. Review convergence warnings rather than reporting an unconverged
model as finalized.

TF-IDF uses sparse float32 word unigrams/bigrams, min_df=2, max_df=1.0, at most
50,000 features, and balanced LinearSVC (C=1, max_iter=5000, seed=42). Vocabulary
and IDF fit only on training projects. SBERT uses frozen all-mpnet-base-v2,
its saved pooling, L2 normalization, max_length=384, and balanced liblinear
LogisticRegression (C=1). Embeddings are computed without gradients and cached
by ordered text plus encoder identity; caches have SHA-256 checks. Label-free
frozen embeddings may be cached for the entire dataset; only the linear head
is fit per training fold. Model parameter hashes must agree before/after
embedding extraction.

Optional transfer (not a change to Stage 1):

```bash
python -m bugclassinet.cli train-stage2-baseline --data data/processed/mandelbugs_stage2/stage2.parquet --model frozen_stage1_encoder_transfer --stage1-model PATH_TO_COMPLETE_STAGE1_MODEL --evidence-mode initial --cache-dir data/interim/stage2_embeddings --output-dir outputs/stage2/frozen_stage1_initial
```

Transfer uses the existing strict completed-checkpoint loader, saved tokenizer
and max_length, attention-masked mean pooling from the frozen base encoder,
L2 normalization, and balanced LogisticRegression. It never saves into or
trains the Stage-1 checkpoint.

## ModernBERT (GPU, after baselines)

```bash
python -m bugclassinet.cli train-stage2-modernbert --data data/processed/mandelbugs_stage2/stage2.parquet --evidence-mode initial --config configs/models/stage2_modernbert.yaml --output-dir outputs/stage2/modernbert_initial --seeds 13 42 97
```

Pretrained `answerdotai/ModernBERT-base`, not training from scratch. Defaults:
max_length=256, batch=8, AdamW lr=1e-5, weight_decay=0.01, gradient clipping=1.0,
at most 10 epochs, patience=2 on **inner Macro-F1**. Class weights are
`inner_train_rows / (2 * inner_class_count)` in `[BOH, MANDELBUG]` order.
Inference is unweighted argmax. Dynamic padding occurs per batch; tokenization
is lazy and loaders use zero workers. Each fold/seed starts afresh from the
same resolved pretrained revision. Strict improvement wins; ties keep the
earlier epoch. The outer test is predicted once, after restoring the best
inner-validation model. It is not used to choose an epoch.

This separate trainer uses one GPU, float32 and eager attention for compatibility;
it does not introduce Stage-1 precision modes, distributed training, DAPT, or
resume changes. CUDA deterministic algorithms are requested with warnings for
unsupported operations. Bitwise equality across different hardware/library
versions is not guaranteed; package versions and seeds are recorded. Pin
`model_revision`/`encoder_revision` to the recorded commit for a replication.
The model config hash and saved parameter/file hashes are also recorded.

`freeze_lower_layers: N` freezes embeddings and the first N encoder layers;
its manifest strategy is explicitly distinct from full fine-tuning. Keep this
as a separately named experiment; do not choose N by outer test scores.

Outputs contain seed/project directories with split identities, epoch histories,
class weights, best models/tokenizers and predictions. These are **Stage-2 model
artifacts**, not resumable Stage-1 Trainer checkpoints. Preserve outputs between
Kaggle sessions; do not assume this new trainer resumes partial folds.

## Reporting and interpretation

Each experiment writes `summary.json`, `per_fold_metrics.csv`,
`aggregate_metrics.json`, `classification_report.csv`, `confusion_matrix.csv`,
`predictions.parquet`, `run_manifest.json`, `bootstrap_ci.json`, and
`per_project_seed_summary.csv`. Each also includes a fold-wise training-majority
baseline under `majority/`; ties select BOH. Predictions retain issue identity,
project (audit only), true/predicted labels, seed, normalized text group, and
MANDELBUG score with its semantics.

Macro-F1 is primary. Reports include class precision/recall/F1/support, MCC,
balanced accuracy, accuracy, and MANDELBUG PR-AUC defined as **average precision**
(not trapezoidal PR integration). PR-AUC is null for a single-class test fold.
Confusion matrices use true rows and predicted columns in `[BOH, MANDELBUG]`
order. SVM scores are margins, not probabilities. Aggregate prediction-level
Macro-F1 and the equally weighted mean per-project Macro-F1 are both reported.
Multiple seeds are summarized with mean and sample standard deviation; each
seed's predictions and metrics remain available.

95% intervals use a deterministic percentile bootstrap, resampling text groups
within projects and pairing seeds for the same issues. They are conditional on
these four observed projects, **not** uncertainty over a population of new
projects, and they do not refit models. Small project/class supports and
retrieval-induced selection bias must accompany results.

Use [the thin Kaggle notebook](../notebooks/kaggle/12_stage2_mandelbugs_lopo.ipynb)
to run the same CLI. CPU suffices for audit, enrichment, preparation, TF-IDF,
and SBERT (GPU optional for embeddings). ModernBERT explicitly requires GPU.
No NLBSE test data is loaded by this Stage-2 workflow.
