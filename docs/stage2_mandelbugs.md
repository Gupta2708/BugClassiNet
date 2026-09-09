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
| MySQL 21704 | HTTP 403 (Akamai edge, host-wide) | Resolved by `--allow-web-archive` |
| HTTPD 7441 | HTTP 200 anti-bot challenge | Resolved by the authenticated Bugzilla REST API |
| AXIS AXIS-1 | HTTP 200; description parsed | Verify more records during enrichment |

These are environment- and time-specific results, not guarantees about all IDs.
An unknown layout produces PARSE_ERROR, an empty description EMPTY_CONTENT.
401/403/429 or a recognized anti-bot page opens a host circuit for that run.
When several records describe one issue, a transient failure (`RATE_LIMITED`,
`AUTH_REQUIRED`, `AUTH_FAILED`, `NETWORK_ERROR`) always outranks the terminal
`NOT_FOUND` claim, and every status has a distinct rank, so no reported status
depends on candidate order. Reporting "this issue does not exist" for a merely
blocked retrieval misdirects the operator into abandoning recoverable data.
Redirects are recorded as UNSUPPORTED for manual verification, not blindly
followed. Requests use a descriptive User-Agent, serial delay, bounded retries
and exponential backoff for connection/5xx errors. `Retry-After` is recorded;
rate-limited hosts are not retried in the same run. Responses and metadata use
atomic cache writes; raw caches are checksum-checked when reused.

### Preserved first cloud retrieval

The first anonymous Kaggle enrichment at Git commit
`a0ba6dc53663de969422aaf0c7bbd9b97c90ed54` is preserved as evidence. It
processed all 852 source annotations and observed 487 SUCCESS, 364
AUTH_REQUIRED, and one EMPTY_CONTENT result:

| Project | SUCCESS | AUTH_REQUIRED | EMPTY_CONTENT |
|---|---:|---:|---:|
| Linux | 289 | 0 | 0 |
| AXIS | 198 | 0 | 1 |
| HTTPD | 0 | 143 | 0 |
| MySQL | 0 | 221 | 0 |

This was an anonymous/cloud result, not evidence that the underlying reports
are unavailable. The sampled MySQL historical page is publicly accessible in a
normal browser, while Kaggle was blocked. Retrieve MySQL from a permitted local
environment and import its successful `reports.parquet`. Access Apache HTTPD
only through its documented authenticated Bugzilla REST API. No browser
spoofing, proxy rotation, authentication bypass, anti-bot bypass, or repeated
403 retry is permitted.

### Portable MySQL and authenticated HTTPD completion

One-report local MySQL smoke test (always use a dedicated output directory):

```powershell
.\.venv\Scripts\python.exe -m bugclassinet.cli mandelbugs-enrich `
  --labels data\interim\mandelbugs_labels\labels.parquet `
  --output-dir outputs\smoke\mysql-21704 `
  --cache-dir data\interim\mandelbugs_mysql_cache `
  --manual-dir data\manual_enrichment `
  --retry-failures `
  --only-issue MySQL:21704
```

If that succeeds, collect all MySQL reports into a new local output directory:

```powershell
.\.venv\Scripts\python.exe -m bugclassinet.cli mandelbugs-enrich `
  --labels data\interim\mandelbugs_labels\labels.parquet `
  --output-dir data\interim\mandelbugs_mysql_reports `
  --cache-dir data\interim\mandelbugs_mysql_cache `
  --manual-dir data\manual_enrichment `
  --sleep-seconds 1.0 `
  --retry-failures `
  --only-project MySQL
```

The project selector prevents this local collection run from contacting other
trackers.

#### When the MySQL origin refuses every automated client

Observed on 2026-09-09 from a local desktop: `https://bugs.mysql.com/bug.php?id=21704`
and the site root both return **HTTP 403** from `AkamaiGHost` with Oracle's generic
"Technical Difficulties" page. The block is host-wide and client-based, not per
issue and not an authentication prompt, so no key, delay, or retry resolves it.
Browser spoofing, proxy rotation and anti-bot bypass remain prohibited.

`--allow-web-archive` adds an explicit, opt-in fallback to the public Internet
Archive for MySQL only. It queries the documented CDX index for captures of the
same tracker URL, walks forward from the **earliest** `statuscode:200` capture
through at most `WEB_ARCHIVE_MAX_SNAPSHOTS` (8) candidates, and fetches each
with the `id_` modifier so the archived bytes arrive without Wayback's injected
banner or rewritten links. The first capture that parses as a real report wins:
a crawler can capture a login wall instead of the report, and one unusable
capture must not discard an otherwise recoverable issue. `web_archive_attempts`
records every capture tried and its outcome. A rate-limited or blocked archive
ends the walk and is reported as such, never as "no usable capture". The
existing MySQL parser then applies unchanged. The
blocked origin is contacted once, fails, and is never retried; the archive is a
different public source, not a bypass of Oracle's edge.

```powershell
.\.venv\Scripts\python.exe -m bugclassinet.cli mandelbugs-enrich `
  --labels data\interim\mandelbugs_labels\labels.parquet `
  --output-dir data\interim\mandelbugs_mysql_reports `
  --cache-dir data\interim\mandelbugs_mysql_cache `
  --manual-dir data\manual_enrichment `
  --sleep-seconds 3.0 `
  --retry-failures `
  --allow-web-archive `
  --only-project MySQL
```

Records use `retrieval_source=WEB_ARCHIVE` and carry `archive_timestamp` and
`archive_url`; `retrieval_audit.json` reports `allow_web_archive` and
`web_archive_rows`. `_quality` ranks `WEB_ARCHIVE` below live `AUTO_REMOTE` and
`PRIOR_REPORT` but above manual imports, so a later direct retrieval supersedes an
archived one. An origin failure that is more informative than "no snapshot" still
wins the merge, and the archive outcome is preserved in `web_archive_status`. An
archived login or challenge capture is rejected as `EMPTY_CONTENT`, never accepted
as evidence. The archive rate-limits: use `--sleep-seconds 3.0` or higher, since
each issue costs one index query plus one capture fetch.

Two limitations belong in any write-up. Coverage is partial — issues with no
capture are a genuine `NOT_FOUND` and a source of selection bias. And capture
dates differ per issue, so publish `archive_timestamp` alongside the retrieval
date; an early capture is closer to the historical report than today's page, but
it is still not guaranteed to be the report as filed. Upload `reports.parquet` as a private Kaggle dataset and pass it with
repeatable `--prior-reports`. Only SUCCESS rows
with nonblank descriptions are eligible. The audit records the imported file
name, SHA-256, row count, eligible rows, and reused identities. Final rows use
`retrieval_source=PRIOR_REPORT` and retain the upstream source value. No source
machine path is written to the manifest.

Apache Bugzilla documents
[`GET /rest/bug/{id}`](https://bz.apache.org/bugzilla/docs/en/html/api/core/v1/bug.html)
and
[`GET /rest/bug/{id}/comment`](https://bz.apache.org/bugzilla/docs/en/html/api/core/v1/comment.html),
with comment zero being the description. The adapter uses those JSON endpoints
and the documented `Bugzilla_api_key` call argument.
Authentication follows the official
[Bugzilla REST general API documentation](https://bz.apache.org/bugzilla/docs/en/html/api/core/v1/general.html);
no HTML-login automation is used.
For the ASF deployment, requests use the native `/bugzilla/rest.cgi` entry
point because its optional `/bugzilla/rest` rewrite returns an HTML 404. The
canonical documented `Bugzilla_api_key` parameter is used.
It extracts summary, creation time, public comment zero, later public comments,
OS, hardware, and component metadata. Private comments are excluded. A requested
id can be an alias of, or have been merged into, a different canonical issue;
that response is still evidence for the annotated bug, so it is retained with
`tracker_resolved_bug_id` and `tracker_id_substituted` recording the
substitution, and comments are read under the canonical id the server returned.
Responses carrying no identifier, or more than one bug, are still rejected. Status and
resolution can exist in the raw response but are never copied into default model
text. API endpoints saved to disk contain no credentials.

Credentials are read only from `BUGCLASSINET_APACHE_BUGZILLA_API_KEY`. They are
not accepted as CLI arguments and are never printed, included in exceptions,
manifests, cache metadata, or saved endpoint URLs. Missing credentials produce
AUTH_REQUIRED; rejected credentials produce AUTH_FAILED; 429 produces
RATE_LIMITED; 404/410 produces NOT_FOUND. Authentication and rate-limit failures
open a host circuit for the current run. To retry after replacing an invalid key,
pass `--retry-failures`.

In Kaggle, load the secret immediately before enrichment:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["BUGCLASSINET_APACHE_BUGZILLA_API_KEY"] = UserSecretsClient().get_secret(
    "APACHE_BUGZILLA_API_KEY"
)
```

Then merge the current Linux/AXIS report snapshot, local MySQL results, manual
AXIS fallback, and authenticated HTTPD retrieval:

```bash
python -m bugclassinet.cli mandelbugs-enrich \
  --labels /kaggle/working/stage2/labels/labels.parquet \
  --output-dir /kaggle/working/stage2/final_reports \
  --cache-dir /kaggle/working/stage2/final_cache \
  --prior-reports /kaggle/input/first-cloud-reports/reports.parquet \
  --prior-reports /kaggle/input/local-mysql-reports/reports.parquet \
  --manual-dir /kaggle/input/mandelbugs-manual-imports
```

For a later entirely offline reconstruction, supply every saved successful
snapshot plus manual imports and add `--offline`. Offline always forbids network
access, even with `--retry-failures`. A previous successful record, prior report,
and manual record are compared deterministically: usable content outranks any
failure, more populated content fields outrank fewer, and only materially richer
text replaces otherwise equivalent stable evidence. Source type is the final
tie-breaker. `merge_candidate_sources` records the sources considered.

Both enrichment and preparation write `stage2_readiness.csv/json`. The table
reports source rows, unique IDs, successful and classified-successful reports,
BOH/NAM/ARB/MANDELBUG usable counts, UNK count, and coverage for every project.
`recommended_for_four_project_lopo` becomes true only when all Linux, MySQL,
HTTPD, and AXIS data are present and each has usable BOH and MANDELBUG examples.
Preparation recomputes readiness after duplicate/conflict quarantine; use that
final value before LOPO training.

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

### Observed completed retrieval

Recorded on 2026-09-09, after the authenticated HTTPD route and the MySQL
archive fallback. These are observed run outputs, not asserted constants:

| Project | Unique IDs | Successful | Coverage | BOH usable | MANDELBUG usable |
|---|---:|---:|---:|---:|---:|
| HTTPD | 143 | 143 | 1.000 | 116 | 25 |
| MySQL | 216 | 213 | 0.986 | 123 | 77 |

HTTPD retrieval reproduces the official annotation table exactly
(116 BOH, 15 NAM, 10 ARB, 2 UNK). Two earlier HTTPD attempts are instructive
and are recorded here because both produced misleading audits rather than
obvious errors:

- An unauthenticated attempt cached `NOT_FOUND` for the whole project. ASF
  Bugzilla now requires login even for public bugs, and its REST API answers
  anonymous requests with HTTP 401.
- A later authenticated attempt retrieved 30 issues in 110 seconds and then
  tripped the rate limiter. The remaining 113 issues were never requested, and
  the stale `NOT_FOUND` records outranked the fresh rate-limit records under the
  previous tie-breaking rule, so the audit reported 113 nonexistent bugs. The
  status ranking above was corrected in response. Use a fresh output directory
  and `--sleep-seconds 3.0` for a full-project ASF run.

Three MySQL issues (`48993`, `50451`, `56982`) have no capture of any status in
the archive and remain unretrieved; `48993` is already quarantined as a
duplicate-ID conflict, so two classified issues are lost. Five further issues
were recovered only by the capture walk, needing between two and five candidates
each, and would have been reported as `EMPTY_CONTENT` by a single-capture
strategy.

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
