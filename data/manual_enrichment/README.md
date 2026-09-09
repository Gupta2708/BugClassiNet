# Historical report imports

Put legitimately obtained CSV/Parquet exports here when automated retrieval is
blocked or an old tracker layout is unsupported. Do not bypass authentication,
CAPTCHAs, anti-bot pages, or rate limits. These files are ignored by Git.

Required columns (blank comments/environment are allowed):

```text
project,bug_id,title,initial_description,comments_text,environment_text,tracker_url
```

Project aliases normalize to Linux, MySQL, HTTPD, AXIS. AXIS accepts `1` or
`AXIS-1`; other IDs are positive integers. The description must be nonblank.
Use real issue evidence, never invented descriptions or label-derived text.
Do not copy entire pages or metadata tables into a text column. Keep later
comments out of `initial_description`. Optional `created_at` is preserved.

The importer checks IDs against the official labels, rejects conflicting manual
rows, hashes import files, and records `retrieval_source=MANUAL_IMPORT`. A prior
successful record wins over a replacement. To intentionally investigate a revised
import, use a **new output directory** and preserve the old snapshot.

Run enrichment with `--manual-dir data/manual_enrichment --offline` to merge
imports and cached records without network access. Failed records stay in the
output and failure audit. Never claim that the resulting subset is the entire
original dataset unless coverage supports it.
