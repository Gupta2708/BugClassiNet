# Data layout

Place unmodified source archives under `data/raw/<dataset>/`. Processed Parquet,
splits, samples, and gold data are ignored by Git. Source archive and output
SHA-256 values are recorded in `data_manifest.json` after preparation.
