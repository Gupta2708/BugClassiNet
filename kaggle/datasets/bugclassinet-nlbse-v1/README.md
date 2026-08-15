# Kaggle dataset manifest

Stage the processed Parquet and metadata files with `scripts/create_kaggle_bundle.py`.
The script writes individual files under `kaggle/upload/bugclassinet-nlbse-v1`
for direct use by the Kaggle CLI. Do not upload checkpoints, embeddings, or outputs.
