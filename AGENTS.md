# BugClassiNet-Next contributor guide

Keep reusable code in `src/bugclassinet`; notebooks only orchestrate package APIs.
Do not commit datasets, model checkpoints, embeddings, or generated outputs. Run
`python -m ruff check .`, `python -m ruff format --check .`, and `pytest` before
submitting a change. Preserve official test sets and never place labels, statuses,
or resolutions in model input text.
