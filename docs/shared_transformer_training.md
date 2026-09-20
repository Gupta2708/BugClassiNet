# Shared transformer training is label-agnostic

`train_transformer()` in `src/bugclassinet/models/transformer_classifier.py` is
used by Stage 1, Stage 2 and Stage 3. It contains no knowledge of any specific
label. The three stages differ only in the data they pass in:

| Stage | Labels |
|---|---|
| 1 | `BUG`, `DOCUMENTATION`, `ENHANCEMENT`, `QUESTION` |
| 2 | `BOH`, `MANDELBUG` |
| 3 | `ARB`, `NAM` |

## Class weights

Weights are resolved from the observed training labels and counts by
`_resolve_class_weights()`, which is unchanged. `class_weight_strategy` selects
an exponent applied to the balanced weights `w_c = N / (K * n_c)`:

| Strategy | Exponent |
|---|---|
| `none` | no weighting (`None` is returned and the loss is unweighted) |
| `quarter_balanced` | 0.25 |
| `sqrt_balanced` | 0.5 |
| `balanced` | 1.0 |
| `custom` | weights supplied verbatim via `class_weights` |

Nothing in this path depends on which labels are present, only on how many rows
each one has.

## The reference class is a diagnostic only

`_class_weight_diagnostics()` picks the **most frequent training class** as a
reference so the logged weight ratios have a stable denominator, and reports it
as `reference_label` with `weight_ratios_to_reference`. Ties are broken by label
name so the choice is deterministic.

The reference class **never affects the loss**. The optimizer consumes
`selected_weights` exactly as `_resolve_class_weights()` returned it. Changing
which class is the reference changes the log line and the manifest diagnostic,
nothing else. A zero reference weight cannot arise from the supported strategies,
but is handled by emitting `None` ratios rather than dividing by zero.

In practice the reference is `BUG` for Stage 1, `BOH` for Stage 2 and `NAM` for
Stage 3 — derived from the counts, never hard-coded.

## Stable identifiers

Validation predictions are keyed by whichever stable identifier the dataset
carries: `issue_id` (Stage 1, produced by `harmonize`) or `issue_key` (the
Mandelbugs Stage-2/3 datasets). `identifier_column()` resolves this in
preference order and returns `None` when neither exists, in which case
fingerprinting falls back to content columns. The trainer never fabricates an
identifier, and source-identity verification is not weakened: content columns
are always required, and an identifier is verified when the source publishes one.

## Migrating older checkpoints

Run manifests previously recorded `class_weight_ratios_to_bug`. That key is now
`class_weight_ratios_to_reference`, alongside `class_weight_reference_label`.
`_migrate_stored_manifest()` renames the legacy key when validating a stored
manifest, so checkpoints written before this change still resume.

No experimental results are claimed here. This note describes shared
infrastructure only.
