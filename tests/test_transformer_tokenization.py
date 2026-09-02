from __future__ import annotations

from typing import Any

from bugclassinet.evaluation.metrics import classification_metrics
from bugclassinet.models.transformer_classifier import (
    _tokenization_fingerprint,
    _tokenize_batch,
)


class DeterministicTokenizer:
    pad_token_id = 99

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _encode(text: str, max_length: int) -> list[int]:
        word_ids = [200 + sum(map(ord, word)) % 997 for word in text.split()]
        values = [101, *word_ids, 102]
        if len(values) > max_length:
            values = [*values[: max_length - 1], 102]
        return values

    def __call__(
        self,
        texts: list[str],
        *,
        truncation: bool,
        max_length: int,
        padding: bool | str,
    ) -> dict[str, list[list[int]]]:
        self.calls.append(
            {
                "truncation": truncation,
                "max_length": max_length,
                "padding": padding,
            }
        )
        assert truncation is True
        encoded = [self._encode(text, max_length) for text in texts]
        masks = [[1] * len(values) for values in encoded]
        if padding == "max_length":
            for values, mask in zip(encoded, masks, strict=True):
                padding_size = max_length - len(values)
                values.extend([self.pad_token_id] * padding_size)
                mask.extend([0] * padding_size)
        elif padding is not False:
            raise AssertionError(f"Unexpected tokenizer padding mode: {padding!r}")
        return {"input_ids": encoded, "attention_mask": masks}


class DynamicCollator:
    def __init__(self, tokenizer: DeterministicTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, list[Any]]:
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[int] = []
        for feature in features:
            padding_size = width - len(feature["input_ids"])
            input_ids.append(
                [*feature["input_ids"], *([self.tokenizer.pad_token_id] * padding_size)]
            )
            attention_mask.append([*feature["attention_mask"], *([0] * padding_size)])
            labels.append(feature["labels"])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class FingerprintDataset:
    _fingerprint = "source-fixture"

    def __len__(self) -> int:
        return 4


class FingerprintTokenizer:
    name_or_path = "microsoft/deberta-v3-small"
    vocab_size = 128_000

    def __init__(self, commit: str) -> None:
        self.init_kwargs = {"_commit_hash": commit}


def test_token_cache_fingerprint_is_stable_and_revision_aware() -> None:
    arguments = (
        FingerprintDataset(),
        "train",
        "microsoft/deberta-v3-small",
        FingerprintTokenizer("revision-a"),
        256,
        {"BUG": 0, "QUESTION": 1},
    )

    first = _tokenization_fingerprint(*arguments)
    second = _tokenization_fingerprint(*arguments)
    changed = _tokenization_fingerprint(
        *arguments[:3], FingerprintTokenizer("revision-b"), *arguments[4:]
    )

    assert first == second
    assert first != changed


def _legacy_encode(
    rows: list[dict[str, str]],
    tokenizer: DeterministicTokenizer,
    label_to_id: dict[str, int],
    max_length: int,
) -> dict[str, list[Any]]:
    encoded = tokenizer(
        [row["text"] for row in rows],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    encoded["labels"] = [label_to_id[row["canonical_label"]] for row in rows]
    return encoded


def _predictions(batch: dict[str, list[Any]], label_count: int) -> list[int]:
    return [
        sum(token * mask for token, mask in zip(tokens, masks, strict=True)) % label_count
        for tokens, masks in zip(batch["input_ids"], batch["attention_mask"], strict=True)
    ]


def test_dynamic_tokenization_matches_legacy_semantics_and_metrics() -> None:
    rows = [
        {"text": "crash now", "canonical_label": "BUG"},
        {"text": "improve installation guide", "canonical_label": "DOCUMENTATION"},
        {
            "text": "add configurable dark theme to every project dashboard today please",
            "canonical_label": "ENHANCEMENT",
        },
        {"text": "why timeout", "canonical_label": "QUESTION"},
    ]
    labels = sorted({row["canonical_label"] for row in rows})
    label_to_id = {label: index for index, label in enumerate(labels)}
    max_length = 8

    legacy_tokenizer = DeterministicTokenizer()
    legacy = _legacy_encode(rows, legacy_tokenizer, label_to_id, max_length)
    dynamic_tokenizer = DeterministicTokenizer()
    collator = DynamicCollator(dynamic_tokenizer)

    legacy_predictions: list[int] = []
    dynamic_predictions: list[int] = []
    dynamic_widths: list[int] = []
    for start in range(0, len(rows), 2):
        batch_rows = rows[start : start + 2]
        encoded = _tokenize_batch(
            {
                "text": [row["text"] for row in batch_rows],
                "canonical_label": [row["canonical_label"] for row in batch_rows],
            },
            dynamic_tokenizer,
            label_to_id,
            max_length,
        )
        features = [
            {column: values[index] for column, values in encoded.items()}
            for index in range(len(batch_rows))
        ]
        dynamic = collator(features)
        width = len(dynamic["input_ids"][0])
        dynamic_widths.append(width)

        expected = {
            "input_ids": [values[:width] for values in legacy["input_ids"][start : start + 2]],
            "attention_mask": [
                values[:width] for values in legacy["attention_mask"][start : start + 2]
            ],
            "labels": legacy["labels"][start : start + 2],
        }
        assert dynamic == expected
        legacy_predictions.extend(_predictions(expected, len(labels)))
        dynamic_predictions.extend(_predictions(dynamic, len(labels)))

    assert dynamic_widths[0] < max_length
    assert dynamic_widths[1] == max_length
    assert dynamic_tokenizer.calls == [
        {"truncation": True, "max_length": max_length, "padding": False},
        {"truncation": True, "max_length": max_length, "padding": False},
    ]
    assert legacy_predictions == dynamic_predictions

    truth = [row["canonical_label"] for row in rows]
    legacy_output = [labels[index] for index in legacy_predictions]
    dynamic_output = [labels[index] for index in dynamic_predictions]
    assert classification_metrics(truth, legacy_output) == classification_metrics(
        truth, dynamic_output
    )


def test_max_length_changes_only_truncation_not_labels() -> None:
    batch = {
        "text": ["one two three four five six", "short"],
        "canonical_label": ["QUESTION", "BUG"],
    }
    label_to_id = {"BUG": 0, "QUESTION": 1}
    tokenizer = DeterministicTokenizer()

    at_four = _tokenize_batch(batch, tokenizer, label_to_id, max_length=4)
    at_eight = _tokenize_batch(batch, tokenizer, label_to_id, max_length=8)

    assert len(at_four["input_ids"][0]) == 4
    assert len(at_eight["input_ids"][0]) == 8
    assert at_four["labels"] == at_eight["labels"] == [1, 0]
    assert tokenizer.calls == [
        {"truncation": True, "max_length": 4, "padding": False},
        {"truncation": True, "max_length": 8, "padding": False},
    ]
