import math
from pathlib import Path

import numpy as np
import pytest
from sklearn.utils.class_weight import compute_class_weight

from bugclassinet.models.transformer_classifier import (
    TransformerTrainingConfig,
    _balanced_weights,
    _resolve_class_weights,
)
from bugclassinet.settings import load_yaml


def test_streamed_class_counts_preserve_balanced_weight_formula() -> None:
    labels = ["BUG", "DOCUMENTATION", "QUESTION"]
    counts = {"BUG": 5, "DOCUMENTATION": 2, "QUESTION": 3}
    truth = [label for label in labels for _ in range(counts[label])]

    expected = compute_class_weight("balanced", classes=np.asarray(labels), y=np.asarray(truth))

    np.testing.assert_allclose(_balanced_weights(labels, counts), expected)


def test_full_train_balanced_weights_reproduce_existing_values() -> None:
    labels = ["BUG", "DOCUMENTATION", "ENHANCEMENT", "QUESTION"]
    counts = {
        "BUG": 579_398,
        "DOCUMENTATION": 48_664,
        "ENHANCEMENT": 394_835,
        "QUESTION": 66_797,
    }

    actual = _resolve_class_weights(labels, counts, "balanced")

    np.testing.assert_allclose(
        actual,
        [0.47018370791752817, 5.59804989314483, 0.68996796130029, 4.078379268530024],
    )


def test_sqrt_balanced_is_exact_square_root_of_balanced() -> None:
    labels = ["BUG", "DOCUMENTATION", "QUESTION"]
    counts = {"BUG": 5, "DOCUMENTATION": 2, "QUESTION": 3}
    balanced = _resolve_class_weights(labels, counts, "balanced")
    sqrt_balanced = _resolve_class_weights(labels, counts, "sqrt_balanced")

    np.testing.assert_allclose(sqrt_balanced, np.sqrt(balanced))


def test_quarter_balanced_is_exact_fourth_root_of_balanced() -> None:
    labels = ["BUG", "DOCUMENTATION", "ENHANCEMENT", "QUESTION"]
    counts = {
        "BUG": 579_398,
        "DOCUMENTATION": 48_664,
        "ENHANCEMENT": 394_835,
        "QUESTION": 66_797,
    }
    balanced = _resolve_class_weights(labels, counts, "balanced")
    quarter_balanced = _resolve_class_weights(labels, counts, "quarter_balanced")

    np.testing.assert_allclose(quarter_balanced, np.power(balanced, 0.25))


def test_none_returns_unweighted_cross_entropy_marker() -> None:
    assert _resolve_class_weights(["BUG", "QUESTION"], {"BUG": 8, "QUESTION": 2}, "none") is None


def test_custom_weights_are_used_exactly_without_normalization() -> None:
    labels = ["BUG", "DOCUMENTATION", "QUESTION"]
    custom = {"BUG": 0.75, "DOCUMENTATION": 2.25, "QUESTION": math.pi}

    actual = _resolve_class_weights(
        labels,
        {"BUG": 5, "DOCUMENTATION": 2, "QUESTION": 3},
        "custom",
        custom,
    )

    np.testing.assert_array_equal(actual, [0.75, 2.25, math.pi])


def test_invalid_weight_strategy_fails_clearly() -> None:
    with pytest.raises(ValueError, match="Invalid class_weight_strategy"):
        TransformerTrainingConfig(model_name="tiny", class_weight_strategy="invalid")


def test_200k_ablation_configs_change_only_weight_strategy() -> None:
    config_dir = Path("configs/models")
    names = {
        "balanced": "deberta_stage1_200k_balanced_256.yaml",
        "quarter_balanced": "deberta_stage1_200k_quarter_balanced_256.yaml",
        "sqrt_balanced": "deberta_stage1_200k_sqrt_balanced_256.yaml",
        "none": "deberta_stage1_200k_unweighted_256.yaml",
    }
    loaded = {strategy: load_yaml(config_dir / name) for strategy, name in names.items()}
    without_strategy = {
        strategy: {key: value for key, value in config.items() if key != "class_weight_strategy"}
        for strategy, config in loaded.items()
    }

    assert len({repr(config) for config in without_strategy.values()}) == 1
    for strategy, config in loaded.items():
        assert config["class_weight_strategy"] == strategy
        assert config["max_length"] == 256
        assert config["epochs"] == 1
        assert config["seed"] == 42

    length_ablation = load_yaml(config_dir / "deberta_stage1_200k_bestweight_384.yaml")
    assert length_ablation["max_length"] == 384
    assert length_ablation["class_weight_strategy"] == "replace_after_256_ablation"
