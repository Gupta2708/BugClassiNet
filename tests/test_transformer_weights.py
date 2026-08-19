import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from bugclassinet.models.transformer_classifier import _balanced_weights


def test_streamed_class_counts_preserve_balanced_weight_formula() -> None:
    labels = ["BUG", "DOCUMENTATION", "QUESTION"]
    counts = {"BUG": 5, "DOCUMENTATION": 2, "QUESTION": 3}
    truth = [label for label in labels for _ in range(counts[label])]

    expected = compute_class_weight("balanced", classes=np.asarray(labels), y=np.asarray(truth))

    np.testing.assert_allclose(_balanced_weights(labels, counts), expected)
