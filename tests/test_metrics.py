from bugclassinet.evaluation.metrics import classification_metrics


def test_metrics_report_expected_fields() -> None:
    metrics = classification_metrics(["BUG", "QUESTION"], ["BUG", "BUG"])
    assert {"macro_f1", "micro_f1", "weighted_f1", "mcc", "balanced_accuracy"} <= set(metrics)
    assert metrics["confusion_matrix"] == [[1, 0], [1, 0]]
