import pandas as pd

from bugclassinet.data.split import make_validation_split


def test_split_has_no_id_overlap() -> None:
    frame = pd.DataFrame(
        {"issue_id": range(12), "canonical_label": ["BUG", "ENHANCEMENT"] * 6, "text": ["x"] * 12}
    )
    train, validation, strategy = make_validation_split(frame, 0.25, seed=7)
    assert not (set(train.issue_id) & set(validation.issue_id))
    assert strategy == "stratified_no_repository_metadata"
