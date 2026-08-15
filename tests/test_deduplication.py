import pandas as pd

from bugclassinet.data.deduplicate import (
    deduplicate_train,
    duplicate_overlap,
    ensure_unique_issue_ids,
    exclude_test_overlap,
)


def test_deduplicate_and_overlap() -> None:
    train = pd.DataFrame({"text_hash": ["a", "a", "b"], "issue_id": [1, 2, 3]})
    test = pd.DataFrame({"text_hash": ["a", "c"], "issue_id": [4, 5]})
    deduped, removed = deduplicate_train(train)
    assert removed == 1
    assert deduped["issue_id"].tolist() == [1, 3]
    assert duplicate_overlap(deduped, test)["issue_id"].tolist() == [4]


def test_duplicate_source_ids_are_disambiguated_without_dropping_records() -> None:
    frame = pd.DataFrame(
        {"issue_id": [7, 7], "text_hash": ["a" * 64, "b" * 64], "text": ["first", "second"]}
    )
    result, affected = ensure_unique_issue_ids(frame)
    assert affected == 2
    assert result["source_issue_id"].tolist() == ["7", "7"]
    assert result["issue_id"].is_unique


def test_exclude_test_overlap_does_not_modify_official_test() -> None:
    train = pd.DataFrame({"text_hash": ["a", "b"], "issue_id": [1, 2]})
    official_test = pd.DataFrame({"text_hash": ["a", "c"], "issue_id": [3, 4]})
    before = official_test.copy(deep=True)
    clean, removed = exclude_test_overlap(train, official_test)
    assert removed == 1
    assert clean["text_hash"].tolist() == ["b"]
    pd.testing.assert_frame_equal(official_test, before)
