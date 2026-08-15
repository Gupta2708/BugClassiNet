import pytest

from bugclassinet.data.schema import SchemaInferenceError, infer_issue_schema


def test_schema_detects_common_columns() -> None:
    schema = infer_issue_schema(["number", "Title", "description", "labels", "repo"])
    assert schema.issue_id == "number"
    assert schema.title == "Title"
    assert schema.body == "description"
    assert schema.original_label == "labels"
    assert schema.repository == "repo"


def test_schema_fails_with_columns() -> None:
    with pytest.raises(SchemaInferenceError, match="Available columns"):
        infer_issue_schema(["foo", "bar"])
