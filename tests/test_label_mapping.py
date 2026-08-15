from bugclassinet.data.harmonize import canonicalize_stage1_label


def test_stage1_label_mapping() -> None:
    assert canonicalize_stage1_label("bug") == "BUG"
    assert canonicalize_stage1_label("feature request") == "ENHANCEMENT"
    assert canonicalize_stage1_label("question") == "QUESTION"
    assert canonicalize_stage1_label("documentation") == "DOCUMENTATION"
