import pandas as pd

from bugclassinet.models.tfidf_classifier import train_tfidf


def test_tfidf_trains_on_tiny_fixture(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "text": ["crash on save", "feature request", "why fails", "update docs"],
            "canonical_label": ["BUG", "ENHANCEMENT", "QUESTION", "DOCUMENTATION"],
        }
    )
    model = train_tfidf(frame, tmp_path / "model.joblib")
    assert (tmp_path / "model.joblib").is_file()
    assert model.predict(["crash now"])[0] == "BUG"
