import pandas as pd
from scipy import sparse

from bugclassinet.features.tfidf import TfidfFeatureConfig, make_tfidf_features
from bugclassinet.models.tfidf_classifier import train_tfidf


def test_tfidf_trains_on_tiny_fixture(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "text": ["crash on save", "feature request", "why fails", "update docs"],
            "canonical_label": ["BUG", "ENHANCEMENT", "QUESTION", "DOCUMENTATION"],
        }
    )
    model = train_tfidf(
        frame,
        tmp_path / "model.joblib",
        {
            "word_min_df": 1,
            "word_max_df": 1.0,
            "char_min_df": 1,
            "char_max_df": 1.0,
            "word_max_features": 50,
            "char_max_features": 50,
        },
    )
    assert (tmp_path / "model.joblib").is_file()
    assert (tmp_path / "training_config.json").is_file()
    assert model.predict(["crash now"])[0] == "BUG"


def test_word_char_features_stay_sparse_float32() -> None:
    config = TfidfFeatureConfig(
        word_max_features=10,
        char_max_features=12,
        word_min_df=1,
        char_min_df=1,
        word_max_df=1.0,
        char_max_df=1.0,
    )
    matrix = make_tfidf_features(config).fit_transform(["crash save", "feature request"])
    assert sparse.issparse(matrix)
    assert matrix.dtype.name == "float32"
    assert matrix.shape[1] <= 22


def test_word_only_configuration_omits_character_features() -> None:
    config = TfidfFeatureConfig(
        feature_mode="word",
        word_max_features=10,
        word_min_df=1,
        word_max_df=1.0,
    )
    features = make_tfidf_features(config)
    matrix = features.fit_transform(["crash save", "feature request"])
    assert sparse.issparse(matrix)
    assert matrix.dtype.name == "float32"
    assert not hasattr(features, "transformer_list")
