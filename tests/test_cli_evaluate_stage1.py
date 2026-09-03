from pathlib import Path

from bugclassinet.cli import build_parser
from bugclassinet.training import commands


def test_evaluate_stage1_cli_accepts_new_and_legacy_option_names(tmp_path, monkeypatch) -> None:
    captured = []

    def fake_evaluate(model, data, output):
        captured.append((model, data, output))
        return {"accuracy": 1.0}

    monkeypatch.setattr(commands, "evaluate_saved_stage1", fake_evaluate)
    parser = build_parser()
    for options in (
        ["--model", "model", "--data", "validation.parquet"],
        ["--model-path", "model", "--test", "validation.parquet"],
    ):
        args = parser.parse_args(["evaluate-stage1", *options, "--output-dir", str(tmp_path)])
        commands.evaluate_stage1(args)

    assert captured == [
        (Path("model"), "validation.parquet", str(tmp_path)),
        (Path("model"), "validation.parquet", str(tmp_path)),
    ]
