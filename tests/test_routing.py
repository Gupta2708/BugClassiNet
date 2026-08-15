from bugclassinet.inference.pipeline import HierarchicalPipeline
from bugclassinet.inference.schemas import IssueInput


class Predictor:
    def __init__(self, label: str) -> None:
        self.label, self.calls = label, 0

    def predict(self, texts: list[str]) -> list[str]:
        self.calls += 1
        return [self.label]


def test_non_bug_stops_after_stage1() -> None:
    one, two, three = Predictor("QUESTION"), Predictor("BOH"), Predictor("ARB")
    result = HierarchicalPipeline(one, two, three).predict_one(IssueInput(title="how"))
    assert result.final_label == "QUESTION"
    assert (two.calls, three.calls) == (0, 0)


def test_bug_boh_stops_after_stage2() -> None:
    one, two, three = Predictor("BUG"), Predictor("BOH"), Predictor("ARB")
    result = HierarchicalPipeline(one, two, three).predict_one(IssueInput(title="broken"))
    assert result.final_label == "BOH"
    assert three.calls == 0


def test_man_routes_to_stage3() -> None:
    one, two, three = Predictor("BUG"), Predictor("MAN"), Predictor("NAM")
    result = HierarchicalPipeline(one, two, three).predict_one(IssueInput(title="wrong"))
    assert result.final_label == "NAM"
    assert three.calls == 1
