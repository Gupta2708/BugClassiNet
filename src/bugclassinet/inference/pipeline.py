"""Model-agnostic conditional inference pipeline."""

from __future__ import annotations

from typing import Protocol

from bugclassinet.inference.routing import route_labels
from bugclassinet.inference.schemas import HierarchyPrediction, IssueInput


class TextPredictor(Protocol):
    """Minimal predictor protocol compatible with sklearn and adapter models."""

    def predict(self, texts: list[str]) -> list[str]: ...


class HierarchicalPipeline:
    """Run later stages only when routing conditions are met."""

    def __init__(self, stage1: TextPredictor, stage2: TextPredictor, stage3: TextPredictor) -> None:
        self.stage1, self.stage2, self.stage3 = stage1, stage2, stage3

    def predict_one(self, issue: IssueInput) -> HierarchyPrediction:
        """Classify one issue through only its applicable hierarchy stages."""
        text = [issue.text]
        stage1_label = self.stage1.predict(text)[0]
        if stage1_label != "BUG":
            return route_labels(stage1_label)
        stage2_label = self.stage2.predict(text)[0]
        if stage2_label == "BOH":
            return route_labels(stage1_label, stage2_label)
        return route_labels(stage1_label, stage2_label, self.stage3.predict(text)[0])
