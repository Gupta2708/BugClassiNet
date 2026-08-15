"""Typed inference input and output values."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IssueInput:
    """Inference input, deliberately excluding labels and resolution data."""

    title: str = ""
    description: str = ""

    @property
    def text(self) -> str:
        return f"[TITLE]\n{self.title.strip()}\n\n[DESCRIPTION]\n{self.description.strip()}"


@dataclass(frozen=True)
class HierarchyPrediction:
    """Prediction returned by the conditional hierarchy."""

    stage1: str
    stage2: str | None = None
    stage3: str | None = None

    @property
    def final_label(self) -> str:
        return self.stage3 or self.stage2 or self.stage1
