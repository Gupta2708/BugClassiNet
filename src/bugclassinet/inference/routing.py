"""Pure hierarchy routing rules."""

from __future__ import annotations

from bugclassinet.inference.schemas import HierarchyPrediction


def route_labels(
    stage1: str, stage2: str | None = None, stage3: str | None = None
) -> HierarchyPrediction:
    """Validate conditional routing and return a final hierarchical prediction."""
    if stage1 != "BUG":
        return HierarchyPrediction(stage1=stage1)
    if stage2 is None:
        raise ValueError("BUG Stage 1 predictions require a Stage 2 label")
    if stage2 == "BOH":
        return HierarchyPrediction(stage1="BUG", stage2="BOH")
    if stage2 != "MAN":
        raise ValueError(f"Unexpected Stage 2 label: {stage2}")
    if stage3 not in {"ARB", "NAM"}:
        raise ValueError("MAN Stage 2 predictions require ARB or NAM Stage 3 label")
    return HierarchyPrediction(stage1="BUG", stage2="MAN", stage3=stage3)
