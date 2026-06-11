"""Causal-world diagnosis scoring."""

from __future__ import annotations

from mechbench.interface.schemas import WORLD_LABELS, FinalReport


def score_causal_classification(
    report: FinalReport,
    true_label: str,
    labels: tuple[str, ...] = WORLD_LABELS,
    weight: float = 20.0,
) -> float:
    probabilities = report.normalized_probabilities(labels)
    brier = 0.0
    for label in labels:
        target = 1.0 if label == true_label else 0.0
        brier += (probabilities[label] - target) ** 2
    chance_brier = 1.0 - (1.0 / len(labels))
    skill = (chance_brier - brier) / chance_brier
    return weight * max(0.0, skill)
