"""Improvement validity scoring."""

from __future__ import annotations

from typing import Any

from mechbench.interface.schemas import FinalReport


def score_improvement_validity(
    report: FinalReport,
    world_config: dict[str, Any],
    weight: float = 10.0,
) -> float:
    truth = world_config.get("improvement_truth", {})
    true_status = truth.get("status")
    true_delta = truth.get("delta")
    reported_status = report.raw_improvement.get("status")
    score = 0.0
    if true_status and reported_status == true_status:
        score += 0.6
    if true_delta is not None:
        try:
            reported_delta = float(report.raw_improvement.get("estimated_delta"))
            error = abs(reported_delta - float(true_delta))
            score += 0.4 * max(0.0, 1.0 - error / 0.05)
        except (TypeError, ValueError):
            pass
    elif true_status is None:
        score += 0.6
    return weight * min(score, 1.0)

