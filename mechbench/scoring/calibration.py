"""Calibration scoring."""

from __future__ import annotations

from mechbench.interface.schemas import WORLD_LABELS, FinalReport, InterventionPrediction, InterventionSpec


def score_calibration(
    report: FinalReport,
    predictions: list[InterventionPrediction],
    interventions: list[InterventionSpec],
    true_label: str,
    labels: tuple[str, ...] = WORLD_LABELS,
    weight: float = 10.0,
) -> float:
    probabilities = report.normalized_probabilities(labels)
    brier = sum((probabilities[label] - (1.0 if label == true_label else 0.0)) ** 2 for label in labels)
    chance_brier = 1.0 - (1.0 / len(labels))
    world_score = max(0.0, (chance_brier - brier) / chance_brier)

    prediction_by_id = {prediction.intervention_id: prediction for prediction in predictions}
    intervention_scores = []
    for intervention in interventions:
        if intervention.expected_delta is None:
            continue
        prediction = prediction_by_id.get(intervention.intervention_id)
        if prediction is None:
            intervention_scores.append(0.0)
            continue
        lower = min(prediction.lower_90, prediction.upper_90)
        upper = max(prediction.lower_90, prediction.upper_90)
        contains = lower <= float(intervention.expected_delta) <= upper
        interval_target = 1.0 if contains else 0.0
        confidence = prediction.confidence
        brier_skill = (0.25 - (confidence - interval_target) ** 2) / 0.25
        intervention_scores.append(max(0.0, brier_skill))

    intervention_score = sum(intervention_scores) / len(intervention_scores) if intervention_scores else world_score
    return weight * (0.6 * world_score + 0.4 * intervention_score)
