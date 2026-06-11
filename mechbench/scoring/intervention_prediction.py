"""Held-out intervention prediction scoring."""

from __future__ import annotations

from mechbench.interface.schemas import InterventionPrediction, InterventionSpec


def _interval_score(prediction: InterventionPrediction, truth: float, tolerance: float) -> float:
    lower = min(prediction.lower_90, prediction.upper_90)
    upper = max(prediction.lower_90, prediction.upper_90)
    width = max(upper - lower, 0.0)
    contains = lower <= truth <= upper
    if contains:
        width_penalty = min(width / max(tolerance, 1e-9), 1.0)
        return max(0.0, 1.0 - width_penalty)
    miss = min(abs(truth - lower), abs(truth - upper))
    return max(0.0, 1.0 - miss / max(tolerance, 1e-9))


def score_intervention_predictions(
    predictions: list[InterventionPrediction],
    interventions: list[InterventionSpec],
    weight: float = 40.0,
    tolerance: float = 0.05,
) -> float:
    scored_interventions = [item for item in interventions if item.expected_delta is not None]
    if not scored_interventions:
        return 0.0
    by_id = {prediction.intervention_id: prediction for prediction in predictions}
    total = 0.0
    for intervention in scored_interventions:
        prediction = by_id.get(intervention.intervention_id)
        if prediction is None:
            continue
        truth = float(intervention.expected_delta)
        numerical = max(0.0, 1.0 - abs(prediction.predicted_delta_mean - truth) / tolerance) ** 2
        qualitative = (
            1.0
            if intervention.qualitative_result
            and prediction.qualitative_result == intervention.qualitative_result
            else 0.0
        )
        uncertainty = _interval_score(prediction, truth, tolerance)
        intervention_score = 0.50 * numerical + 0.25 * qualitative + 0.25 * uncertainty
        if prediction.qualitative_result == "uncertain":
            intervention_score = min(intervention_score, 0.35)
        total += intervention_score
    return weight * total / len(scored_interventions)
