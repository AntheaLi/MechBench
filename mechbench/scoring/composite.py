"""Composite scoring."""

from __future__ import annotations

from mechbench.core.evidence_ledger import EvidenceLedger
from mechbench.core.family import World
from mechbench.interface.schemas import EpisodeScore, FinalReport, InterventionPrediction
from mechbench.scoring.calibration import score_calibration
from mechbench.scoring.causal_classification import score_causal_classification
from mechbench.scoring.efficiency import score_efficiency
from mechbench.scoring.evidence_grounding import score_evidence_grounding
from mechbench.scoring.improvement_validity import score_improvement_validity
from mechbench.scoring.intervention_prediction import score_intervention_predictions
from mechbench.scoring.protocol_integrity import score_protocol_integrity


def score_episode(
    world: World,
    report: FinalReport,
    predictions: list[InterventionPrediction],
    ledger: EvidenceLedger,
) -> EpisodeScore:
    tolerance = float(world.certificate.get("prediction_tolerance", 0.05) or 0.05)
    components = {
        "heldout_intervention_prediction": score_intervention_predictions(
            predictions, world.interventions, tolerance=tolerance
        ),
        "causal_world_diagnosis": score_causal_classification(report, world.causal_label),
        "improvement_validity": score_improvement_validity(report, world.config),
        "calibration": score_calibration(report, predictions, world.interventions, world.causal_label),
        "evidence_grounding": score_evidence_grounding(report, ledger),
        "experimental_efficiency": score_efficiency(ledger, world.certificate),
        "protocol_integrity": score_protocol_integrity(report, ledger),
    }
    return EpisodeScore(total=sum(components.values()), components=components)

