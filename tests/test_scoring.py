from __future__ import annotations

import unittest

from mechbench.interface.schemas import FinalReport, InterventionPrediction
from mechbench.scoring.causal_classification import score_causal_classification
from mechbench.scoring.intervention_prediction import score_intervention_predictions
from mechbench.registry import Registry


class ScoringTest(unittest.TestCase):
    def test_causal_score_rewards_true_probability(self) -> None:
        high = FinalReport.from_dict(
            {
                "episode_id": "x",
                "raw_improvement": {},
                "fair_comparison": {},
                "causal_probabilities": {"parameter_laundering": 1.0},
                "mechanism_support": "x",
            }
        )
        low = FinalReport.from_dict(
            {
                "episode_id": "x",
                "raw_improvement": {},
                "fair_comparison": {},
                "causal_probabilities": {"true_mechanism": 1.0},
                "mechanism_support": "x",
            }
        )
        self.assertGreater(
            score_causal_classification(high, "parameter_laundering"),
            score_causal_classification(low, "parameter_laundering"),
        )

    def test_intervention_score_uses_private_outcomes(self) -> None:
        world = Registry("families").worlds["fixture/parameter_laundering"]
        predictions = [
            InterventionPrediction(
                intervention_id=item.intervention_id,
                predicted_delta_mean=float(item.expected_delta or 0.0),
                lower_90=float(item.expected_delta or 0.0) - 0.001,
                upper_90=float(item.expected_delta or 0.0) + 0.001,
                qualitative_result=item.qualitative_result or "unknown",
                confidence=0.9,
            )
            for item in world.interventions
        ]
        self.assertGreater(score_intervention_predictions(predictions, world.interventions), 35.0)


if __name__ == "__main__":
    unittest.main()

