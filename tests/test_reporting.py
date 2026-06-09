from __future__ import annotations

import unittest

from analysis.reporting import render_calibration_report


class ReportingTest(unittest.TestCase):
    def test_renders_enriched_calibration_report(self) -> None:
        markdown = render_calibration_report(
            {
                "world_count": 2,
                "public_only_world_classification_accuracy": 0.25,
                "gates": {"oracle_score_at_least_92": True},
                "agents": {
                    "oracle": {"mean": 98.0, "median": 98.0, "min": 97.5, "max": 98.5}
                },
                "per_component": {
                    "oracle": {
                        "causal_world_diagnosis": 20.0,
                        "heldout_intervention_prediction": 39.0,
                    }
                },
                "per_world_type": {
                    "oracle": {
                        "true_mechanism": {"mean": 98.0, "median": 98.0, "min": 98.0, "max": 98.0}
                    }
                },
                "per_episode": [
                    {
                        "world_id": "simulator/compiled_0001",
                        "causal_label": "true_mechanism",
                        "oracle": 98.0,
                    }
                ],
            }
        )
        self.assertIn("# MechanismBench Calibration Report", markdown)
        self.assertIn("oracle_score_at_least_92", markdown)
        self.assertIn("heldout_intervention_prediction", markdown)
        self.assertIn("simulator/compiled_0001", markdown)


if __name__ == "__main__":
    unittest.main()
