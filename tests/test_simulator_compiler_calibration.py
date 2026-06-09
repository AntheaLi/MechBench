from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mechbench.simulator.calibration import calibrate_simulator, public_only_world_classifier_accuracy
from mechbench.simulator.compiler import SimulatorEpisodeCompiler, WORLD_TYPES
from mechbench.registry import Registry
from mechbench.utils.config import load_config


class SimulatorCompilerCalibrationTest(unittest.TestCase):
    def test_compiler_writes_balanced_opaque_worlds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "worlds"
            result = SimulatorEpisodeCompiler(output_dir=output_dir, seed=11).compile(count=10)
            self.assertEqual(len(result.episode_ids), 10)
            self.assertEqual(set(result.counts_by_world_type), set(WORLD_TYPES))
            self.assertTrue(all(count == 2 for count in result.counts_by_world_type.values()))
            for episode_id in result.episode_ids:
                self.assertRegex(episode_id, r"^simulator/compiled_\d{4}$")
                world_name = episode_id.split("/", 1)[1]
                config = load_config(output_dir / world_name / "world.yaml")
                self.assertEqual(config["world_id"], episode_id)
                self.assertNotIn(config["causal_label"], episode_id)
                self.assertTrue(config["certificate"]["certified"])

    def test_public_only_classifier_runs(self) -> None:
        registry = Registry("families")
        world_ids = [
            "simulator/true_mechanism_001",
            "simulator/seed_hacking_001",
            "simulator/parameter_laundering_001",
            "simulator/norm_laundering_001",
            "simulator/compute_laundering_lite_001",
        ]
        accuracy = public_only_world_classifier_accuracy(registry, world_ids)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_calibration_report_runs_on_manual_worlds(self) -> None:
        report = calibrate_simulator(
            families_dir="families",
            selector="manual",
            agent_names=["oracle", "scripted"],
        ).to_dict()
        self.assertEqual(report["world_count"], 5)
        self.assertGreaterEqual(report["agents"]["oracle"]["mean"], 92.0)
        self.assertIn("scripted_score_65_to_80", report["gates"])

    def test_calibration_report_contains_per_world_and_per_component_breakdowns(self) -> None:
        report = calibrate_simulator(
            families_dir="families",
            selector="manual",
            agent_names=["oracle", "scripted"],
        ).to_dict()

        # per_world_type: each agent has an entry per world type present
        self.assertIn("per_world_type", report)
        for agent_name in ("oracle", "scripted"):
            self.assertIn(agent_name, report["per_world_type"])
            agent_world_breakdown = report["per_world_type"][agent_name]
            self.assertEqual(set(agent_world_breakdown.keys()), set(WORLD_TYPES))
            for label, summary in agent_world_breakdown.items():
                self.assertIn("mean", summary)
                self.assertIn("min", summary)
                self.assertIn("max", summary)
                self.assertIn("median", summary)

        # per_component: each agent has mean scores for every scoring component
        self.assertIn("per_component", report)
        expected_components = {
            "heldout_intervention_prediction",
            "causal_world_diagnosis",
            "improvement_validity",
            "calibration",
            "evidence_grounding",
            "experimental_efficiency",
            "protocol_integrity",
        }
        for agent_name in ("oracle", "scripted"):
            self.assertIn(agent_name, report["per_component"])
            self.assertEqual(set(report["per_component"][agent_name].keys()), expected_components)
            # Oracle should get near-perfect causal diagnosis
            if agent_name == "oracle":
                self.assertGreaterEqual(report["per_component"]["oracle"]["causal_world_diagnosis"], 19.0)


if __name__ == "__main__":
    unittest.main()

