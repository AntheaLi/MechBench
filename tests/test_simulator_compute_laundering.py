from __future__ import annotations

import unittest

from agents.oracle import OracleAgent
from agents.scripted_causal_control import ScriptedCausalControlAgent
from mechbench.core.budget import Budget
from mechbench.core.experiment_runner import ExperimentRunner
from mechbench.interface.schemas import ExperimentRequest
from mechbench.registry import Registry


class SimulatorComputeLaunderingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = Registry("families")
        self.episode = self.registry.get_episode("simulator/compute_laundering_lite_001")
        self.family = self.episode.family
        self.world = self.episode.world

    def run_variant(self, variant: str, params: dict | None = None):
        runner = ExperimentRunner(self.family, self.world, Budget(self.world.budget))
        return runner.run(
            ExperimentRequest.from_dict(
                {
                    "variant": variant,
                    "train_seed": 0,
                    "data_seed": 0,
                    "eval_splits": ["validation_id"],
                    "collect": ["activation_norms"],
                    "params": params or {},
                }
            )
        )

    def test_discovers_world_and_compute_actions(self) -> None:
        self.assertIn("simulator/compute_laundering_lite_001", self.registry.list_worlds())
        self.assertEqual(self.world.causal_label, "compute_laundering_lite")
        actions = {action.name for action in self.family.get_available_actions(self.world)}
        self.assertIn("baseline_compute_matched", actions)
        self.assertIn("proposed_compute_limited", actions)

    def test_compute_matching_removes_apparent_advantage(self) -> None:
        proposed = self.run_variant("proposed")
        baseline_compute = self.run_variant("baseline_compute_matched")
        proposed_limited = self.run_variant("proposed_compute_limited")
        learning_curve = self.run_variant("matched_flop_learning_curve")

        proposed_delta = proposed.metrics["validation_id.delta"]
        self.assertGreater(proposed_delta, 0.030)
        self.assertGreater(proposed.model_stats["estimated_train_flops"], baseline_compute.model_stats["estimated_train_flops"] - 1)
        self.assertGreater(baseline_compute.metrics["validation_id.delta"], proposed_delta - 0.006)
        self.assertLess(proposed_limited.metrics["validation_id.delta"], proposed_delta - 0.020)
        self.assertLess(learning_curve.metrics["validation_id.delta"], proposed_delta - 0.020)

    def test_equal_step_multiseed_still_positive(self) -> None:
        panel = self.run_variant("multi_seed_replication", {"n_seeds": 7})
        self.assertGreater(panel.metrics["validation_id.delta"], 0.030)
        self.assertLess(panel.metrics["validation_id.delta_std"], 0.002)

    def test_oracle_scores_high_on_compute_laundering(self) -> None:
        result = self.episode.run(OracleAgent())
        self.assertGreater(result.score.total, 92.0)

    def test_scripted_agent_detects_compute_laundering(self) -> None:
        result = self.episode.run(ScriptedCausalControlAgent())
        self.assertGreater(result.report.causal_probabilities["compute_laundering_lite"], 0.80)
        self.assertEqual(result.report.mechanism_support, "claimed_mechanism_not_supported")
        self.assertEqual(result.report.fair_comparison["compute_matched"], "gain_does_not_survive")
        self.assertGreaterEqual(result.score.total, 65.0)
        self.assertLessEqual(result.score.total, 85.0)  # per-episode; Phase 1 gate bounds the mean


if __name__ == "__main__":
    unittest.main()
