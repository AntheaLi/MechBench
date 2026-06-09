from __future__ import annotations

import unittest

from agents.oracle import OracleAgent
from agents.scripted_causal_control import ScriptedCausalControlAgent
from mechbench.core.budget import Budget
from mechbench.core.experiment_runner import ExperimentRunner
from mechbench.interface.schemas import ExperimentRequest
from mechbench.registry import Registry


class SimulatorNormLaunderingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = Registry("families")
        self.episode = self.registry.get_episode("simulator/norm_laundering_001")
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

    def test_discovers_world(self) -> None:
        self.assertIn("simulator/norm_laundering_001", self.registry.list_worlds())
        self.assertEqual(self.world.causal_label, "norm_laundering")

    def test_performance_tracks_branch_norm(self) -> None:
        proposed = self.run_variant("proposed")
        norm_matched = self.run_variant("norm_matched_proposed")
        destroyed = self.run_variant("geometry_destroyed_branch")
        scale_generic = self.run_variant("scale_matched_generic")
        generic = self.run_variant("generic_learned_branch")

        proposed_delta = proposed.metrics["validation_id.delta"]
        self.assertGreater(proposed_delta, 0.030)
        self.assertGreater(proposed.model_stats["branch_base_norm_ratio"], 1.30)
        self.assertLess(norm_matched.metrics["validation_id.delta"], proposed_delta - 0.020)
        self.assertLess(norm_matched.model_stats["branch_base_norm_ratio"], 0.70)
        self.assertGreater(destroyed.metrics["validation_id.delta"], proposed_delta - 0.004)
        self.assertGreater(scale_generic.metrics["validation_id.delta"], proposed_delta - 0.002)
        self.assertLess(generic.metrics["validation_id.delta"], proposed_delta - 0.020)

    def test_multiseed_control_stays_positive(self) -> None:
        panel = self.run_variant("multi_seed_replication", {"n_seeds": 7})
        self.assertGreater(panel.metrics["validation_id.delta"], 0.030)
        self.assertLess(panel.metrics["validation_id.delta_std"], 0.002)

    def test_oracle_scores_high_on_norm_laundering(self) -> None:
        result = self.episode.run(OracleAgent())
        self.assertGreater(result.score.total, 92.0)

    def test_scripted_agent_detects_norm_laundering(self) -> None:
        result = self.episode.run(ScriptedCausalControlAgent())
        self.assertGreater(result.report.causal_probabilities["norm_laundering"], 0.80)
        self.assertEqual(result.report.mechanism_support, "claimed_mechanism_not_supported")
        self.assertGreaterEqual(result.score.total, 65.0)
        self.assertLessEqual(result.score.total, 85.0)  # per-episode; Phase 1 gate bounds the mean


if __name__ == "__main__":
    unittest.main()
