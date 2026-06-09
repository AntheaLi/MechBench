from __future__ import annotations

import unittest

from agents.oracle import OracleAgent
from agents.scripted_causal_control import ScriptedCausalControlAgent
from mechbench.core.budget import Budget
from mechbench.core.experiment_runner import ExperimentRunner
from mechbench.interface.schemas import ExperimentRequest
from mechbench.registry import Registry


class SimulatorSeedHackingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = Registry("families")
        self.episode = self.registry.get_episode("simulator/seed_hacking_001")
        self.family = self.episode.family
        self.world = self.episode.world

    def run_variant(
        self,
        variant: str,
        train_seed: int = 0,
        data_seed: int = 0,
        params: dict | None = None,
    ):
        runner = ExperimentRunner(self.family, self.world, Budget(self.world.budget))
        return runner.run(
            ExperimentRequest.from_dict(
                {
                    "variant": variant,
                    "train_seed": train_seed,
                    "data_seed": data_seed,
                    "eval_splits": ["validation_id"],
                    "params": params or {},
                }
            )
        )

    def test_discovers_world(self) -> None:
        self.assertIn("simulator/seed_hacking_001", self.registry.list_worlds())
        self.assertEqual(self.world.causal_label, "seed_hacking")

    def test_public_demo_seed_is_positive_outlier(self) -> None:
        public_seed = self.run_variant("proposed", train_seed=0, data_seed=0)
        fresh_seed = self.run_variant("proposed", train_seed=4, data_seed=4)
        self.assertGreater(public_seed.metrics["validation_id.delta"], 0.028)
        self.assertLess(fresh_seed.metrics["validation_id.delta"], 0.008)

    def test_multiseed_replication_collapses_gain(self) -> None:
        panel = self.run_variant("multi_seed_replication", params={"n_seeds": 7})
        self.assertLess(panel.metrics["validation_id.delta"], 0.012)
        self.assertGreater(panel.metrics["validation_id.delta_std"], 0.009)

    def test_geometry_and_norm_are_not_certifying_without_seed_panel(self) -> None:
        destroyed = self.run_variant("geometry_destroyed_branch")
        norm_matched = self.run_variant("norm_matched_proposed")
        self.assertLess(destroyed.metrics["validation_id.delta"], 0.008)
        self.assertGreater(norm_matched.metrics["validation_id.delta"], 0.025)

    def test_oracle_scores_high_on_seed_hacking(self) -> None:
        result = self.episode.run(OracleAgent())
        self.assertGreater(result.score.total, 92.0)

    def test_scripted_agent_detects_seed_hacking(self) -> None:
        result = self.episode.run(ScriptedCausalControlAgent())
        self.assertGreater(result.report.causal_probabilities["seed_hacking"], 0.80)
        self.assertEqual(result.report.raw_improvement["status"], "not_reliably_positive")
        self.assertGreaterEqual(result.score.total, 65.0)
        self.assertLessEqual(result.score.total, 85.0)  # per-episode; Phase 1 gate bounds the mean


if __name__ == "__main__":
    unittest.main()
