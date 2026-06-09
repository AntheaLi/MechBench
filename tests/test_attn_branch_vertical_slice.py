from __future__ import annotations

import importlib.util
import unittest

from agents.oracle import OracleAgent
from mechbench.core.budget import Budget
from mechbench.core.experiment_runner import ExperimentRunner
from mechbench.interface.schemas import ExperimentRequest
from mechbench.registry import Registry


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class AttnBranchVerticalSliceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = Registry("families")

    def test_registers_two_certified_episodes_per_phase2_world(self) -> None:
        worlds = [
            world for world in self.registry.worlds.values()
            if world.family == "attn_branch" and world.certificate.get("vertical_slice")
        ]
        counts: dict[str, int] = {}
        for world in worlds:
            counts[world.causal_label] = counts.get(world.causal_label, 0) + 1
            self.assertTrue(world.certificate.get("certified"))
            self.assertTrue(world.certificate.get("vertical_slice"))
        self.assertEqual(
            counts,
            {
                "true_mechanism": 2,
                "seed_hacking": 2,
                "parameter_laundering": 2,
                "norm_laundering": 2,
            },
        )

    def test_runs_tiny_pytorch_branch_experiment(self) -> None:
        episode = self.registry.get_episode("attn_branch/true_mechanism_001")
        runner = ExperimentRunner(episode.family, episode.world, Budget(episode.world.budget))
        result = runner.run(
            ExperimentRequest.from_dict(
                {
                    "variant": "proposed",
                    "eval_splits": ["validation_id"],
                    "collect": ["activation_norms"],
                    "hypothesis_tested": "The structured branch should improve retrieval.",
                }
            )
        )
        self.assertGreater(result.metrics["validation_id.delta"], 0.02)
        self.assertEqual(result.model_stats["branch_type"], "geometric")
        self.assertIn("branch_base_norm_ratio", result.model_stats)
        self.assertIn("estimated_train_flops", result.model_stats)

    def test_seed_panel_collapses_seed_hacking_gain(self) -> None:
        episode = self.registry.get_episode("attn_branch/seed_hacking_001")
        runner = ExperimentRunner(episode.family, episode.world, Budget(episode.world.budget))
        panel = runner.run(
            ExperimentRequest.from_dict(
                {
                    "variant": "multi_seed_replication",
                    "eval_splits": ["validation_id"],
                    "params": {"n_seeds": 3},
                    "hypothesis_tested": "The public gain should replicate across seeds.",
                }
            )
        )
        public_delta = (
            episode.world.headline.proposed_metrics["validation_id.accuracy"]
            - episode.world.headline.baseline_metrics["validation_id.accuracy"]
        )
        self.assertLess(panel.metrics["validation_id.delta"], public_delta - 0.015)

    def test_oracle_solves_vertical_slice_episode(self) -> None:
        result = self.registry.get_episode("attn_branch/true_mechanism_001").run(OracleAgent())
        self.assertGreaterEqual(result.score.total, 90.0)


if __name__ == "__main__":
    unittest.main()
