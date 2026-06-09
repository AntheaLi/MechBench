from __future__ import annotations

import unittest

from agents.oracle import OracleAgent
from agents.scripted_causal_control import ScriptedCausalControlAgent
from mechbench.core.budget import Budget
from mechbench.core.experiment_runner import ExperimentRunner
from mechbench.interface.schemas import ExperimentRequest
from mechbench.registry import Registry


class SimulatorParameterLaunderingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = Registry("families")
        self.episode = self.registry.get_episode("simulator/parameter_laundering_001")
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
        self.assertIn("simulator/parameter_laundering_001", self.registry.list_worlds())
        self.assertEqual(self.world.causal_label, "parameter_laundering")

    def test_matched_learned_controls_recover_gain(self) -> None:
        proposed = self.run_variant("proposed")
        generic = self.run_variant("generic_learned_branch")
        wider = self.run_variant("parameter_matched_wider")
        destroyed = self.run_variant("geometry_destroyed_branch")
        fixed = self.run_variant("fixed_random_branch")

        proposed_delta = proposed.metrics["validation_id.delta"]
        generic_delta = generic.metrics["validation_id.delta"]
        wider_delta = wider.metrics["validation_id.delta"]
        destroyed_delta = destroyed.metrics["validation_id.delta"]
        fixed_delta = fixed.metrics["validation_id.delta"]

        self.assertGreater(proposed_delta, 0.030)
        self.assertGreater(generic_delta, proposed_delta - 0.004)
        self.assertGreater(wider_delta, proposed_delta - 0.005)
        self.assertGreater(destroyed_delta, proposed_delta - 0.005)
        self.assertLess(fixed_delta, proposed_delta - 0.025)

    def test_multiseed_control_stays_positive(self) -> None:
        panel = self.run_variant("multi_seed_replication", {"n_seeds": 7})
        self.assertGreater(panel.metrics["validation_id.delta"], 0.030)
        self.assertLess(panel.metrics["validation_id.delta_std"], 0.002)

    def test_private_intervention_pattern(self) -> None:
        interventions = {item.intervention_id: item for item in self.world.interventions}
        self.assertEqual(
            interventions["hidden_equal_param_generic_learned_branch"].qualitative_result,
            "gain_survives",
        )
        self.assertEqual(
            interventions["hidden_destroy_geometry_preserve_norm"].qualitative_result,
            "gain_survives",
        )
        self.assertEqual(
            interventions["hidden_fixed_random_branch"].qualitative_result,
            "gain_disappears",
        )

    def test_oracle_scores_high_on_parameter_laundering(self) -> None:
        result = self.episode.run(OracleAgent())
        self.assertGreater(result.score.total, 92.0)

    def test_scripted_agent_detects_parameter_laundering(self) -> None:
        result = self.episode.run(ScriptedCausalControlAgent())
        self.assertGreater(result.report.causal_probabilities["parameter_laundering"], 0.80)
        self.assertEqual(result.report.mechanism_support, "claimed_mechanism_not_supported")
        self.assertGreaterEqual(result.score.total, 65.0)
        self.assertLessEqual(result.score.total, 85.0)  # per-episode; Phase 1 gate bounds the mean


if __name__ == "__main__":
    unittest.main()
