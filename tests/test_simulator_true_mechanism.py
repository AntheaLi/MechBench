from __future__ import annotations

import unittest

from agents.oracle import OracleAgent
from agents.scripted_causal_control import ScriptedCausalControlAgent
from mechbench.core.budget import Budget
from mechbench.core.experiment_runner import ExperimentRunner
from mechbench.interface.schemas import ExperimentRequest
from mechbench.registry import Registry


class SimulatorTrueMechanismTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = Registry("families")
        self.episode = self.registry.get_episode("simulator/true_mechanism_001")
        self.family = self.episode.family
        self.world = self.episode.world

    def run_variant(self, variant: str, splits: list[str] | None = None) -> dict:
        runner = ExperimentRunner(self.family, self.world, Budget(self.world.budget))
        result = runner.run(
            ExperimentRequest.from_dict(
                {
                    "variant": variant,
                    "train_seed": 0,
                    "data_seed": 0,
                    "eval_splits": splits or ["validation_id"],
                    "collect": ["activation_norms"],
                }
            )
        )
        return result.to_dict()

    def test_discovers_world(self) -> None:
        self.assertIn("simulator/true_mechanism_001", self.registry.list_worlds())
        self.assertEqual(self.world.causal_label, "true_mechanism")
        actions = {action.name for action in self.family.get_available_actions(self.world)}
        self.assertIn("multi_seed_replication", actions)

    def test_public_controls_have_true_mechanism_signature(self) -> None:
        proposed = self.run_variant("proposed")
        generic = self.run_variant("generic_learned_branch")
        wider = self.run_variant("parameter_matched_wider")
        destroyed = self.run_variant("geometry_destroyed_branch")
        norm_matched = self.run_variant("norm_matched_proposed")

        proposed_delta = proposed["metrics"]["validation_id.delta"]
        generic_delta = generic["metrics"]["validation_id.delta"]
        wider_delta = wider["metrics"]["validation_id.delta"]
        destroyed_delta = destroyed["metrics"]["validation_id.delta"]
        norm_delta = norm_matched["metrics"]["validation_id.delta"]

        self.assertGreater(proposed_delta, 0.025)
        self.assertLess(generic_delta, proposed_delta - 0.015)
        self.assertLess(wider_delta, proposed_delta - 0.015)
        self.assertLess(destroyed_delta, proposed_delta - 0.020)
        self.assertGreater(norm_delta, proposed_delta - 0.003)

    def test_effect_survives_ood_split(self) -> None:
        result = self.run_variant("proposed", ["validation_long", "validation_position_shift"])
        self.assertGreater(result["metrics"]["validation_long.delta"], 0.025)
        self.assertGreater(result["metrics"]["validation_position_shift.delta"], 0.025)

    def test_multiseed_public_control_is_stable(self) -> None:
        runner = ExperimentRunner(self.family, self.world, Budget(self.world.budget))
        result = runner.run(
            ExperimentRequest.from_dict(
                {
                    "variant": "multi_seed_replication",
                    "eval_splits": ["validation_id"],
                    "params": {"n_seeds": 7},
                    "hypothesis_tested": "The gain should survive unseen seeds.",
                }
            )
        )
        self.assertGreater(result.metrics["validation_id.delta"], 0.030)
        self.assertLess(result.metrics["validation_id.delta_std"], 0.002)

    def test_oracle_scores_high_on_true_mechanism(self) -> None:
        result = self.episode.run(OracleAgent())
        self.assertGreater(result.score.total, 92.0)

    def test_private_intervention_pattern(self) -> None:
        interventions = {item.intervention_id: item for item in self.world.interventions}
        self.assertEqual(len(interventions), 4)
        self.assertEqual(interventions["hidden_destroy_routing_structure"].qualitative_result, "gain_disappears")
        self.assertEqual(interventions["hidden_partial_mechanism_ablation"].qualitative_result, "gain_shrinks")
        self.assertEqual(interventions["hidden_unseen_task_structure_preserved"].qualitative_result, "gain_survives")
        self.assertEqual(interventions["hidden_unseen_seed_panel"].qualitative_result, "gain_survives")

    def test_scripted_agent_solves_true_mechanism(self) -> None:
        result = self.episode.run(ScriptedCausalControlAgent())
        self.assertGreater(result.report.causal_probabilities["true_mechanism"], 0.80)
        self.assertGreaterEqual(result.score.total, 65.0)
        self.assertLessEqual(result.score.total, 85.0)  # per-episode; Phase 1 gate bounds the mean


if __name__ == "__main__":
    unittest.main()
