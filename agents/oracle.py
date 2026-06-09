"""Oracle baseline with private world access.

This is a harness diagnostic, not a valid benchmark participant.
"""

from __future__ import annotations

from agents.base import MechBenchAgent
from mechbench.interface.agent_api import AgentAPI
from mechbench.interface.schemas import WORLD_LABELS, EvidenceItem, FinalReport, InterventionPrediction


class OracleAgent(MechBenchAgent):
    DECISIVE_CONTROL_BY_WORLD = {
        "true_mechanism": "geometry_destroyed_branch",
        "seed_hacking": "multi_seed_replication",
        "parameter_laundering": "generic_learned_branch",
        "norm_laundering": "norm_matched_proposed",
        "compute_laundering_lite": "baseline_compute_matched",
    }

    def investigate(self, api: AgentAPI) -> tuple[FinalReport, list[InterventionPrediction]]:
        actions = api.list_actions()
        action_by_name = {action["name"]: action for action in actions}
        action_names = [action["name"] for action in actions[:2]]
        decisive_control = self.DECISIVE_CONTROL_BY_WORLD.get(api.world.causal_label)
        if decisive_control in action_by_name and decisive_control not in action_names:
            action_names.append(decisive_control)

        evidence_ids = []
        for action_name in action_names:
            params = {"n_seeds": 7} if action_name == "multi_seed_replication" else {}
            result = api.run_experiment(
                {
                    "variant": action_name,
                    "eval_splits": ["validation_id"],
                    "params": params,
                    "hypothesis_tested": "Oracle diagnostic evidence collection.",
                }
            )
            evidence_ids.append(result["experiment_id"])

        probabilities = {label: 0.0 for label in WORLD_LABELS}
        probabilities[api.world.causal_label] = 1.0
        truth = api.world.config.get("improvement_truth", {})
        report = FinalReport(
            episode_id=api.episode_id,
            raw_improvement={
                "status": truth.get("status", "unknown"),
                "confidence": 1.0,
                "estimated_delta": truth.get("delta", 0.0),
            },
            fair_comparison=api.world.config.get("fair_comparison_truth", {}),
            causal_probabilities=probabilities,
            mechanism_support=api.world.config.get("mechanism_support_truth", "unknown"),
            practical_value=api.world.config.get("practical_value_truth", ""),
            remaining_uncertainty="oracle baseline",
            evidence=[EvidenceItem(claim="Oracle cited controlled experiment outputs.", experiment_ids=evidence_ids)],
            falsifier={"description": "Oracle has access to private intervention outcomes."},
        )
        predictions = [
            InterventionPrediction(
                intervention_id=intervention.intervention_id,
                predicted_delta_mean=float(intervention.expected_delta or 0.0),
                lower_90=float(intervention.expected_delta or 0.0) - 0.001,
                upper_90=float(intervention.expected_delta or 0.0) + 0.001,
                qualitative_result=intervention.qualitative_result or "unknown",
                confidence=0.95,
            )
            for intervention in api.world.interventions
        ]
        api.submit_predictions(predictions)
        api.submit_report(report)
        return report, predictions
