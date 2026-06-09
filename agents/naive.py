"""Naive score-seeking baseline."""

from __future__ import annotations

from agents.base import MechBenchAgent
from mechbench.interface.agent_api import AgentAPI
from mechbench.interface.schemas import WORLD_LABELS, EvidenceItem, FinalReport, InterventionPrediction


class NaiveScoreSeekingAgent(MechBenchAgent):
    def investigate(self, api: AgentAPI) -> tuple[FinalReport, list[InterventionPrediction]]:
        actions = api.list_actions()
        chosen = next((action for action in actions if action["name"] == "proposed"), actions[0])
        result = api.run_experiment(
            {
                "variant": chosen["name"],
                "eval_splits": ["validation_id"],
                "hypothesis_tested": "The method works if the headline score improves.",
            }
        )
        probabilities = {label: 0.05 for label in WORLD_LABELS}
        probabilities["true_mechanism"] = 0.80
        interventions = api.get_interventions()
        predictions = [
            InterventionPrediction(
                intervention_id=item["intervention_id"],
                predicted_delta_mean=0.03,
                lower_90=0.01,
                upper_90=0.05,
                qualitative_result="gain_survives",
                confidence=0.75,
            )
            for item in interventions
        ]
        report = FinalReport(
            episode_id=api.episode_id,
            raw_improvement={
                "status": "reproducible_positive_gain",
                "confidence": 0.75,
                "estimated_delta": 0.03,
            },
            fair_comparison={
                "parameter_matched": "not_tested",
                "compute_matched": "not_tested",
                "norm_matched": "not_tested",
                "multi_seed": "not_tested",
            },
            causal_probabilities=probabilities,
            mechanism_support="claimed_mechanism_supported",
            practical_value="headline improvement appears useful",
            remaining_uncertainty="few controls were run",
            evidence=[EvidenceItem(claim="The proposed method produced a positive-looking score.", experiment_ids=[result["experiment_id"]])],
            falsifier={"description": "Run matched controls."},
        )
        api.submit_predictions(predictions)
        api.submit_report(report)
        return report, predictions
