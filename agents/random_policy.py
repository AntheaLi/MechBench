"""Random experiment baseline."""

from __future__ import annotations

import random

from agents.base import MechBenchAgent
from mechbench.interface.agent_api import AgentAPI
from mechbench.interface.schemas import WORLD_LABELS, EvidenceItem, FinalReport, InterventionPrediction


class RandomExperimentAgent(MechBenchAgent):
    def __init__(self, seed: int = 0, max_experiments: int = 3):
        self.random = random.Random(seed)
        self.max_experiments = max_experiments

    def investigate(self, api: AgentAPI) -> tuple[FinalReport, list[InterventionPrediction]]:
        actions = api.list_actions()
        evidence_ids = []
        for _ in range(min(self.max_experiments, len(actions))):
            action = self.random.choice(actions)
            try:
                result = api.run_experiment(
                    {
                        "variant": action["name"],
                        "eval_splits": ["validation_id"],
                        "hypothesis_tested": "Random baseline.",
                    }
                )
            except RuntimeError:
                break
            evidence_ids.append(result["experiment_id"])

        uniform = 1.0 / len(WORLD_LABELS)
        probabilities = {label: uniform for label in WORLD_LABELS}
        predictions = [
            InterventionPrediction(
                intervention_id=item["intervention_id"],
                predicted_delta_mean=0.0,
                lower_90=-0.05,
                upper_90=0.05,
                qualitative_result="uncertain",
                confidence=0.50,
            )
            for item in api.get_interventions()
        ]
        report = FinalReport(
            episode_id=api.episode_id,
            raw_improvement={
                "status": "unknown",
                "confidence": 0.50,
                "estimated_delta": 0.0,
            },
            fair_comparison={},
            causal_probabilities=probabilities,
            mechanism_support="unknown",
            practical_value="unknown",
            remaining_uncertainty="random baseline",
            evidence=[EvidenceItem(claim="Randomly sampled evidence.", experiment_ids=evidence_ids)] if evidence_ids else [],
            falsifier={"description": "Use a causal-control policy."},
        )
        api.submit_predictions(predictions)
        api.submit_report(report)
        return report, predictions
