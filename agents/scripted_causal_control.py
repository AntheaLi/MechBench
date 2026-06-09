"""Scripted causal-control baseline."""

from __future__ import annotations

from agents.base import MechBenchAgent
from mechbench.interface.agent_api import AgentAPI
from mechbench.interface.schemas import WORLD_LABELS, EvidenceItem, FinalReport, InterventionPrediction


class ScriptedCausalControlAgent(MechBenchAgent):
    """A small heuristic agent that runs high-information causal controls."""

    CONTROL_ORDER = [
        "multi_seed_replication",
        "baseline_compute_matched",
        "proposed_compute_limited",
        "generic_learned_branch",
        "geometry_destroyed_branch",
        "norm_matched_proposed",
        "parameter_matched_wider",
        "scale_matched_generic",
    ]

    def investigate(self, api: AgentAPI) -> tuple[FinalReport, list[InterventionPrediction]]:
        baseline_accuracy = self._headline_metric(api.headline, "baseline_metrics")
        proposed_accuracy = self._headline_metric(api.headline, "proposed_metrics")
        raw_delta = proposed_accuracy - baseline_accuracy

        available = {action["name"]: action for action in api.list_actions()}
        evidence_ids: list[str] = []
        deltas: dict[str, float] = {}
        for variant in self.CONTROL_ORDER:
            if variant not in available:
                continue
            try:
                result = api.run_experiment(
                    {
                        "variant": variant,
                        "eval_splits": ["validation_id"],
                        "collect": ["activation_norms"] if "norm" in variant or "geometry" in variant else [],
                        "params": {"n_seeds": 7} if variant == "multi_seed_replication" else {},
                        "hypothesis_tested": self._hypothesis_for_variant(variant),
                    }
                )
            except RuntimeError:
                break
            evidence_ids.append(result["experiment_id"])
            deltas[variant] = self._result_delta(result, baseline_accuracy)
            if variant == "multi_seed_replication" and deltas[variant] < 0.012 and raw_delta > 0.015:
                break

        label = self._diagnose(raw_delta, deltas)
        estimated_delta = deltas.get("multi_seed_replication", raw_delta) if label == "seed_hacking" else raw_delta
        probabilities = self._probabilities(label)
        report = FinalReport(
            episode_id=api.episode_id,
            raw_improvement={
                "status": "not_reliably_positive" if label == "seed_hacking" else "reproducible_positive_gain",
                "confidence": 0.78,
                "estimated_delta": round(estimated_delta, 4),
            },
            fair_comparison={
                "parameter_matched": self._comparison_status(
                    raw_delta,
                    deltas.get("parameter_matched_wider", deltas.get("generic_learned_branch")),
                ),
                "compute_matched": self._compute_comparison_status(raw_delta, deltas),
                "norm_matched": self._comparison_status(raw_delta, deltas.get("norm_matched_proposed")),
                "multi_seed": self._comparison_status(raw_delta, deltas.get("multi_seed_replication")),
            },
            causal_probabilities=probabilities,
            mechanism_support=(
                "claimed_mechanism_supported"
                if label == "true_mechanism"
                else "claimed_mechanism_not_supported"
            ),
            practical_value=self._practical_value(label),
            remaining_uncertainty="scripted controls do not exhaust seed or compute alternatives",
            evidence=[
                EvidenceItem(
                    claim=self._evidence_claim(label),
                    experiment_ids=evidence_ids,
                )
            ],
            falsifier={
                "description": "Destroying routing geometry should remove the gain while norm matching should preserve it if the claimed mechanism is causal."
            },
        )
        predictions = self._predict_interventions(api.get_interventions(), label, raw_delta, deltas)
        api.submit_predictions(predictions)
        api.submit_report(report)
        return report, predictions

    def _headline_metric(self, headline: dict, key: str) -> float:
        metrics = headline.get(key, {})
        if "validation_id.accuracy" in metrics:
            return float(metrics["validation_id.accuracy"])
        return float(next(iter(metrics.values())))

    def _result_delta(self, result: dict, baseline_accuracy: float) -> float:
        metrics = result.get("metrics", {})
        if "validation_id.delta" in metrics:
            return float(metrics["validation_id.delta"])
        if "validation_id.accuracy" in metrics:
            return float(metrics["validation_id.accuracy"]) - baseline_accuracy
        return 0.0

    def _diagnose(self, raw_delta: float, deltas: dict[str, float]) -> str:
        generic = deltas.get("generic_learned_branch")
        wider = deltas.get("parameter_matched_wider")
        destroyed = deltas.get("geometry_destroyed_branch")
        norm_matched = deltas.get("norm_matched_proposed")
        scale_generic = deltas.get("scale_matched_generic")
        multi_seed = deltas.get("multi_seed_replication")
        baseline_compute = deltas.get("baseline_compute_matched")
        proposed_limited = deltas.get("proposed_compute_limited")

        matched_controls = [value for value in [generic, wider] if value is not None]
        matched_best = max(matched_controls) if matched_controls else None

        if raw_delta < 0.015:
            return "seed_hacking"
        if multi_seed is not None and multi_seed < 0.012 and multi_seed < raw_delta - 0.015:
            return "seed_hacking"
        if (
            baseline_compute is not None
            and proposed_limited is not None
            and baseline_compute >= raw_delta - 0.007
            and proposed_limited < raw_delta - 0.015
        ):
            return "compute_laundering_lite"
        if norm_matched is not None and norm_matched < raw_delta - 0.015:
            if scale_generic is None or scale_generic >= raw_delta - 0.012:
                return "norm_laundering"
        if matched_best is not None and matched_best >= raw_delta - 0.007:
            if destroyed is None or destroyed >= raw_delta - 0.010:
                return "parameter_laundering"
        if (
            destroyed is not None
            and destroyed < raw_delta - 0.015
            and norm_matched is not None
            and norm_matched >= raw_delta - 0.006
        ):
            return "true_mechanism"
        # Fallback true_mechanism: geometry destruction removes gain AND generic
        # doesn't recover it, but norm_matched wasn't run (budget exhausted).
        if (
            destroyed is not None
            and destroyed < raw_delta - 0.015
            and generic is not None
            and generic < raw_delta - 0.015
            and baseline_compute is None
        ):
            return "true_mechanism"
        # Fallback norm_laundering: geometry destruction *preserves* gain but
        # generic capacity doesn't recover it, and no compute signals.
        # This pattern (high destroyed, low generic) is unique to norm_laundering.
        if (
            destroyed is not None
            and destroyed >= raw_delta - 0.008
            and generic is not None
            and generic < raw_delta - 0.015
            and baseline_compute is None
        ):
            return "norm_laundering"
        return "compute_laundering_lite"

    def _probabilities(self, label: str) -> dict[str, float]:
        probabilities = {world_label: 0.04 for world_label in WORLD_LABELS}
        probabilities[label] = 0.84
        return probabilities

    def _comparison_status(self, raw_delta: float, control_delta: float | None) -> str:
        if control_delta is None:
            return "not_tested"
        if control_delta >= raw_delta - 0.007:
            return "gain_survives"
        return "gain_does_not_survive"

    def _compute_comparison_status(self, raw_delta: float, deltas: dict[str, float]) -> str:
        baseline_compute = deltas.get("baseline_compute_matched")
        proposed_limited = deltas.get("proposed_compute_limited")
        if baseline_compute is None and proposed_limited is None:
            return "not_tested"
        if (
            baseline_compute is not None
            and baseline_compute >= raw_delta - 0.007
            and proposed_limited is not None
            and proposed_limited < raw_delta - 0.015
        ):
            return "gain_does_not_survive"
        if proposed_limited is not None and proposed_limited >= raw_delta - 0.007:
            return "gain_survives"
        return "inconclusive"

    def _practical_value(self, label: str) -> str:
        if label == "true_mechanism":
            return "the modification appears to provide mechanism-specific value"
        if label == "seed_hacking":
            return "the public gain is not reliable across seeds"
        if label == "parameter_laundering":
            return "the modification may still provide useful generic capacity"
        if label == "norm_laundering":
            return "the apparent value comes from output scale rather than claimed structure"
        if label == "compute_laundering_lite":
            return "the apparent value comes from additional estimated training compute"
        return "practical value remains uncertain after controls"

    def _predict_interventions(
        self,
        interventions: list[dict],
        label: str,
        raw_delta: float,
        deltas: dict[str, float],
    ) -> list[InterventionPrediction]:
        return [
            self._predict_one(intervention, label, raw_delta, deltas)
            for intervention in interventions
        ]

    def _predict_one(
        self,
        intervention: dict,
        label: str,
        raw_delta: float,
        deltas: dict[str, float],
    ) -> InterventionPrediction:
        description = intervention.get("description", "").lower()
        qualitative = self._coarse_intervention_direction(label, description)

        estimated = self._estimate_from_experiments(description, label, raw_delta, deltas)
        if estimated is not None:
            # Shrink toward a weak prior — the agent is extrapolating from
            # experiments it ran to unseen interventions, which shouldn't be exact.
            prior = raw_delta * 0.5
            predicted_mean = max(0.0, estimated * 0.75 + prior * 0.25)
            half_width = 0.009
            confidence = 0.68
        elif qualitative == "gain_survives":
            predicted_mean = max(0.0, raw_delta - 0.003)
            half_width = 0.014
            confidence = 0.55
        elif qualitative == "gain_disappears":
            predicted_mean = 0.005
            half_width = 0.009
            confidence = 0.55
        else:
            predicted_mean = max(0.0, raw_delta * 0.50)
            half_width = 0.015
            confidence = 0.50

        return InterventionPrediction(
            intervention_id=intervention["intervention_id"],
            predicted_delta_mean=round(predicted_mean, 4),
            lower_90=round(predicted_mean - half_width, 4),
            upper_90=round(predicted_mean + half_width, 4),
            qualitative_result=qualitative,
            confidence=confidence,
        )

    def _estimate_from_experiments(
        self,
        description: str,
        label: str,
        raw_delta: float,
        deltas: dict[str, float],
    ) -> float | None:
        """Use actual experiment deltas to estimate a held-out intervention outcome."""

        if label == "seed_hacking":
            multi_seed = deltas.get("multi_seed_replication")
            if multi_seed is not None:
                return multi_seed

        if label == "true_mechanism":
            if "destroy" in description or "routing structure" in description:
                destroyed = deltas.get("geometry_destroyed_branch")
                if destroyed is not None:
                    return destroyed
            if "partial" in description or "ablat" in description or "half capacity" in description:
                destroyed = deltas.get("geometry_destroyed_branch")
                if destroyed is not None:
                    # Midpoint between full destruction and full gain
                    return (destroyed + raw_delta) * 0.5
                return raw_delta * 0.45
            return raw_delta - 0.003

        if label == "parameter_laundering":
            if "generic learned" in description or "equal parameter" in description:
                return deltas.get("generic_learned_branch")
            if "destroy" in description or "geometry" in description:
                return deltas.get("geometry_destroyed_branch")
            if "fixed random" in description or "non-learned" in description:
                return 0.003
            if "seed" in description:
                return raw_delta - 0.002

        if label == "norm_laundering":
            if "reduce" in description and "norm" in description:
                return deltas.get("norm_matched_proposed")
            if "increase" in description and "generic" in description:
                return deltas.get("scale_matched_generic")
            if "destroy" in description and "geometry" in description:
                return deltas.get("geometry_destroyed_branch")

        if label == "compute_laundering_lite":
            if "baseline" in description and "compute" in description:
                return deltas.get("baseline_compute_matched")
            if "restrict" in description or "proposed" in description and "baseline" in description:
                return deltas.get("proposed_compute_limited")
            if "learning curve" in description or "matched compute" in description:
                limited = deltas.get("proposed_compute_limited")
                if limited is not None:
                    return limited
                return 0.004

        return None

    def _coarse_intervention_direction(self, label: str, description: str) -> str:
        if label == "seed_hacking":
            return "gain_disappears"
        if label == "true_mechanism":
            if "destroy" in description or "geometry" in description:
                return "gain_disappears"
            if "partial" in description or "ablat" in description or "half capacity" in description:
                return "gain_shrinks"
            return "gain_survives"
        if label == "parameter_laundering":
            if "fixed" in description or "random" in description:
                return "gain_disappears"
            return "gain_survives"
        if label == "norm_laundering":
            if "reduce" in description or "reduced" in description:
                return "gain_disappears"
            return "gain_survives"
        if label == "compute_laundering_lite":
            if "baseline" in description and "proposed" in description:
                return "gain_survives"
            if "restrict" in description or "learning curve" in description or "matched compute" in description:
                return "gain_disappears"
        return "uncertain"

    def _hypothesis_for_variant(self, variant: str) -> str:
        hypotheses = {
            "multi_seed_replication": "The headline gain may be a favorable seed outlier.",
            "baseline_compute_matched": "The gain may come from additional estimated training compute.",
            "proposed_compute_limited": "The proposed model may fail when restricted to the baseline compute budget.",
            "generic_learned_branch": "The gain may come from generic learned capacity.",
            "parameter_matched_wider": "The gain may come from parameter count rather than branch structure.",
            "geometry_destroyed_branch": "The claimed routing geometry may be necessary for the gain.",
            "norm_matched_proposed": "The gain may come from branch output scale rather than structure.",
            "scale_matched_generic": "A generic branch with matched scale may recover the gain.",
        }
        return hypotheses.get(variant, "Run a causal-control experiment.")

    def _evidence_claim(self, label: str) -> str:
        if label == "seed_hacking":
            return "A preregistered multi-seed control collapses the public demonstration gain."
        if label == "compute_laundering_lite":
            return "Estimated-compute matched controls remove the apparent advantage."
        if label == "norm_laundering":
            return "Norm-matched and scale-matched controls show that branch output scale, not structure, drives the gain."
        return "Matched public controls distinguish mechanism-specific gain from generic confounds."
