"""LLM agent scaffold with pluggable JSON backends.

The scaffold owns the benchmark mechanics: prompt construction, budget-aware
experiment execution, JSON parsing, schema validation, fallback repair, and
trace capture. A real provider can be added by implementing ``LLMBackend``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from agents.base import MechBenchAgent
from mechbench.interface.agent_api import AgentAPI
from mechbench.interface.schemas import (
    WORLD_LABELS,
    EvidenceItem,
    FinalReport,
    InterventionPrediction,
)


class LLMBackend(Protocol):
    """Minimal backend protocol for model providers or deterministic mocks."""

    def complete(self, messages: list[dict[str, str]], *, purpose: str) -> str:
        """Return a model response string for the given prompt messages."""


@dataclass(frozen=True)
class ExperimentObservation:
    experiment_id: str
    variant: str
    delta: float
    metrics: dict[str, float]
    model_stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "variant": self.variant,
            "delta": self.delta,
            "metrics": dict(self.metrics),
            "model_stats": dict(self.model_stats),
        }


class HeuristicMockLLMBackend:
    """Deterministic backend used for tests and offline scaffold development."""

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

    def complete(self, messages: list[dict[str, str]], *, purpose: str) -> str:
        context = _extract_json_object(messages[-1]["content"])
        if purpose == "plan":
            return json.dumps(self._plan(context), sort_keys=True)
        if purpose == "final":
            return json.dumps(self._final(context), sort_keys=True)
        raise ValueError(f"unknown scaffold LLM purpose: {purpose}")

    def _plan(self, context: dict[str, Any]) -> dict[str, Any]:
        available = {action["name"]: action for action in context.get("actions", [])}
        experiments = []
        for variant in self.CONTROL_ORDER:
            if variant not in available:
                continue
            experiment = {
                "variant": variant,
                "eval_splits": ["validation_id"],
                "hypothesis_tested": _hypothesis_for_variant(variant),
            }
            if variant == "multi_seed_replication":
                experiment["params"] = {"n_seeds": 7}
            if "norm" in variant or "geometry" in variant:
                experiment["collect"] = ["activation_norms"]
            experiments.append(experiment)
        return {
            "rationale": "Run high-information causal controls before predicting private interventions.",
            "experiments": experiments,
        }

    def _final(self, context: dict[str, Any]) -> dict[str, Any]:
        headline = context.get("headline", {})
        raw_delta = _headline_delta(headline)
        observations = context.get("observations", [])
        deltas = {
            str(item.get("variant")): float(item.get("delta", 0.0))
            for item in observations
        }
        evidence_ids = [str(item["experiment_id"]) for item in observations if item.get("experiment_id")]
        label = _diagnose(raw_delta, deltas)
        estimated_delta = deltas.get("multi_seed_replication", raw_delta) if label == "seed_hacking" else raw_delta
        report = {
            "episode_id": context.get("episode_id", ""),
            "raw_improvement": {
                "status": "not_reliably_positive" if label == "seed_hacking" else "reproducible_positive_gain",
                "confidence": 0.74,
                "estimated_delta": round(estimated_delta, 4),
            },
            "fair_comparison": {
                "parameter_matched": _comparison_status(
                    raw_delta,
                    deltas.get("parameter_matched_wider", deltas.get("generic_learned_branch")),
                ),
                "compute_matched": _compute_comparison_status(raw_delta, deltas),
                "norm_matched": _comparison_status(raw_delta, deltas.get("norm_matched_proposed")),
                "multi_seed": _comparison_status(raw_delta, deltas.get("multi_seed_replication")),
            },
            "causal_probabilities": _probabilities(label, confidence=0.78),
            "mechanism_support": (
                "claimed_mechanism_supported"
                if label == "true_mechanism"
                else "claimed_mechanism_not_supported"
            ),
            "practical_value": _practical_value(label),
            "remaining_uncertainty": "LLM scaffold used a bounded set of public controls.",
            "evidence": [
                {
                    "claim": _evidence_claim(label),
                    "experiment_ids": evidence_ids,
                }
            ] if evidence_ids else [],
            "falsifier": {
                "description": "Run a targeted hidden intervention that breaks the favored causal explanation while preserving matched controls."
            },
        }
        predictions = [
            _predict_intervention(intervention, label, raw_delta, deltas).to_dict()
            for intervention in context.get("interventions", [])
        ]
        return {"report": report, "predictions": predictions}


class LLMScaffoldAgent(MechBenchAgent):
    """Budget-aware LLM scaffold for MechanismBench episodes."""

    SYSTEM_PROMPT = (
        "You are a MechanismBench causal-auditing agent. You may only rely on "
        "public context and experiment observations supplied in the prompt. "
        "Return strict JSON matching the requested shape."
    )

    def __init__(
        self,
        backend: LLMBackend | None = None,
        max_experiments: int = 8,
        inspect_paths: tuple[str, ...] = ("README.md", "results/initial_result.json"),
    ):
        self.backend = backend or HeuristicMockLLMBackend()
        self.max_experiments = max_experiments
        self.inspect_paths = inspect_paths
        self.last_trace: list[dict[str, Any]] = []

    def investigate(self, api: AgentAPI) -> tuple[FinalReport, list[InterventionPrediction]]:
        self.last_trace = []
        public_context = self._public_context(api)
        plan_payload = self._ask_json(
            purpose="plan",
            instruction=(
                "Choose controlled experiments to run. Return JSON with keys "
                "`rationale` and `experiments`, where experiments is a list of "
                "run_experiment configs."
            ),
            context=public_context,
        )
        observations = self._execute_plan(api, plan_payload)
        final_context = {
            **public_context,
            "observations": [item.to_dict() for item in observations],
            "budget_remaining": api.budget_remaining,
            "evidence_summary": api.evidence_summary(),
        }
        final_payload = self._ask_json(
            purpose="final",
            instruction=(
                "Return JSON with keys `report` and `predictions`. The report "
                "must match the MechanismBench FinalReport schema. Each "
                "prediction must include intervention_id, predicted_delta, "
                "qualitative_result, and confidence."
            ),
            context=final_context,
        )
        report, predictions = self._parse_final_payload(api, final_payload, observations)
        api.submit_predictions(predictions)
        api.submit_report(report)
        return report, predictions

    def _public_context(self, api: AgentAPI) -> dict[str, Any]:
        inspected = {}
        for path in self.inspect_paths:
            try:
                inspected[path] = api.inspect(path)[:4000]
            except Exception as exc:  # pragma: no cover - defensive against custom workspaces
                inspected[path] = f"INSPECT_FAILED: {type(exc).__name__}: {exc}"
        context = {
            "episode_id": api.episode_id,
            "headline": api.headline,
            "description": api.description,
            "budget_remaining": api.budget_remaining,
            "actions": api.list_actions(),
            "interventions": api.get_interventions(),
            "files": api.list_files(),
            "inspected": inspected,
        }
        self.last_trace.append({"type": "context", "payload": context})
        return context

    def _ask_json(self, purpose: str, instruction: str, context: dict[str, Any]) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{instruction}\n\n"
                    "CONTEXT_JSON:\n"
                    f"{json.dumps(context, sort_keys=True)}"
                ),
            },
        ]
        self.last_trace.append({"type": "llm_request", "purpose": purpose, "messages": messages})
        response = self.backend.complete(messages, purpose=purpose)
        self.last_trace.append({"type": "llm_response", "purpose": purpose, "content": response})
        try:
            parsed = _extract_json_object(response)
        except ValueError as exc:
            self.last_trace.append({"type": "llm_parse_error", "purpose": purpose, "error": str(exc)})
            return {}
        return parsed

    def _execute_plan(self, api: AgentAPI, plan_payload: dict[str, Any]) -> list[ExperimentObservation]:
        actions = {action["name"]: action for action in api.list_actions()}
        experiments = plan_payload.get("experiments", [])
        if not isinstance(experiments, list):
            experiments = []
        observations: list[ExperimentObservation] = []
        seen_variants: set[str] = set()
        for item in experiments:
            if len(observations) >= self.max_experiments:
                break
            if not isinstance(item, dict):
                continue
            variant = str(item.get("variant", item.get("action", "")))
            if not variant or variant not in actions or variant in seen_variants:
                continue
            rough_cost = float(actions[variant].get("default_cost", 1.0))
            if rough_cost > api.budget_remaining:
                self.last_trace.append({
                    "type": "tool_skip",
                    "variant": variant,
                    "reason": "rough_cost_exceeds_remaining_budget",
                    "budget_remaining": api.budget_remaining,
                })
                continue
            request = {
                "variant": variant,
                "eval_splits": [str(split) for split in item.get("eval_splits", ["validation_id"])],
                "collect": [str(value) for value in item.get("collect", [])],
                "params": item.get("params", {}) if isinstance(item.get("params", {}), dict) else {},
                "hypothesis_tested": str(item.get("hypothesis_tested", _hypothesis_for_variant(variant))),
            }
            self.last_trace.append({"type": "tool_call", "tool": "run_experiment", "request": request})
            try:
                result = api.run_experiment(request)
            except RuntimeError as exc:
                self.last_trace.append({
                    "type": "tool_error",
                    "tool": "run_experiment",
                    "variant": variant,
                    "error": str(exc),
                })
                break
            observation = ExperimentObservation(
                experiment_id=str(result["experiment_id"]),
                variant=variant,
                delta=_result_delta(result, _headline_baseline(api.headline)),
                metrics={str(k): float(v) for k, v in result.get("metrics", {}).items()},
                model_stats=dict(result.get("model_stats", {})),
            )
            observations.append(observation)
            seen_variants.add(variant)
            self.last_trace.append({"type": "tool_result", "observation": observation.to_dict()})
        return observations

    def _parse_final_payload(
        self,
        api: AgentAPI,
        payload: dict[str, Any],
        observations: list[ExperimentObservation],
    ) -> tuple[FinalReport, list[InterventionPrediction]]:
        try:
            report = FinalReport.from_dict(payload.get("report", {}))
        except Exception as exc:
            self.last_trace.append({"type": "schema_error", "schema": "FinalReport", "error": str(exc)})
            report = self._fallback_report(api, observations)
        if report.episode_id != api.episode_id:
            report = FinalReport.from_dict({**report.to_dict(), "episode_id": api.episode_id})
        if not report.evidence and observations:
            report = FinalReport.from_dict({
                **report.to_dict(),
                "evidence": [
                    EvidenceItem(
                        claim="The scaffold cited all controlled experiments it ran.",
                        experiment_ids=[item.experiment_id for item in observations],
                    ).to_dict()
                ],
            })

        predictions = []
        raw_predictions = payload.get("predictions", [])
        if not isinstance(raw_predictions, list):
            raw_predictions = []
        for item in raw_predictions:
            try:
                predictions.append(InterventionPrediction.from_dict(item))
            except Exception as exc:
                self.last_trace.append({"type": "schema_error", "schema": "InterventionPrediction", "error": str(exc)})
        predictions = self._complete_predictions(api, report, predictions, observations)
        return report, predictions

    def _fallback_report(self, api: AgentAPI, observations: list[ExperimentObservation]) -> FinalReport:
        raw_delta = _headline_delta(api.headline)
        deltas = {item.variant: item.delta for item in observations}
        label = _diagnose(raw_delta, deltas)
        evidence_ids = [item.experiment_id for item in observations]
        return FinalReport(
            episode_id=api.episode_id,
            raw_improvement={
                "status": "not_reliably_positive" if label == "seed_hacking" else "reproducible_positive_gain",
                "confidence": 0.55,
                "estimated_delta": round(deltas.get("multi_seed_replication", raw_delta), 4)
                if label == "seed_hacking"
                else round(raw_delta, 4),
            },
            fair_comparison={
                "parameter_matched": _comparison_status(
                    raw_delta,
                    deltas.get("parameter_matched_wider", deltas.get("generic_learned_branch")),
                ),
                "compute_matched": _compute_comparison_status(raw_delta, deltas),
                "norm_matched": _comparison_status(raw_delta, deltas.get("norm_matched_proposed")),
                "multi_seed": _comparison_status(raw_delta, deltas.get("multi_seed_replication")),
            },
            causal_probabilities=_probabilities(label, confidence=0.60),
            mechanism_support="claimed_mechanism_supported" if label == "true_mechanism" else "claimed_mechanism_not_supported",
            practical_value=_practical_value(label),
            remaining_uncertainty="Fallback report generated after invalid LLM output.",
            evidence=[EvidenceItem(claim="Fallback cited controlled experiment outputs.", experiment_ids=evidence_ids)] if evidence_ids else [],
            falsifier={"description": "Run a private held-out intervention targeted at the favored explanation."},
        )

    def _complete_predictions(
        self,
        api: AgentAPI,
        report: FinalReport,
        predictions: list[InterventionPrediction],
        observations: list[ExperimentObservation],
    ) -> list[InterventionPrediction]:
        by_id = {prediction.intervention_id: prediction for prediction in predictions}
        raw_delta = _headline_delta(api.headline)
        label = max(report.normalized_probabilities(), key=report.normalized_probabilities().get)
        deltas = {item.variant: item.delta for item in observations}
        completed = []
        for intervention in api.get_interventions():
            intervention_id = intervention["intervention_id"]
            completed.append(
                by_id.get(intervention_id)
                or _predict_intervention(intervention, label, raw_delta, deltas)
            )
        return completed


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("response did not contain a JSON object")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def _headline_baseline(headline: dict[str, Any]) -> float:
    metrics = headline.get("baseline_metrics", {})
    if "validation_id.accuracy" in metrics:
        return float(metrics["validation_id.accuracy"])
    return float(next(iter(metrics.values())))


def _headline_delta(headline: dict[str, Any]) -> float:
    baseline = _headline_baseline(headline)
    proposed_metrics = headline.get("proposed_metrics", {})
    proposed = (
        float(proposed_metrics["validation_id.accuracy"])
        if "validation_id.accuracy" in proposed_metrics
        else float(next(iter(proposed_metrics.values())))
    )
    return proposed - baseline


def _result_delta(result: dict[str, Any], baseline_accuracy: float) -> float:
    metrics = result.get("metrics", {})
    if "validation_id.delta" in metrics:
        return float(metrics["validation_id.delta"])
    if "validation_id.accuracy" in metrics:
        return float(metrics["validation_id.accuracy"]) - baseline_accuracy
    return 0.0


def _diagnose(raw_delta: float, deltas: dict[str, float]) -> str:
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
    if (
        destroyed is not None
        and destroyed < raw_delta - 0.015
        and generic is not None
        and generic < raw_delta - 0.015
        and baseline_compute is None
    ):
        return "true_mechanism"
    if (
        destroyed is not None
        and destroyed >= raw_delta - 0.008
        and generic is not None
        and generic < raw_delta - 0.015
        and baseline_compute is None
    ):
        return "norm_laundering"
    return "compute_laundering_lite"


def _probabilities(label: str, confidence: float) -> dict[str, float]:
    remainder = max(0.0, 1.0 - confidence)
    other = remainder / (len(WORLD_LABELS) - 1)
    probabilities = {world_label: other for world_label in WORLD_LABELS}
    probabilities[label] = confidence
    return probabilities


def _comparison_status(raw_delta: float, control_delta: float | None) -> str:
    if control_delta is None:
        return "not_tested"
    if control_delta >= raw_delta - 0.007:
        return "gain_survives"
    return "gain_does_not_survive"


def _compute_comparison_status(raw_delta: float, deltas: dict[str, float]) -> str:
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


def _predict_intervention(
    intervention: dict[str, Any],
    label: str,
    raw_delta: float,
    deltas: dict[str, float],
) -> InterventionPrediction:
    description = intervention.get("description", "").lower()
    qualitative = _coarse_direction(label, description)
    estimate = _estimate_from_observations(description, label, raw_delta, deltas)
    if estimate is None:
        if qualitative == "gain_survives":
            estimate = raw_delta - 0.004
        elif qualitative == "gain_disappears":
            estimate = 0.004
        elif qualitative == "gain_shrinks":
            estimate = raw_delta * 0.45
        else:
            estimate = raw_delta * 0.5
    half_width = 0.012 if qualitative != "uncertain" else 0.018
    return InterventionPrediction(
        intervention_id=str(intervention["intervention_id"]),
        predicted_delta_mean=round(max(0.0, estimate), 4),
        lower_90=round(max(0.0, estimate) - half_width, 4),
        upper_90=round(max(0.0, estimate) + half_width, 4),
        qualitative_result=qualitative,
        confidence=0.64 if qualitative != "uncertain" else 0.50,
    )


def _estimate_from_observations(
    description: str,
    label: str,
    raw_delta: float,
    deltas: dict[str, float],
) -> float | None:
    if label == "seed_hacking":
        return deltas.get("multi_seed_replication")
    if label == "true_mechanism":
        if "destroy" in description or "geometry" in description:
            return deltas.get("geometry_destroyed_branch")
        if "partial" in description or "ablat" in description or "half capacity" in description:
            destroyed = deltas.get("geometry_destroyed_branch", 0.0)
            return (destroyed + raw_delta) * 0.5
        return raw_delta - 0.003
    if label == "parameter_laundering":
        if "generic learned" in description or "equal-parameter" in description or "equal parameter" in description:
            return deltas.get("generic_learned_branch")
        if "destroy" in description or "geometry" in description:
            return deltas.get("geometry_destroyed_branch")
        if "fixed random" in description or "non-learned" in description:
            return 0.004
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
        if "restrict" in description or ("proposed" in description and "baseline" in description):
            return deltas.get("proposed_compute_limited")
        if "learning curve" in description or "matched compute" in description:
            return deltas.get("proposed_compute_limited")
    return None


def _coarse_direction(label: str, description: str) -> str:
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


def _hypothesis_for_variant(variant: str) -> str:
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


def _practical_value(label: str) -> str:
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


def _evidence_claim(label: str) -> str:
    if label == "seed_hacking":
        return "A multi-seed control collapses the public demonstration gain."
    if label == "compute_laundering_lite":
        return "Estimated-compute matched controls remove the apparent advantage."
    if label == "norm_laundering":
        return "Norm-matched and scale-matched controls show that branch output scale drives the gain."
    return "Matched public controls distinguish mechanism-specific gain from generic confounds."
