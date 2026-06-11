"""Dataclass schemas for the generic MechanismBench harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


WORLD_LABELS = (
    "true_mechanism",
    "seed_hacking",
    "parameter_laundering",
    "norm_laundering",
    "compute_laundering_lite",
)


def _float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc


def _dict(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return dict(value)


def to_plain_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_plain_dict(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain_dict(item) for item in value]
    return value


@dataclass(frozen=True)
class BudgetConfig:
    max_cost: float = 10.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BudgetConfig":
        data = data or {}
        return cls(max_cost=_float(data.get("max_cost", data.get("standard_budget", 10.0)), "max_cost"))

    def to_dict(self) -> dict[str, Any]:
        return to_plain_dict(self)


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    default_cost: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionSpec":
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            params=_dict(data.get("params", {}), "params"),
            default_cost=_float(data.get("default_cost", 1.0), "default_cost"),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_plain_dict(self)


@dataclass(frozen=True)
class ExperimentRequest:
    variant: str
    train_seed: int = 0
    data_seed: int = 0
    train_tokens: int = 0
    eval_splits: list[str] = field(default_factory=lambda: ["validation_id"])
    collect: list[str] = field(default_factory=list)
    hypothesis_tested: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRequest":
        variant = data.get("variant", data.get("action"))
        if not variant:
            raise ValueError("experiment request requires 'variant'")
        return cls(
            variant=str(variant),
            train_seed=int(data.get("train_seed", 0)),
            data_seed=int(data.get("data_seed", 0)),
            train_tokens=int(data.get("train_tokens", 0)),
            eval_splits=[str(item) for item in data.get("eval_splits", ["validation_id"])],
            collect=[str(item) for item in data.get("collect", [])],
            hypothesis_tested=str(data.get("hypothesis_tested", "")),
            params=_dict(data.get("params", {}), "params"),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_plain_dict(self)


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    status: str
    cost: float
    config_hash: str
    model_stats: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentResult":
        return cls(
            experiment_id=str(data.get("experiment_id", "")),
            status=str(data.get("status", "completed")),
            cost=_float(data.get("cost", 1.0), "cost"),
            config_hash=str(data.get("config_hash", "")),
            model_stats=_dict(data.get("model_stats", {}), "model_stats"),
            metrics={str(k): float(v) for k, v in _dict(data.get("metrics", {}), "metrics").items()},
            artifacts={str(k): str(v) for k, v in _dict(data.get("artifacts", {}), "artifacts").items()},
        )

    def with_identity(self, experiment_id: str, config_hash: str, cost: float) -> "ExperimentResult":
        return ExperimentResult(
            experiment_id=experiment_id,
            status=self.status,
            cost=cost,
            config_hash=config_hash,
            model_stats=dict(self.model_stats),
            metrics=dict(self.metrics),
            artifacts=dict(self.artifacts),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_plain_dict(self)


@dataclass(frozen=True)
class Headline:
    baseline_metrics: dict[str, float]
    proposed_metrics: dict[str, float]
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Headline":
        baseline = data.get("baseline_metrics", data.get("baseline_metric", {}))
        proposed = data.get("proposed_metrics", data.get("modified_metric", {}))
        return cls(
            baseline_metrics={str(k): float(v) for k, v in _dict(baseline, "baseline_metrics").items()},
            proposed_metrics={str(k): float(v) for k, v in _dict(proposed, "proposed_metrics").items()},
            description=str(data.get("description", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_plain_dict(self)


@dataclass(frozen=True)
class InterventionSpec:
    intervention_id: str
    description: str
    expected_delta: float | None = None
    qualitative_result: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InterventionSpec":
        intervention_id = data.get("intervention_id", data.get("id"))
        if not intervention_id:
            raise ValueError("intervention requires 'id' or 'intervention_id'")
        expected_delta = data.get("expected_delta")
        if expected_delta is None and isinstance(data.get("expected_result"), dict):
            expected_delta = data["expected_result"].get("delta")
        return cls(
            intervention_id=str(intervention_id),
            description=str(data.get("description", "")),
            expected_delta=None if expected_delta is None else float(expected_delta),
            qualitative_result=data.get("qualitative_result"),
            metadata=_dict(data.get("metadata", {}), "metadata"),
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "description": self.description,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return to_plain_dict(self)


@dataclass(frozen=True)
class EvidenceItem:
    claim: str
    experiment_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceItem":
        return cls(
            claim=str(data.get("claim", "")),
            experiment_ids=[str(item) for item in data.get("experiment_ids", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return to_plain_dict(self)


@dataclass(frozen=True)
class FinalReport:
    episode_id: str
    raw_improvement: dict[str, Any]
    fair_comparison: dict[str, Any]
    causal_probabilities: dict[str, float]
    mechanism_support: str
    practical_value: str = ""
    remaining_uncertainty: str = ""
    evidence: list[EvidenceItem] = field(default_factory=list)
    falsifier: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinalReport":
        probabilities = {str(k): float(v) for k, v in _dict(data.get("causal_probabilities", {}), "causal_probabilities").items()}
        return cls(
            episode_id=str(data.get("episode_id", "")),
            raw_improvement=_dict(data.get("raw_improvement", {}), "raw_improvement"),
            fair_comparison=_dict(data.get("fair_comparison", {}), "fair_comparison"),
            causal_probabilities=probabilities,
            mechanism_support=str(data.get("mechanism_support", "")),
            practical_value=str(data.get("practical_value", "")),
            remaining_uncertainty=str(data.get("remaining_uncertainty", "")),
            evidence=[EvidenceItem.from_dict(item) for item in data.get("evidence", [])],
            falsifier=_dict(data.get("falsifier", {}), "falsifier"),
        )

    def normalized_probabilities(self, labels: tuple[str, ...] = WORLD_LABELS) -> dict[str, float]:
        values = {label: max(0.0, float(self.causal_probabilities.get(label, 0.0))) for label in labels}
        total = sum(values.values())
        if total <= 0:
            return {label: 1.0 / len(labels) for label in labels}
        return {label: value / total for label, value in values.items()}

    def to_dict(self) -> dict[str, Any]:
        return to_plain_dict(self)


@dataclass(frozen=True)
class InterventionPrediction:
    intervention_id: str
    predicted_delta_mean: float
    lower_90: float
    upper_90: float
    qualitative_result: str
    confidence: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InterventionPrediction":
        predicted_delta = data.get("predicted_delta", {})
        if not isinstance(predicted_delta, dict):
            raise ValueError("predicted_delta must be an object")
        return cls(
            intervention_id=str(data.get("intervention_id", data.get("id", ""))),
            predicted_delta_mean=_float(predicted_delta.get("mean", data.get("predicted_delta_mean", 0.0)), "predicted_delta.mean"),
            lower_90=_float(predicted_delta.get("lower_90", data.get("lower_90", 0.0)), "lower_90"),
            upper_90=_float(predicted_delta.get("upper_90", data.get("upper_90", 0.0)), "upper_90"),
            qualitative_result=str(data.get("qualitative_result", "")),
            confidence=max(0.0, min(1.0, _float(data.get("confidence", 0.5), "confidence"))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "predicted_delta": {
                "mean": self.predicted_delta_mean,
                "lower_90": self.lower_90,
                "upper_90": self.upper_90,
            },
            "qualitative_result": self.qualitative_result,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class EpisodeScore:
    total: float
    components: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return to_plain_dict(self)


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    world_id: str
    report: FinalReport
    predictions: list[InterventionPrediction]
    score: EpisodeScore
    ledger: list[dict[str, Any]]
    workspace: str

    def to_dict(self) -> dict[str, Any]:
        return to_plain_dict(self)

