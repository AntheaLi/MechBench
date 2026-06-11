"""Calibration loop for simulator episodes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, median
from typing import Any

from agents.naive import NaiveScoreSeekingAgent
from agents.oracle import OracleAgent
from agents.random_policy import RandomExperimentAgent
from agents.scripted_causal_control import ScriptedCausalControlAgent
from mechbench.interface.schemas import EpisodeResult
from mechbench.registry import Registry


AGENTS = {
    "oracle": OracleAgent,
    "scripted": ScriptedCausalControlAgent,
    "naive": NaiveScoreSeekingAgent,
    "random": RandomExperimentAgent,
}

SIMULATOR_LABELS = (
    "true_mechanism",
    "seed_hacking",
    "parameter_laundering",
    "norm_laundering",
    "compute_laundering_lite",
)


@dataclass(frozen=True)
class CalibrationResult:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.payload


def simulator_world_ids(registry: Registry, selector: str = "compiled") -> list[str]:
    ids = [
        world_id
        for world_id, world in registry.worlds.items()
        if world.family == "simulator" and world.causal_label in SIMULATOR_LABELS
    ]
    if selector == "compiled":
        ids = [world_id for world_id in ids if registry.worlds[world_id].name.startswith("compiled_")]
    elif selector == "manual":
        ids = [world_id for world_id in ids if not registry.worlds[world_id].name.startswith("compiled_")]
    elif selector != "all":
        requested = {item.strip() for item in selector.split(",") if item.strip()}
        ids = [world_id for world_id in ids if world_id in requested]
    return sorted(ids)


def calibrate_simulator(
    families_dir: str = "families",
    selector: str = "compiled",
    agent_names: list[str] | None = None,
) -> CalibrationResult:
    registry = Registry(families_dir)
    world_ids = simulator_world_ids(registry, selector)
    if not world_ids:
        raise ValueError(f"no simulator worlds found for selector '{selector}'")
    agent_names = agent_names or ["oracle", "scripted", "naive", "random"]
    agent_results: dict[str, list[EpisodeResult]] = {
        agent_name: _evaluate_agent(registry, world_ids, agent_name)
        for agent_name in agent_names
    }
    agent_totals: dict[str, list[float]] = {
        agent_name: [r.score.total for r in results]
        for agent_name, results in agent_results.items()
    }
    world_counts = _world_counts(registry, world_ids)
    public_accuracy = public_only_world_classifier_accuracy(registry, world_ids)
    payload: dict[str, Any] = {
        "world_count": len(world_ids),
        "world_counts": world_counts,
        "public_only_world_classification_accuracy": public_accuracy,
        "agents": {
            agent_name: _score_summary(scores)
            for agent_name, scores in agent_totals.items()
        },
        "per_world_type": _per_world_type_breakdown(registry, agent_results),
        "per_component": _per_component_breakdown(agent_results),
        "per_episode": _per_episode_dump(registry, agent_results),
        "gates": _phase1_gates(len(world_ids), public_accuracy, agent_totals),
    }
    return CalibrationResult(payload)


def _evaluate_agent(registry: Registry, world_ids: list[str], agent_name: str) -> list[EpisodeResult]:
    if agent_name not in AGENTS:
        raise ValueError(f"unknown calibration agent: {agent_name}")
    results: list[EpisodeResult] = []
    for index, world_id in enumerate(world_ids):
        if agent_name == "random":
            agent = RandomExperimentAgent(seed=index)
        else:
            agent = AGENTS[agent_name]()
        result = registry.get_episode(world_id).run(agent)
        results.append(result)
    return results


def _world_counts(registry: Registry, world_ids: list[str]) -> dict[str, int]:
    counts = {label: 0 for label in SIMULATOR_LABELS}
    for world_id in world_ids:
        counts[registry.worlds[world_id].causal_label] += 1
    return counts


def _score_summary(scores: list[float]) -> dict[str, float]:
    return {
        "mean": round(mean(scores), 3),
        "median": round(median(scores), 3),
        "min": round(min(scores), 3),
        "max": round(max(scores), 3),
    }


def _per_world_type_breakdown(
    registry: Registry,
    agent_results: dict[str, list[EpisodeResult]],
) -> dict[str, dict[str, dict[str, float]]]:
    """For each agent × world_type, compute score summary."""
    breakdown: dict[str, dict[str, dict[str, float]]] = {}
    for agent_name, results in agent_results.items():
        by_label: dict[str, list[float]] = {label: [] for label in SIMULATOR_LABELS}
        for result in results:
            label = registry.worlds[result.world_id].causal_label
            if label in by_label:
                by_label[label].append(result.score.total)
        agent_breakdown: dict[str, dict[str, float]] = {}
        for label in SIMULATOR_LABELS:
            scores = by_label[label]
            if scores:
                agent_breakdown[label] = _score_summary(scores)
        breakdown[agent_name] = agent_breakdown
    return breakdown


def _per_component_breakdown(
    agent_results: dict[str, list[EpisodeResult]],
) -> dict[str, dict[str, float]]:
    """For each agent × scoring component, compute mean score."""
    breakdown: dict[str, dict[str, float]] = {}
    for agent_name, results in agent_results.items():
        if not results:
            continue
        component_sums: dict[str, float] = {}
        component_counts: dict[str, int] = {}
        for result in results:
            for component, value in result.score.components.items():
                component_sums[component] = component_sums.get(component, 0.0) + value
                component_counts[component] = component_counts.get(component, 0) + 1
        breakdown[agent_name] = {
            component: round(component_sums[component] / component_counts[component], 3)
            for component in sorted(component_sums)
        }
    return breakdown


def _per_episode_dump(
    registry: Registry,
    agent_results: dict[str, list[EpisodeResult]],
) -> list[dict[str, Any]]:
    """Per-episode scores for every agent, sorted by world_id."""
    episodes: dict[str, dict[str, Any]] = {}
    for agent_name, results in agent_results.items():
        for result in results:
            entry = episodes.setdefault(result.world_id, {
                "world_id": result.world_id,
                "causal_label": registry.worlds[result.world_id].causal_label,
            })
            entry[agent_name] = round(result.score.total, 2)
    return [episodes[wid] for wid in sorted(episodes)]


def _phase1_gates(
    world_count: int,
    public_accuracy: float,
    agent_scores: dict[str, list[float]],
) -> dict[str, bool]:
    oracle_mean = mean(agent_scores.get("oracle", [0.0]))
    scripted_mean = mean(agent_scores.get("scripted", [0.0]))
    naive_mean = mean(agent_scores.get("naive", [100.0]))
    random_mean = mean(agent_scores.get("random", [100.0]))
    return {
        "at_least_100_certified_simulator_episodes": world_count >= 100,
        "public_only_classifier_at_most_35": public_accuracy <= 0.35,
        "oracle_score_at_least_92": oracle_mean >= 92.0,
        "scripted_score_65_to_80": 65.0 <= scripted_mean <= 80.0,
        "naive_score_at_most_45": naive_mean <= 45.0,
        "random_score_at_most_35": random_mean <= 35.0,
    }


def public_only_world_classifier_accuracy(registry: Registry, world_ids: list[str]) -> float:
    """Leave-one-out nearest-centroid classifier over public headline features."""

    if len(world_ids) <= 1:
        return 1.0
    correct = 0
    for held_out in world_ids:
        train_ids = [world_id for world_id in world_ids if world_id != held_out]
        centroids = _centroids(registry, train_ids)
        prediction = _predict_label(registry, held_out, centroids)
        if prediction == registry.worlds[held_out].causal_label:
            correct += 1
    return round(correct / len(world_ids), 4)


def _centroids(registry: Registry, world_ids: list[str]) -> dict[str, list[float]]:
    by_label: dict[str, list[list[float]]] = {}
    for world_id in world_ids:
        world = registry.worlds[world_id]
        by_label.setdefault(world.causal_label, []).append(_public_features(world))
    return {
        label: [
            sum(feature[index] for feature in features) / len(features)
            for index in range(len(features[0]))
        ]
        for label, features in by_label.items()
    }


def _predict_label(registry: Registry, world_id: str, centroids: dict[str, list[float]]) -> str:
    features = _public_features(registry.worlds[world_id])
    return min(
        centroids,
        key=lambda label: _distance(features, centroids[label]),
    )


def _public_features(world) -> list[float]:
    baseline = _headline_metric(world.headline.baseline_metrics)
    proposed = _headline_metric(world.headline.proposed_metrics)
    return [
        baseline,
        proposed,
        proposed - baseline,
        len(world.headline.description) / 200.0,
    ]


def _headline_metric(metrics: dict[str, float]) -> float:
    if "validation_id.accuracy" in metrics:
        return float(metrics["validation_id.accuracy"])
    return float(next(iter(metrics.values())))


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))

