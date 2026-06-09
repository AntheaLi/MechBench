"""Factorized simulator family for v0 causal-world calibration."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from mechbench.core.family import FamilyInterface, World
from mechbench.interface.schemas import ActionSpec, ExperimentRequest, ExperimentResult


FACTOR_KEYS = (
    "mechanism",
    "capacity",
    "compute",
    "norm",
    "seed",
    "task",
    "interactions",
)


class Family(FamilyInterface):
    """Deterministic simulator using explicit additive causal factors."""

    def get_available_actions(self, world: World) -> list[ActionSpec]:
        actions = self.config.get("actions", [])
        simulator = world.config.get("simulator", {})
        variants = set(simulator.get("variants", {}))
        special_actions = {"multi_seed_replication"}
        available = []
        for item in actions:
            action = ActionSpec.from_dict(item)
            if action.name in variants or action.name in special_actions:
                available.append(action)
        return available

    def estimate_cost(self, world: World, request: ExperimentRequest) -> float:
        action_costs = {
            action.name: action.default_cost for action in self.get_available_actions(world)
        }
        base_cost = action_costs.get(request.variant, 1.0)
        split_multiplier = max(1, len(request.eval_splits))
        collect_cost = 0.05 * len(request.collect)
        return round(base_cost * (1.0 + 0.08 * (split_multiplier - 1)) + collect_cost, 3)

    def run_experiment(self, world: World, request: ExperimentRequest) -> ExperimentResult:
        simulator = world.config.get("simulator", {})
        if request.variant == "multi_seed_replication":
            return self._run_multi_seed_replication(world, request)

        variant_config = simulator.get("variants", {}).get(request.variant)
        if variant_config is None:
            raise ValueError(f"unknown simulator variant: {request.variant}")

        metrics = {}
        base_accuracy = simulator.get("base_accuracy", {})
        split_effects = variant_config.get("split_effects", {})
        for split in request.eval_splits:
            baseline = float(base_accuracy.get(split, base_accuracy.get("validation_id", 0.70)))
            delta = self._variant_delta(world, request, split, variant_config)
            metrics[f"{split}.accuracy"] = round(self._clamp_accuracy(baseline + delta), 4)
            metrics[f"{split}.delta"] = round(delta, 4)

        model_stats = {
            "parameter_count": int(variant_config.get("parameter_count", simulator.get("base_parameter_count", 1_000_000))),
            "estimated_train_flops": float(
                variant_config.get("estimated_train_flops", simulator.get("base_train_flops", 1.0e9))
            ),
            "branch_base_norm_ratio": float(variant_config.get("branch_base_norm_ratio", 0.0)),
            "simulator_factor_sum": round(
                sum(float(variant_config.get("factors", {}).get(key, 0.0)) for key in FACTOR_KEYS),
                4,
            ),
        }
        artifacts = {
            "metrics": "simulator/generated_metrics.json",
            "training_curve": "simulator/generated_training_curve.json",
        }
        if "activation_norms" in request.collect:
            artifacts["norm_summary"] = "simulator/generated_norms.json"

        return ExperimentResult(
            experiment_id="",
            status="completed",
            cost=self.estimate_cost(world, request),
            config_hash="",
            model_stats=model_stats,
            metrics=metrics,
            artifacts=artifacts,
        )

    def _run_multi_seed_replication(self, world: World, request: ExperimentRequest) -> ExperimentResult:
        simulator = world.config.get("simulator", {})
        variants = simulator.get("variants", {})
        baseline_config = variants["baseline"]
        proposed_config = variants["proposed"]
        base_accuracy = simulator.get("base_accuracy", {})
        n_seeds = int(request.params.get("n_seeds", 5) or 5)
        n_seeds = max(2, min(n_seeds, 20))

        metrics = {}
        for split in request.eval_splits:
            baseline = float(base_accuracy.get(split, base_accuracy.get("validation_id", 0.70)))
            seed_deltas = []
            proposed_accuracies = []
            for seed in range(n_seeds):
                seeded_request = ExperimentRequest(
                    variant="proposed",
                    train_seed=seed,
                    data_seed=request.data_seed + seed,
                    train_tokens=request.train_tokens,
                    eval_splits=[split],
                    collect=request.collect,
                    hypothesis_tested=request.hypothesis_tested,
                    params=request.params,
                )
                baseline_request = ExperimentRequest(
                    variant="baseline",
                    train_seed=seed,
                    data_seed=request.data_seed + seed,
                    train_tokens=request.train_tokens,
                    eval_splits=[split],
                    collect=request.collect,
                    hypothesis_tested=request.hypothesis_tested,
                    params=request.params,
                )
                baseline_delta = self._variant_delta(world, baseline_request, split, baseline_config)
                proposed_delta = self._variant_delta(world, seeded_request, split, proposed_config)
                seed_deltas.append(proposed_delta - baseline_delta)
                proposed_accuracies.append(self._clamp_accuracy(baseline + proposed_delta))
            mean_delta = sum(seed_deltas) / len(seed_deltas)
            mean_accuracy = sum(proposed_accuracies) / len(proposed_accuracies)
            variance = sum((delta - mean_delta) ** 2 for delta in seed_deltas) / len(seed_deltas)
            metrics[f"{split}.accuracy"] = round(mean_accuracy, 4)
            metrics[f"{split}.delta"] = round(mean_delta, 4)
            metrics[f"{split}.delta_std"] = round(variance**0.5, 4)

        return ExperimentResult(
            experiment_id="",
            status="completed",
            cost=self.estimate_cost(world, request),
            config_hash="",
            model_stats={
                "parameter_count": int(proposed_config.get("parameter_count", simulator.get("base_parameter_count", 1_000_000))),
                "estimated_train_flops": float(proposed_config.get("estimated_train_flops", simulator.get("base_train_flops", 1.0e9))),
                "branch_base_norm_ratio": float(proposed_config.get("branch_base_norm_ratio", 0.0)),
                "seed_panel_size": n_seeds,
            },
            metrics=metrics,
            artifacts={
                "metrics": "simulator/generated_multiseed_metrics.json",
                "training_curve": "simulator/generated_multiseed_training_curve.json",
            },
        )

    def analyze_artifact(self, world: World, experiment_id: str, analysis_spec: dict[str, Any]) -> dict[str, Any]:
        return {
            "experiment_id": experiment_id,
            "status": "completed",
            "analysis_type": analysis_spec.get("type", "summary"),
            "summary": "Simulator artifacts expose public metrics and norm summaries only.",
        }

    def _variant_delta(
        self,
        world: World,
        request: ExperimentRequest,
        split: str,
        variant_config: dict[str, Any],
    ) -> float:
        factors = variant_config.get("factors", {})
        delta = sum(float(factors.get(key, 0.0)) for key in FACTOR_KEYS)
        delta += float(variant_config.get("split_effects", {}).get(split, 0.0))
        delta += self._configured_seed_effect(request, split, variant_config)
        delta += self._seed_noise(world, request, split, variant_config)
        return delta

    def _configured_seed_effect(
        self,
        request: ExperimentRequest,
        split: str,
        variant_config: dict[str, Any],
    ) -> float:
        seed_effects = variant_config.get("seed_effects", {})
        split_effects = seed_effects.get(split, seed_effects.get("*", {}))
        if not isinstance(split_effects, dict):
            return 0.0
        keys = (
            f"{request.train_seed}:{request.data_seed}",
            str(request.train_seed),
            "default",
        )
        for key in keys:
            if key in split_effects:
                return float(split_effects[key])
        return 0.0

    def _seed_noise(
        self,
        world: World,
        request: ExperimentRequest,
        split: str,
        variant_config: dict[str, Any],
    ) -> float:
        scale = float(variant_config.get("seed_noise", world.config.get("simulator", {}).get("seed_noise", 0.0)))
        if scale <= 0:
            return 0.0
        key = f"{world.world_id}:{request.variant}:{request.train_seed}:{request.data_seed}:{split}"
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        raw = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        return (raw - 0.5) * 2.0 * scale

    def _clamp_accuracy(self, value: float) -> float:
        return max(0.0, min(1.0, value))
