"""PyTorch vertical-slice family for structured attention-branch episodes."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

from mechbench.core.family import FamilyInterface, World
from mechbench.interface.schemas import ActionSpec, ExperimentRequest, ExperimentResult

_FAMILY_ROOT = Path(__file__).resolve().parent
if str(_FAMILY_ROOT) not in sys.path:
    sys.path.insert(0, str(_FAMILY_ROOT))

try:
    from base_model.model import ModelConfig
    from base_model.train import TrainConfig, train_model
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    ModelConfig = None  # type: ignore[misc,assignment]
    TrainConfig = None  # type: ignore[misc,assignment]
    train_model = None  # type: ignore[misc,assignment]


class Family(FamilyInterface):
    """Train/evaluate tiny retrieval models under configured branch variants."""

    SPECIAL_ACTIONS = {"multi_seed_replication"}

    def __init__(self, root, config):
        super().__init__(root, config)
        self._cache: dict[str, dict[str, Any]] = {}

    def get_available_actions(self, world: World) -> list[ActionSpec]:
        variants = set(world.config.get("variants", {}))
        available = []
        for item in self.config.get("actions", []):
            action = ActionSpec.from_dict(item)
            if action.name in variants or action.name in self.SPECIAL_ACTIONS:
                available.append(action)
        return available

    def estimate_cost(self, world: World, request: ExperimentRequest) -> float:
        base = super().estimate_cost(world, request)
        train_cfg = self._training_config_dict(world, request.variant, request)
        epochs = max(1, int(train_cfg.get("epochs", 1)))
        examples = max(1, int(train_cfg.get("train_examples", 1)))
        reference = float(world.config.get("cost_reference_examples", 512))
        return round(base * epochs * examples / reference, 3)

    def run_experiment(self, world: World, request: ExperimentRequest) -> ExperimentResult:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("attn_branch family requires PyTorch (pip install torch)")
        if request.variant == "multi_seed_replication":
            return self._run_multi_seed(world, request)
        if request.variant not in world.config.get("variants", {}):
            raise ValueError(f"unknown attn_branch variant: {request.variant}")
        payload = self._run_variant(world, request.variant, request)
        return ExperimentResult.from_dict(payload)

    def _run_multi_seed(self, world: World, request: ExperimentRequest) -> ExperimentResult:
        max_panel = int(world.config.get("max_seed_panel", 3))
        n_seeds = int(request.params.get("n_seeds", max_panel) or max_panel)
        n_seeds = max(2, min(n_seeds, max_panel))

        per_seed = []
        for seed in range(n_seeds):
            seeded_request = replace(
                request,
                variant="proposed",
                train_seed=seed,
                data_seed=request.data_seed + seed,
                params={},
            )
            per_seed.append(self._run_variant(world, "proposed", seeded_request))

        metrics: dict[str, float] = {}
        for split in request.eval_splits:
            accuracies = [item["metrics"][f"{split}.accuracy"] for item in per_seed]
            deltas = [item["metrics"][f"{split}.delta"] for item in per_seed]
            metrics[f"{split}.accuracy"] = round(mean(accuracies), 4)
            metrics[f"{split}.delta"] = round(mean(deltas), 4)
            metrics[f"{split}.delta_std"] = round(_std(deltas), 4)

        model_stats = dict(per_seed[-1]["model_stats"])
        model_stats["seed_panel_size"] = n_seeds
        model_stats["seed_panel_train_seeds"] = list(range(n_seeds))
        return ExperimentResult.from_dict(
            {
                "experiment_id": "",
                "status": "completed",
                "cost": self.estimate_cost(world, request),
                "config_hash": "",
                "model_stats": model_stats,
                "metrics": metrics,
                "artifacts": {
                    "metrics": "attn_branch/generated_multiseed_metrics.json",
                    "training_curve": "attn_branch/generated_multiseed_training_curve.json",
                },
            }
        )

    def _run_variant(self, world: World, variant: str, request: ExperimentRequest) -> dict[str, Any]:
        cache_key = self._cache_key(world, variant, request)
        if cache_key in self._cache:
            return deepcopy(self._cache[cache_key])

        model_config = ModelConfig.from_dict(self._model_config_dict(world, variant))
        train_config = TrainConfig.from_dict(self._training_config_dict(world, variant, request))
        result = train_model(model_config, train_config)

        metrics = self._metrics_with_deltas(world, variant, request, result.final_metrics)
        model_stats = dict(result.model_stats)
        model_stats["estimated_train_flops"] = _estimated_train_flops(model_stats, train_config)
        model_stats["metric_adjustment"] = round(
            self._metric_adjustment(world, variant, "validation_id", request),
            4,
        )

        payload = {
            "experiment_id": "",
            "status": "completed",
            "cost": self.estimate_cost(world, request),
            "config_hash": "",
            "model_stats": model_stats,
            "metrics": metrics,
            "artifacts": {
                "metrics": "attn_branch/generated_metrics.json",
                "training_curve": "attn_branch/generated_training_curve.json",
            },
        }
        self._cache[cache_key] = deepcopy(payload)
        return payload

    def _model_config_dict(self, world: World, variant: str) -> dict[str, Any]:
        model_config = _merge_dicts(
            world.config.get("model", {}).get("base", {}),
            world.config.get("variants", {}).get(variant, {}).get("model", {}),
        )
        return model_config

    def _training_config_dict(self, world: World, variant: str, request: ExperimentRequest) -> dict[str, Any]:
        training = _merge_dicts(
            world.config.get("training", {}).get("base", {}),
            world.config.get("variants", {}).get(variant, {}).get("training", {}),
        )
        training["train_seed"] = request.train_seed
        training["data_seed"] = request.data_seed
        training["eval_splits"] = list(request.eval_splits)
        if request.train_tokens > 0:
            training["train_examples"] = request.train_tokens
        for key, value in request.params.get("training", {}).items():
            training[key] = value
        return training

    def _metrics_with_deltas(
        self,
        world: World,
        variant: str,
        request: ExperimentRequest,
        final_metrics: dict[str, float],
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for split in request.eval_splits:
            measured_key = f"{split}.accuracy"
            measured = float(final_metrics.get(measured_key, 0.0))
            adjustment = self._metric_adjustment(world, variant, split, request)
            baseline = self._baseline_reference(world, split)
            target_delta = self._target_delta(world, variant, split)
            if target_delta is None:
                accuracy = _clamp(measured + adjustment)
                delta = accuracy - baseline
            else:
                delta = target_delta + adjustment
                accuracy = _clamp(baseline + delta)
            metrics[measured_key] = round(accuracy, 4)
            metrics[f"{split}.measured_accuracy"] = round(measured, 4)
            metrics[f"{split}.delta"] = round(delta, 4)
        metrics["train_accuracy"] = round(float(final_metrics.get("train_accuracy", 0.0)), 4)
        metrics["train_loss"] = round(float(final_metrics.get("train_loss", 0.0)), 4)
        return metrics

    def _baseline_reference(self, world: World, split: str) -> float:
        references = world.config.get("baseline_reference_metrics", {})
        key = f"{split}.accuracy"
        if key in references:
            return float(references[key])
        if key in world.headline.baseline_metrics:
            return float(world.headline.baseline_metrics[key])
        return float(next(iter(world.headline.baseline_metrics.values())))

    def _metric_adjustment(
        self,
        world: World,
        variant: str,
        split: str,
        request: ExperimentRequest,
    ) -> float:
        variant_config = world.config.get("variants", {}).get(variant, {})
        adjustments = variant_config.get("metric_adjustments", {})
        adjustment = float(adjustments.get(f"{split}.accuracy", adjustments.get(split, 0.0)) or 0.0)

        seed_effects = variant_config.get("seed_effects", {})
        split_effects = seed_effects.get(split, seed_effects.get("*", {}))
        if isinstance(split_effects, dict):
            keys = (
                f"{request.train_seed}:{request.data_seed}",
                str(request.train_seed),
                "default",
            )
            for key in keys:
                if key in split_effects:
                    adjustment += float(split_effects[key])
                    break
        return adjustment

    def _target_delta(self, world: World, variant: str, split: str) -> float | None:
        variant_config = world.config.get("variants", {}).get(variant, {})
        target_deltas = variant_config.get("target_deltas", {})
        value = target_deltas.get(f"{split}.delta", target_deltas.get(split))
        if value is None:
            return None
        return float(value)

    def _cache_key(self, world: World, variant: str, request: ExperimentRequest) -> str:
        return json.dumps(
            {
                "world_id": world.world_id,
                "variant": variant,
                "request": request.to_dict(),
                "model": self._model_config_dict(world, variant),
                "training": self._training_config_dict(world, variant, request),
            },
            sort_keys=True,
        )


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _estimated_train_flops(model_stats: dict[str, Any], train_config: TrainConfig) -> float:
    params = float(model_stats.get("trainable_parameter_count", model_stats.get("parameter_count", 0)))
    return round(6.0 * params * train_config.train_examples * train_config.epochs * train_config.seq_len, 1)


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
