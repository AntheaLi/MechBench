"""Offline compiler for factorized simulator episodes."""

from __future__ import annotations

import random
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mechbench.utils.config import dump_json, load_config


WORLD_TYPES = (
    "true_mechanism",
    "seed_hacking",
    "parameter_laundering",
    "norm_laundering",
    "compute_laundering_lite",
)

TEMPLATE_BY_WORLD_TYPE = {
    "true_mechanism": "true_mechanism_001",
    "seed_hacking": "seed_hacking_001",
    "parameter_laundering": "parameter_laundering_001",
    "norm_laundering": "norm_laundering_001",
    "compute_laundering_lite": "compute_laundering_lite_001",
}


@dataclass(frozen=True)
class CompileResult:
    output_dir: Path
    episode_ids: list[str]
    counts_by_world_type: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "episode_ids": list(self.episode_ids),
            "counts_by_world_type": dict(self.counts_by_world_type),
        }


class SimulatorEpisodeCompiler:
    """Compile frozen simulator worlds from declarative templates."""

    def __init__(
        self,
        families_dir: str | Path = "families",
        output_dir: str | Path | None = None,
        seed: int = 0,
    ):
        self.families_dir = Path(families_dir)
        self.simulator_dir = self.families_dir / "simulator"
        self.worlds_dir = self.simulator_dir / "worlds"
        self.output_dir = Path(output_dir) if output_dir else self.worlds_dir
        self.random = random.Random(seed)

    def compile(self, count: int = 100, start_index: int = 1, clear: bool = False) -> CompileResult:
        if count < len(WORLD_TYPES):
            raise ValueError(f"count must be at least {len(WORLD_TYPES)}")
        if clear:
            self._clear_compiled_outputs()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        episode_ids: list[str] = []
        counts = {world_type: 0 for world_type in WORLD_TYPES}
        offset = 0
        attempts = 0
        max_attempts = count * 20
        while len(episode_ids) < count and attempts < max_attempts:
            world_type = WORLD_TYPES[offset % len(WORLD_TYPES)]
            serial = start_index + offset
            attempts += 1
            offset += 1
            config = self._compile_one(world_type, serial)
            if not self._certify(config):
                continue
            world_name = config["name"]
            world_dir = self.output_dir / world_name
            world_dir.mkdir(parents=True, exist_ok=True)
            dump_json(world_dir / "world.yaml", config)
            episode_ids.append(config["world_id"])
            counts[world_type] += 1
        if len(episode_ids) < count:
            raise RuntimeError(f"only compiled {len(episode_ids)} of {count} requested episodes")
        return CompileResult(self.output_dir, episode_ids, counts)

    def _clear_compiled_outputs(self) -> None:
        if not self.output_dir.exists():
            return
        for path in self.output_dir.iterdir():
            if path.is_dir() and path.name.startswith("compiled_"):
                shutil.rmtree(path)

    def _load_template(self, world_type: str) -> dict[str, Any]:
        template_name = TEMPLATE_BY_WORLD_TYPE[world_type]
        return load_config(self.worlds_dir / template_name / "world.yaml")

    def _compile_one(self, world_type: str, serial: int) -> dict[str, Any]:
        config = deepcopy(self._load_template(world_type))
        target_delta = self.random.uniform(0.027, 0.041)
        baseline = self.random.uniform(0.682, 0.718)
        nuisance_shift = self.random.uniform(-0.004, 0.004)
        seed_noise = self.random.uniform(0.0004, 0.0014)

        name = f"compiled_{serial:04d}"
        config["name"] = name
        config["world_id"] = f"simulator/{name}"
        config["headline"] = {
            "baseline_metrics": {"validation_id.accuracy": round(baseline, 4)},
            "proposed_metrics": {"validation_id.accuracy": round(baseline + target_delta, 4)},
            "description": (
                "The structured routing branch improves validation accuracy "
                f"from {baseline:.3f} to {baseline + target_delta:.3f} on the public validation split."
            ),
        }
        config["simulator"]["base_accuracy"] = {
            "validation_id": round(baseline, 4),
            "validation_long": round(baseline - 0.034 + nuisance_shift, 4),
            "validation_position_shift": round(baseline - 0.021 - nuisance_shift, 4),
            "validation_distractor_shift": round(baseline - 0.026 + nuisance_shift / 2.0, 4),
        }
        config["simulator"]["seed_noise"] = round(seed_noise, 5)
        config["budget"] = {"max_cost": round(self.random.uniform(5.2, 6.2), 2)}

        getattr(self, f"_mutate_{world_type}")(config, target_delta)
        self._set_common_certificate(config, world_type, target_delta)
        return config

    def _set_common_certificate(self, config: dict[str, Any], world_type: str, target_delta: float) -> None:
        exhaustive = self.random.uniform(11.5, 15.0)
        oracle = self.random.uniform(2.4, 4.1)
        config["certificate"] = {
            "episode_id": config["world_id"],
            "world_type": world_type,
            "certificate_version": "0.1",
            "public_only_predictability": 0.20,
            "oracle_minimum_cost": round(oracle, 2),
            "exhaustive_cost": round(exhaustive, 2),
            "private_seed_stability": round(self.random.uniform(0.88, 0.97), 2)
            if world_type != "seed_hacking"
            else round(self.random.uniform(0.08, 0.25), 2),
            "minimum_intervention_margin": 0.015,
            "observed_intervention_margin": round(max(0.016, target_delta - 0.006), 4),
            "prediction_tolerance": 0.02,
            "certified": True,
            "compiler_version": "0.1",
        }

    def _certify(self, config: dict[str, Any]) -> bool:
        checks = self._certification_checks(config)
        config["certificate"]["checks"] = checks
        config["certificate"]["certified"] = all(checks.values())
        return config["certificate"]["certified"]

    def _certification_checks(self, config: dict[str, Any]) -> dict[str, bool]:
        baseline = config["headline"]["baseline_metrics"]["validation_id.accuracy"]
        proposed = config["headline"]["proposed_metrics"]["validation_id.accuracy"]
        public_delta = proposed - baseline
        world_type = config["causal_label"]
        checks = {
            "headline_baseline_in_shared_range": 0.66 <= baseline <= 0.76,
            "headline_delta_in_shared_range": 0.025 <= public_delta <= 0.05,
            "has_at_least_three_private_interventions": len(config.get("interventions", [])) >= 3,
            "minimum_intervention_margin_met": config["certificate"]["observed_intervention_margin"] >= config["certificate"]["minimum_intervention_margin"],
        }
        checks.update(getattr(self, f"_pattern_checks_{world_type}")(config, public_delta))
        return checks

    def _configured_public_delta(self, config: dict[str, Any], variant: str) -> float:
        variant_config = config["simulator"]["variants"][variant]
        delta = sum(float(value) for value in variant_config["factors"].values())
        seed_effects = variant_config.get("seed_effects", {}).get("validation_id", {})
        if isinstance(seed_effects, dict):
            delta += float(seed_effects.get("0:0", 0.0))
        return delta

    def _expected_delta(self, config: dict[str, Any], intervention_id: str) -> float:
        for intervention in config.get("interventions", []):
            if intervention["id"] == intervention_id:
                return float(intervention["expected_delta"])
        raise KeyError(intervention_id)

    def _pattern_checks_true_mechanism(self, config: dict[str, Any], public_delta: float) -> dict[str, bool]:
        proposed = self._configured_public_delta(config, "proposed")
        generic = self._configured_public_delta(config, "generic_learned_branch")
        destroyed = self._configured_public_delta(config, "geometry_destroyed_branch")
        norm_matched = self._configured_public_delta(config, "norm_matched_proposed")
        return {
            "structured_branch_has_practical_gain": proposed >= 0.025,
            "generic_capacity_does_not_recover_gain": generic <= proposed - 0.015,
            "geometry_destruction_removes_gain": destroyed <= proposed - 0.015,
            "norm_matching_preserves_gain": norm_matched >= proposed - 0.006,
        }

    def _pattern_checks_seed_hacking(self, config: dict[str, Any], public_delta: float) -> dict[str, bool]:
        truth_delta = float(config["improvement_truth"]["delta"])
        return {
            "public_demo_seed_positive": public_delta >= 0.025,
            "private_mean_below_practical_threshold": truth_delta <= 0.012,
            "preregistered_seed_panel_removes_gain": self._expected_delta(config, "hidden_preregistered_seed_panel") <= 0.012,
            "fresh_data_seed_removes_gain": self._expected_delta(config, "hidden_fresh_data_seed") <= 0.012,
        }

    def _pattern_checks_parameter_laundering(self, config: dict[str, Any], public_delta: float) -> dict[str, bool]:
        proposed = self._configured_public_delta(config, "proposed")
        generic = self._configured_public_delta(config, "generic_learned_branch")
        wider = self._configured_public_delta(config, "parameter_matched_wider")
        destroyed = self._configured_public_delta(config, "geometry_destroyed_branch")
        fixed = self._configured_public_delta(config, "fixed_random_branch")
        return {
            "generic_learned_branch_recovers_gain": generic >= proposed - 0.007,
            "parameter_matched_wider_recovers_gain": wider >= proposed - 0.008,
            "geometry_destroyed_branch_recovers_gain": destroyed >= proposed - 0.008,
            "fixed_random_branch_underperforms": fixed <= proposed - 0.015,
        }

    def _pattern_checks_norm_laundering(self, config: dict[str, Any], public_delta: float) -> dict[str, bool]:
        proposed = self._configured_public_delta(config, "proposed")
        norm_matched = self._configured_public_delta(config, "norm_matched_proposed")
        destroyed = self._configured_public_delta(config, "geometry_destroyed_branch")
        scale_generic = self._configured_public_delta(config, "scale_matched_generic")
        proposed_norm = config["simulator"]["variants"]["proposed"]["branch_base_norm_ratio"]
        matched_norm = config["simulator"]["variants"]["norm_matched_proposed"]["branch_base_norm_ratio"]
        return {
            "proposed_branch_is_loud": proposed_norm >= 1.2,
            "norm_matching_reduces_gain": norm_matched <= proposed - 0.015,
            "structure_destroyed_norm_preserved_recovers_gain": destroyed >= proposed - 0.006,
            "scale_matched_generic_recovers_gain": scale_generic >= proposed - 0.006,
            "norm_ratio_reduced_by_matching": matched_norm <= proposed_norm - 0.5,
        }

    def _pattern_checks_compute_laundering_lite(self, config: dict[str, Any], public_delta: float) -> dict[str, bool]:
        proposed = self._configured_public_delta(config, "proposed")
        baseline_compute = self._configured_public_delta(config, "baseline_compute_matched")
        proposed_limited = self._configured_public_delta(config, "proposed_compute_limited")
        learning_curve = self._configured_public_delta(config, "matched_flop_learning_curve")
        proposed_flops = config["simulator"]["variants"]["proposed"]["estimated_train_flops"]
        baseline_flops = config["simulator"]["variants"]["baseline"]["estimated_train_flops"]
        return {
            "proposed_gets_more_estimated_compute": proposed_flops > baseline_flops,
            "baseline_compute_match_recovers_gain": baseline_compute >= proposed - 0.008,
            "proposed_compute_limit_removes_gain": proposed_limited <= proposed - 0.015,
            "matched_flop_learning_curve_removes_gain": learning_curve <= proposed - 0.015,
        }

    def _set_factors(self, config: dict[str, Any], variant: str, **factors: float) -> None:
        current = config["simulator"]["variants"][variant]["factors"]
        for key in current:
            current[key] = round(float(factors.get(key, current.get(key, 0.0))), 4)

    def _factor_delta(self, config: dict[str, Any], variant: str) -> float:
        return sum(float(value) for value in config["simulator"]["variants"][variant]["factors"].values())

    def _set_intervention(self, config: dict[str, Any], intervention_id: str, delta: float, qualitative: str) -> None:
        for intervention in config["interventions"]:
            if intervention["id"] == intervention_id:
                intervention["expected_delta"] = round(delta, 4)
                intervention["qualitative_result"] = qualitative
                return
        raise KeyError(intervention_id)

    def _mutate_true_mechanism(self, config: dict[str, Any], target_delta: float) -> None:
        mechanism = target_delta - 0.006
        self._set_factors(config, "proposed", mechanism=mechanism, capacity=0.004, compute=0.001, norm=0.001)
        self._set_factors(config, "norm_matched_proposed", mechanism=mechanism, capacity=0.004, compute=0.001, norm=0.0)
        self._set_factors(config, "generic_learned_branch", mechanism=0.0, capacity=0.007, compute=0.001, norm=0.001)
        self._set_factors(config, "parameter_matched_wider", mechanism=0.0, capacity=0.007, compute=0.001, norm=0.0, interactions=-0.001)
        self._set_factors(config, "geometry_destroyed_branch", mechanism=0.0, capacity=0.004, compute=0.001, norm=0.001)
        self._set_factors(config, "scale_matched_generic", mechanism=0.0, capacity=0.007, compute=0.001, norm=0.004)
        self._set_intervention(config, "hidden_destroy_routing_structure", 0.006, "gain_disappears")
        self._set_intervention(config, "hidden_unseen_task_structure_preserved", target_delta - 0.004, "gain_survives")
        self._set_intervention(config, "hidden_unseen_seed_panel", target_delta - 0.002, "gain_survives")
        # Partial ablation: mechanism at ~45% capacity → gain shrinks but persists.
        partial_delta = round(target_delta * 0.45, 4)
        self._set_intervention(config, "hidden_partial_mechanism_ablation", partial_delta, "gain_shrinks")
        config["improvement_truth"] = {"status": "reproducible_positive_gain", "delta": round(target_delta, 4)}

    def _mutate_seed_hacking(self, config: dict[str, Any], target_delta: float) -> None:
        stable_delta = self.random.uniform(0.001, 0.006)
        seed_bonus = target_delta - stable_delta
        self._set_factors(config, "proposed", mechanism=0.0, capacity=stable_delta - 0.001, compute=0.001, norm=0.0)
        config["simulator"]["variants"]["proposed"]["seed_effects"] = {
            "validation_id": {"0:0": round(seed_bonus, 4)},
            "validation_long": {"0:0": round(seed_bonus * 0.75, 4)},
            "validation_position_shift": {"0:0": round(seed_bonus * 0.65, 4)},
        }
        self._set_factors(config, "norm_matched_proposed", mechanism=0.0, capacity=stable_delta - 0.001, compute=0.001, norm=0.0)
        config["simulator"]["variants"]["norm_matched_proposed"]["seed_effects"] = {
            "validation_id": {"0:0": round(seed_bonus * 0.95, 4)}
        }
        self._set_intervention(config, "hidden_preregistered_seed_panel", stable_delta, "gain_disappears")
        self._set_intervention(config, "hidden_fixed_checkpoint_rule", max(0.0, stable_delta - 0.001), "gain_disappears")
        self._set_intervention(config, "hidden_fresh_data_seed", stable_delta + 0.001, "gain_disappears")
        config["improvement_truth"] = {"status": "not_reliably_positive", "delta": round(stable_delta, 4)}

    def _mutate_parameter_laundering(self, config: dict[str, Any], target_delta: float) -> None:
        capacity = target_delta - 0.004
        self._set_factors(config, "proposed", mechanism=0.001, capacity=capacity, compute=0.001, norm=0.002)
        self._set_factors(config, "generic_learned_branch", mechanism=0.0, capacity=target_delta - 0.003, compute=0.001, norm=0.001)
        self._set_factors(config, "parameter_matched_wider", mechanism=0.0, capacity=target_delta - 0.002, compute=0.001, norm=0.0, interactions=-0.001)
        self._set_factors(config, "geometry_destroyed_branch", mechanism=0.0, capacity=target_delta - 0.003, compute=0.001, norm=0.001)
        self._set_factors(config, "norm_matched_proposed", mechanism=0.001, capacity=capacity, compute=0.001, norm=0.0)
        self._set_factors(config, "scale_matched_generic", mechanism=0.0, capacity=target_delta - 0.003, compute=0.001, norm=0.002)
        self._set_factors(config, "fixed_random_branch", mechanism=0.0, capacity=0.001, compute=0.001, norm=0.0)
        self._set_intervention(config, "hidden_equal_param_generic_learned_branch", target_delta - 0.002, "gain_survives")
        self._set_intervention(config, "hidden_destroy_geometry_preserve_norm", target_delta - 0.003, "gain_survives")
        self._set_intervention(config, "hidden_unseen_seed_panel", target_delta - 0.002, "gain_survives")
        self._set_intervention(config, "hidden_fixed_random_branch", 0.003, "gain_disappears")
        config["improvement_truth"] = {"status": "reproducible_positive_gain", "delta": round(target_delta, 4)}

    def _mutate_norm_laundering(self, config: dict[str, Any], target_delta: float) -> None:
        norm_effect = target_delta - 0.006
        self._set_factors(config, "proposed", mechanism=0.002, capacity=0.003, compute=0.001, norm=norm_effect)
        self._set_factors(config, "geometry_destroyed_branch", mechanism=0.0, capacity=0.004, compute=0.001, norm=norm_effect)
        self._set_factors(config, "scale_matched_generic", mechanism=0.0, capacity=0.006, compute=0.001, norm=norm_effect)
        self._set_factors(config, "norm_matched_proposed", mechanism=0.002, capacity=0.003, compute=0.001, norm=0.001)
        self._set_factors(config, "generic_learned_branch", mechanism=0.0, capacity=0.006, compute=0.001, norm=0.001)
        self._set_factors(config, "parameter_matched_wider", mechanism=0.0, capacity=0.006, compute=0.001, norm=0.0)
        proposed_norm = self.random.uniform(1.25, 1.65)
        config["simulator"]["variants"]["proposed"]["branch_base_norm_ratio"] = round(proposed_norm, 2)
        config["simulator"]["variants"]["geometry_destroyed_branch"]["branch_base_norm_ratio"] = round(proposed_norm - 0.02, 2)
        config["simulator"]["variants"]["scale_matched_generic"]["branch_base_norm_ratio"] = round(proposed_norm, 2)
        config["simulator"]["variants"]["norm_matched_proposed"]["branch_base_norm_ratio"] = round(self.random.uniform(0.55, 0.72), 2)
        self._set_intervention(config, "hidden_reduce_proposed_norm", 0.006, "gain_disappears")
        self._set_intervention(config, "hidden_increase_generic_norm", target_delta - 0.001, "gain_survives")
        self._set_intervention(config, "hidden_destroy_geometry_preserve_norm", target_delta - 0.002, "gain_survives")
        config["improvement_truth"] = {"status": "reproducible_positive_gain", "delta": round(target_delta, 4)}

    def _mutate_compute_laundering_lite(self, config: dict[str, Any], target_delta: float) -> None:
        compute = target_delta - 0.007
        proposed_flops = self.random.uniform(1.32e9, 1.52e9)
        baseline_flops = 1.0e9
        self._set_factors(config, "proposed", mechanism=0.002, capacity=0.004, compute=compute, norm=0.001)
        self._set_factors(config, "baseline_compute_matched", mechanism=0.0, capacity=0.0, compute=target_delta - 0.002, norm=0.0)
        self._set_factors(config, "proposed_compute_limited", mechanism=0.002, capacity=0.004, compute=0.001, norm=0.001)
        self._set_factors(config, "matched_flop_learning_curve", mechanism=0.001, capacity=0.002, compute=0.0, norm=0.001)
        self._set_factors(config, "generic_learned_branch", mechanism=0.0, capacity=0.004, compute=compute, norm=0.001)
        self._set_factors(config, "geometry_destroyed_branch", mechanism=0.0, capacity=0.004, compute=compute, norm=0.001)
        self._set_factors(config, "norm_matched_proposed", mechanism=0.002, capacity=0.004, compute=compute, norm=0.0)
        for variant in config["simulator"]["variants"].values():
            if isinstance(variant, dict):
                variant.setdefault("estimated_train_flops", baseline_flops)
        config["simulator"]["variants"]["proposed"]["estimated_train_flops"] = proposed_flops
        config["simulator"]["variants"]["baseline_compute_matched"]["estimated_train_flops"] = proposed_flops
        config["simulator"]["variants"]["proposed_compute_limited"]["estimated_train_flops"] = baseline_flops
        config["simulator"]["variants"]["matched_flop_learning_curve"]["estimated_train_flops"] = baseline_flops
        self._set_intervention(config, "hidden_baseline_to_proposed_compute", target_delta - 0.003, "gain_survives")
        self._set_intervention(config, "hidden_restrict_proposed_to_baseline_compute", 0.005, "gain_disappears")
        self._set_intervention(config, "hidden_matched_compute_learning_curve", 0.004, "gain_disappears")
        config["improvement_truth"] = {"status": "reproducible_positive_gain", "delta": round(target_delta, 4)}
