"""Phase 4 study runner.

The study runner packages the first real agent study into a reproducible
one-command workflow: resolve a run pack, estimate API cost before any paid
call, run baseline and LLM agents across repeats, and emit a JSON payload that
the report renderer can summarize.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from agents.llm_scaffold import LLMScaffoldAgent
from agents.naive import NaiveScoreSeekingAgent
from agents.oracle import OracleAgent
from agents.random_policy import RandomExperimentAgent
from agents.scripted_causal_control import ScriptedCausalControlAgent
from mechbench.interface.schemas import WORLD_LABELS, EpisodeResult
from mechbench.registry import Registry
from mechbench.simulator.calibration import SIMULATOR_LABELS, simulator_world_ids
from mechbench.utils.config import dump_json


DEFAULT_PACK_NAME = "phase4_free_smoke"
DEFAULT_INPUT_TOKENS_PER_EPISODE = 3500
DEFAULT_OUTPUT_TOKENS_PER_EPISODE = 1200
SCHEMA_VERSION = "phase4_study_v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKS_DIR = REPO_ROOT / "study_packs"


# Estimates only. Packs can override these when a lab has exact contracted
# pricing or model aliases change.
DEFAULT_MODEL_PRICES_USD_PER_MILLION: dict[str, dict[str, float]] = {
    "gpt-5.5": {"input": 5.0, "output": 30.0},
    "gpt-5.4": {"input": 2.5, "output": 15.0},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.5},
    "claude-opus-4.8": {"input": 5.0, "output": 25.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "fable-5": {"input": 10.0, "output": 50.0},
    "gemini-3.1-pro-preview": {"input": 2.0, "output": 12.0},
    "gemini-3.5-pro": {"input": 2.0, "output": 12.0},
}

ZERO_COST_AGENTS = {"oracle", "scripted", "naive", "random", "llm_mock"}


@dataclass(frozen=True)
class CostEstimate:
    run_id: str
    episodes: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    known_price: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "episodes": self.episodes,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "known_price": self.known_price,
        }


def list_study_packs(packs_dir: str | Path | None = None) -> list[dict[str, str]]:
    """Return available run packs with names and descriptions."""

    root = Path(packs_dir) if packs_dir else PACKS_DIR
    packs: list[dict[str, str]] = []
    if not root.exists():
        return packs
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        packs.append(
            {
                "name": str(payload.get("name", path.stem)),
                "description": str(payload.get("description", "")),
                "path": str(path),
            }
        )
    return packs


def load_study_pack(pack: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load a study pack by dict, filesystem path, or pack name."""

    if isinstance(pack, dict):
        return dict(pack)
    pack_path = Path(pack)
    if pack_path.exists():
        return json.loads(pack_path.read_text(encoding="utf-8"))
    candidate = PACKS_DIR / f"{pack}.json"
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"unknown Phase 4 study pack: {pack}")


def run_phase4_study(
    pack: str | Path | dict[str, Any] = DEFAULT_PACK_NAME,
    *,
    families_dir: str = "families",
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    include_disabled: bool = False,
    worlds_override: str | None = None,
    count_override: int | None = None,
    repeats_override: int | None = None,
    max_cost_usd_override: float | None = None,
    allow_unknown_cost: bool = False,
    skip_missing_credentials: bool = False,
) -> dict[str, Any]:
    """Run or dry-run a Phase 4 study pack."""

    pack_payload = load_study_pack(pack)
    registry = Registry(families_dir)
    world_ids = resolve_world_ids(
        registry,
        pack_payload,
        worlds_override=worlds_override,
        count_override=count_override,
    )
    repeats = int(repeats_override if repeats_override is not None else pack_payload.get("repeats", 1))
    runs = _enabled_runs(pack_payload, include_disabled=include_disabled)
    skipped_runs: list[dict[str, str]] = []
    if not dry_run:
        if skip_missing_credentials:
            runs, skipped_runs = _drop_runs_missing_credentials(runs)
        else:
            _validate_runtime_credentials(runs)
    token_estimate = _token_estimate(pack_payload)
    preflight = estimate_study_cost(
        runs,
        world_count=len(world_ids),
        repeats=repeats,
        token_estimate=token_estimate,
    )
    max_cost_usd = (
        float(max_cost_usd_override)
        if max_cost_usd_override is not None
        else pack_payload.get("max_cost_usd")
    )
    _enforce_preflight_cost(
        preflight,
        max_cost_usd=max_cost_usd,
        allow_unknown_cost=allow_unknown_cost,
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pack": pack_payload,
        "resolved": {
            "families_dir": families_dir,
            "world_ids": world_ids,
            "repeats": repeats,
            "runs": [_public_run_spec(run) for run in runs],
            "skipped_runs": skipped_runs,
            "include_disabled": include_disabled,
        },
        "preflight": preflight,
        "dry_run": dry_run,
        "results": [],
        "summary": {},
    }

    if dry_run:
        payload["summary"] = summarize_phase4_payload(payload)
        return payload

    actual_cost_usd = 0.0
    stopped_reason = ""
    include_traces = bool(pack_payload.get("include_traces", True))
    for run in runs:
        run_id = str(run.get("id", run.get("agent", run.get("provider", "run"))))
        for repeat_index in range(repeats):
            for world_index, world_id in enumerate(world_ids):
                if stopped_reason:
                    break
                entry = _run_one_episode(
                    registry,
                    run,
                    run_id=run_id,
                    repeat_index=repeat_index,
                    world_index=world_index,
                    world_id=world_id,
                    include_traces=include_traces,
                )
                actual_cost_usd += float(entry.get("estimated_api_cost_usd", 0.0))
                entry["cumulative_estimated_api_cost_usd"] = round(actual_cost_usd, 6)
                payload["results"].append(entry)
                if max_cost_usd is not None and actual_cost_usd > float(max_cost_usd):
                    stopped_reason = (
                        f"runtime estimated API cost {actual_cost_usd:.4f} exceeded "
                        f"max_cost_usd {float(max_cost_usd):.4f}"
                    )
            if stopped_reason:
                break
        if stopped_reason:
            break
    if stopped_reason:
        payload["stopped_reason"] = stopped_reason

    payload["summary"] = summarize_phase4_payload(payload)
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        dump_json(root / "study.json", payload)
    return payload


def resolve_world_ids(
    registry: Registry,
    pack: dict[str, Any],
    *,
    worlds_override: str | None = None,
    count_override: int | None = None,
) -> list[str]:
    if worlds_override:
        if worlds_override == "all":
            return registry.list_worlds()
        return [item.strip() for item in worlds_override.split(",") if item.strip()]

    spec = dict(pack.get("worlds", {}))
    selector = str(spec.get("selector", "simulator_compiled_balanced"))
    count = int(count_override if count_override is not None else spec.get("count", 10))

    if selector == "explicit":
        ids = [str(item) for item in spec.get("ids", [])]
    elif selector == "simulator_compiled_all":
        ids = simulator_world_ids(registry, "compiled")
    elif selector == "simulator_manual_all":
        ids = simulator_world_ids(registry, "manual")
    elif selector == "simulator_compiled_balanced":
        ids = _balanced_simulator_subset(registry, count=count)
    elif selector == "attn_phase2_certified":
        ids = [
            world_id
            for world_id in registry.list_worlds("attn_branch")
            if registry.worlds[world_id].name.startswith("phase2_certified_")
        ]
        ids = sorted(ids)[:count] if count else sorted(ids)
    elif selector == "phase4_core_mixed":
        ids = _balanced_simulator_subset(registry, count=count)
        ids.extend(
            sorted(
                world_id
                for world_id in registry.list_worlds("attn_branch")
                if registry.worlds[world_id].name.startswith("phase2_certified_")
            )
        )
    else:
        raise ValueError(f"unknown world selector: {selector}")

    unknown = [world_id for world_id in ids if world_id not in registry.worlds]
    if unknown:
        raise ValueError(f"unknown worlds in study pack: {unknown}")
    return ids


def estimate_study_cost(
    runs: list[dict[str, Any]],
    *,
    world_count: int,
    repeats: int,
    token_estimate: dict[str, int],
) -> dict[str, Any]:
    estimates = [
        _estimate_run_cost(
            run,
            episodes=world_count * repeats,
            input_tokens_per_episode=token_estimate["input_tokens_per_episode"],
            output_tokens_per_episode=token_estimate["output_tokens_per_episode"],
        )
        for run in runs
    ]
    total = sum(item.cost_usd for item in estimates)
    unknown = [item.run_id for item in estimates if not item.known_price]
    return {
        "episodes_per_run": world_count * repeats,
        "input_tokens_per_episode": token_estimate["input_tokens_per_episode"],
        "output_tokens_per_episode": token_estimate["output_tokens_per_episode"],
        "runs": [item.to_dict() for item in estimates],
        "estimated_total_cost_usd": round(total, 6),
        "unknown_price_runs": unknown,
        "known_cost": not unknown,
    }


def summarize_phase4_payload(payload: dict[str, Any]) -> dict[str, Any]:
    results = [item for item in payload.get("results", []) if "score" in item]
    errors = [item for item in payload.get("results", []) if "error" in item]
    by_run: dict[str, list[dict[str, Any]]] = {}
    for entry in results:
        by_run.setdefault(str(entry["run_id"]), []).append(entry)

    run_summaries: dict[str, dict[str, Any]] = {}
    for run_id, entries in by_run.items():
        scores = [float(entry["score"]) for entry in entries]
        run_summaries[run_id] = {
            "episodes": len(entries),
            "mean_score": round(mean(scores), 3),
            "median_score": round(median(scores), 3),
            "std_score": round(_stddev(scores), 3),
            "min_score": round(min(scores), 3),
            "max_score": round(max(scores), 3),
            "api_cost_usd": round(sum(float(e.get("estimated_api_cost_usd", 0.0)) for e in entries), 6),
            "input_tokens": sum(int(e.get("token_usage", {}).get("input_tokens", 0)) for e in entries),
            "output_tokens": sum(int(e.get("token_usage", {}).get("output_tokens", 0)) for e in entries),
            "mean_budget_spent": round(mean(float(e.get("budget_spent", 0.0)) for e in entries), 3),
            "mean_experiment_count": round(mean(float(e.get("experiment_count", 0.0)) for e in entries), 3),
            "mean_intervention_prediction": round(
                mean(float(e.get("components", {}).get("heldout_intervention_prediction", 0.0)) for e in entries),
                3,
            ),
            "mean_intervention_qualitative_accuracy": round(
                mean(float(e.get("intervention_qualitative_accuracy", 0.0)) for e in entries),
                3,
            ),
            "components": _component_means(entries),
            "per_world_type": _per_world_type(entries),
            "confusion": _confusion(entries),
            "variance_by_world": _variance_by_world(entries),
            "common_experiment_sequences": _common_sequences(entries),
        }

    return {
        "completed_episodes": len(results),
        "errors": len(errors),
        "error_examples": errors[:5],
        "estimated_api_cost_usd": round(
            sum(float(e.get("estimated_api_cost_usd", 0.0)) for e in results),
            6,
        ),
        "runs": run_summaries,
    }


def _run_one_episode(
    registry: Registry,
    run: dict[str, Any],
    *,
    run_id: str,
    repeat_index: int,
    world_index: int,
    world_id: str,
    include_traces: bool,
) -> dict[str, Any]:
    world = registry.worlds[world_id]
    agent, backend = _make_agent(run, repeat_index=repeat_index, world_index=world_index)
    t0 = time.monotonic()
    try:
        result = registry.get_episode(world_id).run(agent)
    except Exception as exc:
        _close_backend(backend)
        return {
            **_run_metadata(run, run_id),
            "repeat": repeat_index,
            "world_index": world_index,
            "world_id": world_id,
            "causal_label": world.causal_label,
            "elapsed_seconds": round(time.monotonic() - t0, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }

    elapsed = time.monotonic() - t0
    usage = _usage_dict(backend)
    estimated_cost = _actual_usage_cost(run, usage)
    entry = _episode_entry(
        result,
        run=run,
        run_id=run_id,
        repeat_index=repeat_index,
        world_index=world_index,
        elapsed=elapsed,
        token_usage=usage,
        estimated_api_cost_usd=estimated_cost,
        include_traces=include_traces,
        agent=agent,
        registry=registry,
    )
    _close_backend(backend)
    return entry


def _episode_entry(
    result: EpisodeResult,
    *,
    run: dict[str, Any],
    run_id: str,
    repeat_index: int,
    world_index: int,
    elapsed: float,
    token_usage: dict[str, int],
    estimated_api_cost_usd: float,
    include_traces: bool,
    agent: Any,
    registry: Registry,
) -> dict[str, Any]:
    world = registry.worlds[result.world_id]
    predicted_label = max(
        result.report.normalized_probabilities(),
        key=result.report.normalized_probabilities().get,
    )
    entry: dict[str, Any] = {
        **_run_metadata(run, run_id),
        "repeat": repeat_index,
        "world_index": world_index,
        "world_id": result.world_id,
        "causal_label": world.causal_label,
        "predicted_label": predicted_label,
        "score": round(result.score.total, 4),
        "components": {key: round(value, 4) for key, value in result.score.components.items()},
        "budget_spent": round(_budget_spent(result), 4),
        "experiment_count": _experiment_count(result),
        "experiment_sequence": _experiment_sequence(result, agent),
        "intervention_qualitative_accuracy": round(
            _intervention_qualitative_accuracy(result, registry),
            4,
        ),
        "elapsed_seconds": round(elapsed, 3),
        "token_usage": token_usage,
        "estimated_api_cost_usd": round(estimated_api_cost_usd, 6),
    }
    if include_traces and hasattr(agent, "last_trace"):
        entry["trace"] = getattr(agent, "last_trace")
    return entry


def _make_agent(run: dict[str, Any], *, repeat_index: int, world_index: int):
    agent_name = str(run.get("agent", ""))
    seed = int(run.get("seed", 0)) + repeat_index * 1000 + world_index
    if agent_name == "oracle":
        return OracleAgent(), None
    if agent_name == "scripted":
        return ScriptedCausalControlAgent(), None
    if agent_name == "naive":
        return NaiveScoreSeekingAgent(), None
    if agent_name == "random":
        return RandomExperimentAgent(seed=seed), None
    if agent_name == "llm_mock":
        return LLMScaffoldAgent(max_experiments=int(run.get("max_experiments", 8))), None

    provider = str(run.get("provider", ""))
    model = run.get("model")
    backend_options = dict(run.get("backend_options", {}))
    if "max_tokens" in run:
        backend_options["max_tokens"] = int(run["max_tokens"])
    if "temperature" in run:
        backend_options["temperature"] = float(run["temperature"])
    backend = _make_backend(provider, model=None if model is None else str(model), run=run, options=backend_options)
    return LLMScaffoldAgent(
        backend=backend,
        max_experiments=int(run.get("max_experiments", 8)),
    ), backend


def _make_backend(provider: str, *, model: str | None, run: dict[str, Any], options: dict[str, Any]):
    if not provider:
        raise ValueError(f"paid LLM run {run.get('id')} requires provider")
    from agents.backends import make_backend

    api_key = None
    if run.get("api_key_env"):
        api_key = os.environ.get(str(run["api_key_env"]), "")
    if run.get("base_url"):
        options["base_url"] = str(run["base_url"])
    elif run.get("base_url_env") and os.environ.get(str(run["base_url_env"])):
        options["base_url"] = os.environ[str(run["base_url_env"])]
    return make_backend(provider, model=model, api_key=api_key, **options)


def _close_backend(backend: Any) -> None:
    if backend is not None and hasattr(backend, "close"):
        backend.close()


def _enabled_runs(pack: dict[str, Any], *, include_disabled: bool) -> list[dict[str, Any]]:
    runs = [dict(run) for run in pack.get("runs", [])]
    if not include_disabled:
        runs = [run for run in runs if bool(run.get("enabled", True))]
    if not runs:
        raise ValueError("study pack has no enabled runs")
    return runs


def _validate_runtime_credentials(runs: list[dict[str, Any]]) -> None:
    errors = []
    for run in runs:
        reason = _missing_credential_reason(run)
        if reason:
            errors.append(f"{run.get('id', run.get('model', run.get('agent')))}: {reason}")
    if errors:
        raise ValueError(
            "missing credentials for paid Phase 4 runs. "
            "Set the required environment variables, disable those runs, or pass "
            "--skip-missing-credentials. Missing: "
            + "; ".join(errors)
        )


def _drop_runs_missing_credentials(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    kept = []
    skipped = []
    for run in runs:
        reason = _missing_credential_reason(run)
        if reason:
            skipped.append({
                "run_id": str(run.get("id", run.get("model", run.get("agent", "")))),
                "reason": reason,
            })
        else:
            kept.append(run)
    if not kept:
        raise ValueError("all enabled Phase 4 runs were skipped because credentials were missing")
    return kept, skipped


def _missing_credential_reason(run: dict[str, Any]) -> str:
    if _is_zero_cost_run(run):
        return ""
    provider = str(run.get("provider", ""))
    api_key_env = str(run.get("api_key_env", ""))
    if api_key_env and not os.environ.get(api_key_env):
        return f"{api_key_env} is not set"
    if not api_key_env:
        default_env = {
            "openai": "OPENAI_API_KEY",
            "openai_compatible": "",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }.get(provider, "")
        if default_env and not os.environ.get(default_env):
            return f"{default_env} is not set"
    if provider == "openai_compatible":
        base_url_env = str(run.get("base_url_env", ""))
        if not run.get("base_url") and (not base_url_env or not os.environ.get(base_url_env)):
            return f"{base_url_env or 'base_url'} is not set"
    return ""


def _public_run_spec(run: dict[str, Any]) -> dict[str, Any]:
    safe = dict(run)
    safe.pop("api_key", None)
    safe.pop("api_key_env", None)
    return safe


def _run_metadata(run: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "agent": run.get("agent", ""),
        "provider": run.get("provider"),
        "model": run.get("model"),
        "tags": list(run.get("tags", [])),
    }


def _balanced_simulator_subset(registry: Registry, *, count: int) -> list[str]:
    ids = simulator_world_ids(registry, "compiled")
    by_label: dict[str, list[str]] = {label: [] for label in SIMULATOR_LABELS}
    for world_id in ids:
        label = registry.worlds[world_id].causal_label
        if label in by_label:
            by_label[label].append(world_id)
    selected: list[str] = []
    while len(selected) < count:
        changed = False
        for label in SIMULATOR_LABELS:
            candidates = by_label[label]
            index = len([wid for wid in selected if registry.worlds[wid].causal_label == label])
            if index < len(candidates) and len(selected) < count:
                selected.append(candidates[index])
                changed = True
        if not changed:
            break
    return selected


def _token_estimate(pack: dict[str, Any]) -> dict[str, int]:
    estimate = dict(pack.get("token_estimate", {}))
    return {
        "input_tokens_per_episode": int(
            estimate.get("input_tokens_per_episode", DEFAULT_INPUT_TOKENS_PER_EPISODE)
        ),
        "output_tokens_per_episode": int(
            estimate.get("output_tokens_per_episode", DEFAULT_OUTPUT_TOKENS_PER_EPISODE)
        ),
    }


def _estimate_run_cost(
    run: dict[str, Any],
    *,
    episodes: int,
    input_tokens_per_episode: int,
    output_tokens_per_episode: int,
) -> CostEstimate:
    run_id = str(run.get("id", run.get("agent", run.get("provider", "run"))))
    input_tokens = episodes * input_tokens_per_episode
    output_tokens = episodes * output_tokens_per_episode
    price = _price_for_run(run)
    if _is_zero_cost_run(run):
        return CostEstimate(run_id, episodes, 0, 0, 0.0, True)
    if price is None:
        return CostEstimate(run_id, episodes, input_tokens, output_tokens, 0.0, False)
    cost = (
        input_tokens / 1_000_000 * price["input"]
        + output_tokens / 1_000_000 * price["output"]
    )
    return CostEstimate(run_id, episodes, input_tokens, output_tokens, cost, True)


def _price_for_run(run: dict[str, Any]) -> dict[str, float] | None:
    if "price_per_million_tokens" in run:
        price = dict(run["price_per_million_tokens"])
        return {"input": float(price["input"]), "output": float(price["output"])}
    model = str(run.get("model", ""))
    if model in DEFAULT_MODEL_PRICES_USD_PER_MILLION:
        return DEFAULT_MODEL_PRICES_USD_PER_MILLION[model]
    return None


def _actual_usage_cost(run: dict[str, Any], usage: dict[str, int]) -> float:
    if _is_zero_cost_run(run) or not usage:
        return 0.0
    price = _price_for_run(run)
    if price is None:
        return 0.0
    return (
        int(usage.get("input_tokens", 0)) / 1_000_000 * price["input"]
        + int(usage.get("output_tokens", 0)) / 1_000_000 * price["output"]
    )


def _is_zero_cost_run(run: dict[str, Any]) -> bool:
    return str(run.get("agent", "")) in ZERO_COST_AGENTS and not run.get("provider")


def _enforce_preflight_cost(
    preflight: dict[str, Any],
    *,
    max_cost_usd: float | None,
    allow_unknown_cost: bool,
) -> None:
    unknown = list(preflight.get("unknown_price_runs", []))
    if unknown and not allow_unknown_cost:
        raise ValueError(
            "study pack contains paid runs with unknown pricing: "
            + ", ".join(unknown)
            + ". Add price_per_million_tokens or pass --allow-unknown-cost."
        )
    total = float(preflight.get("estimated_total_cost_usd", 0.0))
    if max_cost_usd is not None and total > float(max_cost_usd):
        raise ValueError(
            f"preflight estimated cost ${total:.4f} exceeds max_cost_usd ${float(max_cost_usd):.4f}"
        )


def _usage_dict(backend: Any) -> dict[str, int]:
    usage = getattr(backend, "usage", None) if backend is not None else None
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    return usage.to_dict()


def _budget_spent(result: EpisodeResult) -> float:
    return sum(
        float(record.get("cost", 0.0))
        for record in result.ledger
        if record.get("action_type") == "run_experiment"
    )


def _experiment_count(result: EpisodeResult) -> int:
    return sum(1 for record in result.ledger if record.get("action_type") == "run_experiment")


def _experiment_sequence(result: EpisodeResult, agent: Any) -> list[str]:
    trace = getattr(agent, "last_trace", None)
    if isinstance(trace, list):
        sequence = [
            str(item.get("request", {}).get("variant"))
            for item in trace
            if item.get("type") == "tool_call" and item.get("request", {}).get("variant")
        ]
        if sequence:
            return sequence
    sequence = []
    workspace = Path(result.workspace)
    for record in result.ledger:
        if record.get("action_type") != "run_experiment":
            continue
        experiment_id = record.get("experiment_id")
        if not experiment_id:
            continue
        request_path = workspace / "evidence" / str(experiment_id) / "request.json"
        if request_path.exists():
            try:
                data = json.loads(request_path.read_text(encoding="utf-8"))
                sequence.append(str(data.get("variant", experiment_id)))
                continue
            except json.JSONDecodeError:
                pass
        sequence.append(str(experiment_id))
    return sequence


def _intervention_qualitative_accuracy(result: EpisodeResult, registry: Registry) -> float:
    truth = {
        intervention.intervention_id: intervention.qualitative_result
        for intervention in registry.worlds[result.world_id].interventions
    }
    if not truth:
        return 0.0
    correct = 0
    count = 0
    for prediction in result.predictions:
        expected = truth.get(prediction.intervention_id)
        if expected is None:
            continue
        count += 1
        if prediction.qualitative_result == expected:
            correct += 1
    return correct / count if count else 0.0


def _component_means(entries: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for entry in entries for key in entry.get("components", {})})
    return {
        key: round(mean(float(entry.get("components", {}).get(key, 0.0)) for entry in entries), 3)
        for key in keys
    }


def _per_world_type(entries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_label: dict[str, list[float]] = {label: [] for label in WORLD_LABELS}
    for entry in entries:
        label = str(entry.get("causal_label", ""))
        if label in by_label:
            by_label[label].append(float(entry["score"]))
    return {
        label: {
            "episodes": len(scores),
            "mean_score": round(mean(scores), 3),
            "std_score": round(_stddev(scores), 3),
        }
        for label, scores in by_label.items()
        if scores
    }


def _confusion(entries: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {
        label: {predicted: 0 for predicted in WORLD_LABELS}
        for label in WORLD_LABELS
    }
    for entry in entries:
        label = str(entry.get("causal_label", ""))
        predicted = str(entry.get("predicted_label", ""))
        if label in matrix and predicted in matrix[label]:
            matrix[label][predicted] += 1
    return matrix


def _variance_by_world(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_world: dict[str, list[float]] = {}
    for entry in entries:
        by_world.setdefault(str(entry["world_id"]), []).append(float(entry["score"]))
    rows = []
    for world_id, scores in sorted(by_world.items()):
        if len(scores) <= 1:
            continue
        rows.append(
            {
                "world_id": world_id,
                "runs": len(scores),
                "mean_score": round(mean(scores), 3),
                "std_score": round(_stddev(scores), 3),
                "min_score": round(min(scores), 3),
                "max_score": round(max(scores), 3),
            }
        )
    return rows


def _common_sequences(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, ...], int] = {}
    for entry in entries:
        sequence = tuple(str(item) for item in entry.get("experiment_sequence", []))
        counts[sequence] = counts.get(sequence, 0) + 1
    rows = [
        {"count": count, "sequence": list(sequence)}
        for sequence, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return rows[:8]


def _stddev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))
