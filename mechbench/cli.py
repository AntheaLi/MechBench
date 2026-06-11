"""Command-line entry point for the generic harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agents.naive import NaiveScoreSeekingAgent
from agents.llm_scaffold import LLMScaffoldAgent
from agents.oracle import OracleAgent
from agents.random_policy import RandomExperimentAgent
from agents.scripted_causal_control import ScriptedCausalControlAgent
from analysis.reporting import write_calibration_report
from mechbench.registry import Registry
from mechbench.simulator.calibration import calibrate_simulator
from mechbench.simulator.compiler import SimulatorEpisodeCompiler
from mechbench.utils.config import dump_json

BUILTIN_AGENTS = [
    "oracle",
    "naive",
    "random",
    "scripted",
    "llm_mock",
    "llm_anthropic",
    "llm_openai",
    "llm_gemini",
]


def _agent_from_name(name: str, *, model: str | None = None):
    if name == "oracle":
        return OracleAgent()
    if name == "naive":
        return NaiveScoreSeekingAgent()
    if name == "random":
        return RandomExperimentAgent()
    if name == "scripted":
        return ScriptedCausalControlAgent()
    if name == "llm_mock":
        return LLMScaffoldAgent()
    if name == "llm_anthropic":
        from agents.backends import AnthropicBackend
        backend = AnthropicBackend(model=model or "claude-sonnet-4-20250514")
        return LLMScaffoldAgent(backend=backend)
    if name == "llm_openai":
        from agents.backends import OpenAIBackend
        backend = OpenAIBackend(model=model or "gpt-4o")
        return LLMScaffoldAgent(backend=backend)
    if name == "llm_gemini":
        from agents.backends import GeminiBackend
        backend = GeminiBackend(model=model or "gemini-3.5-pro")
        return LLMScaffoldAgent(backend=backend)
    raise ValueError(f"unknown built-in agent: {name}")


def _resolve_worlds(registry: Registry, value: str) -> list[str]:
    if value == "all":
        return registry.list_worlds()
    return [item.strip() for item in value.split(",") if item.strip()]


def cmd_list_worlds(args: argparse.Namespace) -> int:
    registry = Registry(args.families_dir)
    for world_id in registry.list_worlds(args.family):
        print(world_id)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    registry = Registry(args.families_dir)
    agent = _agent_from_name(args.agent, model=getattr(args, "model", None))
    results: list[dict[str, Any]] = []
    for world_id in _resolve_worlds(registry, args.worlds):
        if world_id not in registry.worlds:
            raise SystemExit(f"unknown world: {world_id}")
        result = registry.get_episode(world_id).run(agent)
        payload = result.to_dict()
        trace = getattr(agent, "last_trace", None)
        if trace is not None:
            payload["agent_trace"] = trace
        results.append(payload)
        print(f"{world_id}: {result.score.total:.2f}")

    payload = {"results": results}
    if args.output:
        dump_json(Path(args.output), payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    registry = Registry(args.families_dir)
    if args.family:
        world_ids = registry.list_worlds(args.target)
    elif "/" not in args.target:
        world_ids = registry.list_worlds(args.target)
    else:
        world_ids = [args.target]
    if not world_ids:
        raise SystemExit(f"no worlds found for {args.target}")
    for world_id in world_ids:
        world = registry.worlds[world_id]
        checks = {
            "causal_label": bool(world.causal_label),
            "headline": bool(world.headline.baseline_metrics and world.headline.proposed_metrics),
            "budget_positive": world.budget.max_cost > 0,
            "interventions": len(world.interventions) >= 1,
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        print(f"{status} {world_id} {checks}")
    return 0


def cmd_new_family(args: argparse.Namespace) -> int:
    root = Path(args.families_dir) / args.name
    (root / "worlds").mkdir(parents=True, exist_ok=True)
    for subdir in ["base_model", "modification", "training", "evaluation", "actions"]:
        (root / subdir).mkdir(exist_ok=True)
    dump_json(
        root / "family.yaml",
        {
            "name": args.name,
            "description": "New MechanismBench family.",
            "actions": [
                {"name": "baseline", "description": "Run baseline control.", "default_cost": 1.0},
                {"name": "proposed", "description": "Run proposed modification.", "default_cost": 1.0},
            ],
        },
    )
    (root / "family.py").write_text(
        "from mechbench.core.family import TableFamily\n\n"
        "class Family(TableFamily):\n"
        "    pass\n",
        encoding="utf-8",
    )
    print(f"created {root}")
    return 0


def cmd_new_world(args: argparse.Namespace) -> int:
    family_name, world_name = args.name.split("/", 1)
    root = Path(args.families_dir) / family_name / "worlds" / world_name
    root.mkdir(parents=True, exist_ok=True)
    dump_json(
        root / "world.yaml",
        {
            "name": world_name,
            "world_id": f"{family_name}/{world_name}",
            "family": family_name,
            "causal_label": "true_mechanism",
            "headline": {
                "baseline_metrics": {"validation_id.accuracy": 0.70},
                "proposed_metrics": {"validation_id.accuracy": 0.73},
                "description": "The proposed modification improves validation accuracy.",
            },
            "budget": {"max_cost": 10.0},
            "improvement_truth": {
                "status": "reproducible_positive_gain",
                "delta": 0.03,
            },
            "interventions": [
                {
                    "id": "hidden_1",
                    "description": "Private confound-breaking intervention.",
                    "expected_delta": 0.03,
                    "qualitative_result": "gain_survives",
                }
            ],
            "experiment_table": {},
            "certificate": {
                "oracle_minimum_cost": 2.0,
                "exhaustive_cost": 8.0,
                "prediction_tolerance": 0.05,
            },
        },
    )
    print(f"created {root}")
    return 0


def cmd_compile_simulator(args: argparse.Namespace) -> int:
    compiler = SimulatorEpisodeCompiler(
        families_dir=args.families_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    result = compiler.compile(count=args.count, start_index=args.start_index, clear=args.clear)
    payload = result.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_calibrate_simulator(args: argparse.Namespace) -> int:
    agent_names = [item.strip() for item in args.agents.split(",") if item.strip()]
    result = calibrate_simulator(
        families_dir=args.families_dir,
        selector=args.worlds,
        agent_names=agent_names,
    )
    payload = result.to_dict()
    if args.output:
        dump_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_llm_study(args: argparse.Namespace) -> int:
    """Run a small LLM-agent study on a diverse subset of worlds."""
    import time as _time

    registry = Registry(args.families_dir)
    agent = _agent_from_name(args.agent, model=args.model)

    # Select worlds: use explicit list or pick a diverse subset
    if args.worlds:
        world_ids = _resolve_worlds(registry, args.worlds)
    else:
        world_ids = _select_study_subset(registry, count=args.count)

    print(f"Running {args.agent} on {len(world_ids)} worlds...")
    results: list[dict[str, Any]] = []
    for idx, world_id in enumerate(world_ids):
        if world_id not in registry.worlds:
            print(f"  SKIP {world_id} (not found)")
            continue
        t0 = _time.monotonic()
        try:
            result = registry.get_episode(world_id).run(agent)
        except Exception as exc:
            print(f"  [{idx + 1}/{len(world_ids)}] {world_id}: ERROR {exc}")
            results.append({"world_id": world_id, "error": str(exc)})
            continue
        elapsed = _time.monotonic() - t0
        score = result.score.total
        label = registry.worlds[world_id].causal_label
        print(f"  [{idx + 1}/{len(world_ids)}] {world_id} ({label}): {score:.1f}  [{elapsed:.1f}s]")

        entry: dict[str, Any] = {
            "world_id": world_id,
            "causal_label": label,
            "score": round(score, 2),
            "components": {k: round(v, 2) for k, v in result.score.components.items()},
            "elapsed_seconds": round(elapsed, 1),
        }
        trace = getattr(agent, "last_trace", None)
        if trace is not None:
            entry["trace"] = trace
        # Capture token usage if the backend tracks it
        backend = getattr(agent, "backend", None)
        usage = getattr(backend, "usage", None)
        if usage is not None:
            entry["cumulative_usage"] = usage.to_dict()
        results.append(entry)

    # Summary
    scores = [r["score"] for r in results if "score" in r]
    if scores:
        from statistics import mean, median
        print(f"\n--- Study Summary ---")
        print(f"  worlds: {len(scores)}")
        print(f"  mean:   {mean(scores):.1f}")
        print(f"  median: {median(scores):.1f}")
        print(f"  min:    {min(scores):.1f}")
        print(f"  max:    {max(scores):.1f}")

    # Token usage
    backend = getattr(agent, "backend", None)
    usage = getattr(backend, "usage", None)
    if usage is not None:
        u = usage.to_dict()
        print(f"  total tokens: {u['input_tokens']} in + {u['output_tokens']} out ({u['calls']} calls)")

    payload = {"agent": args.agent, "model": args.model, "results": results}
    if args.output:
        dump_json(Path(args.output), payload)
        print(f"\nResults saved to {args.output}")
    return 0


def _select_study_subset(registry: Registry, count: int = 10) -> list[str]:
    """Pick a diverse subset of simulator worlds — 2 per causal label."""
    from mechbench.simulator.calibration import SIMULATOR_LABELS, simulator_world_ids

    world_ids = simulator_world_ids(registry, "compiled")
    by_label: dict[str, list[str]] = {label: [] for label in SIMULATOR_LABELS}
    for wid in world_ids:
        label = registry.worlds[wid].causal_label
        if label in by_label:
            by_label[label].append(wid)
    selected: list[str] = []
    per_label = max(1, count // len(SIMULATOR_LABELS))
    for label in SIMULATOR_LABELS:
        candidates = by_label[label]
        selected.extend(candidates[:per_label])
    return selected[:count]


def cmd_report_calibration(args: argparse.Namespace) -> int:
    write_calibration_report(args.input, args.output, title=args.title)
    print(f"wrote {args.output}")
    return 0


def cmd_phase4_study(args: argparse.Namespace) -> int:
    from mechbench.phase4.reporting import render_phase4_report
    from mechbench.phase4.study import list_study_packs, run_phase4_study

    if args.list_packs:
        for pack in list_study_packs():
            print(f"{pack['name']}: {pack['description']}  [{pack['path']}]")
        return 0

    try:
        payload = run_phase4_study(
            args.pack,
            families_dir=args.families_dir,
            output_dir=args.output_dir if not args.dry_run else None,
            dry_run=args.dry_run,
            include_disabled=args.include_disabled,
            worlds_override=args.worlds,
            count_override=args.count,
            repeats_override=args.repeats,
            max_cost_usd_override=args.max_cost_usd,
            allow_unknown_cost=args.allow_unknown_cost,
            skip_missing_credentials=args.skip_missing_credentials,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    report = render_phase4_report(payload)
    if args.dry_run:
        print(json.dumps(payload["preflight"], indent=2, sort_keys=True))
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(output_dir / "study.json", payload)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"wrote {output_dir / 'study.json'}")
    print(f"wrote {output_dir / 'report.md'}")
    return 0


def cmd_phase4_report(args: argparse.Namespace) -> int:
    from mechbench.phase4.reporting import write_phase4_report

    write_phase4_report(args.input, args.output)
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mechbench")
    parser.add_argument("--families-dir", default="families")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-worlds")
    list_parser.add_argument("--family")
    list_parser.set_defaults(func=cmd_list_worlds)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--agent", choices=BUILTIN_AGENTS, default="naive")
    evaluate_parser.add_argument("--model", default=None, help="Model override for LLM backends")
    evaluate_parser.add_argument("--worlds", default="all")
    evaluate_parser.add_argument("--output")
    evaluate_parser.set_defaults(func=cmd_evaluate)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("target")
    verify_parser.add_argument("--family", action="store_true")
    verify_parser.set_defaults(func=cmd_verify)

    new_family_parser = subparsers.add_parser("new-family")
    new_family_parser.add_argument("name")
    new_family_parser.set_defaults(func=cmd_new_family)

    new_world_parser = subparsers.add_parser("new-world")
    new_world_parser.add_argument("name")
    new_world_parser.set_defaults(func=cmd_new_world)

    compile_parser = subparsers.add_parser("compile-simulator")
    compile_parser.add_argument("--count", type=int, default=100)
    compile_parser.add_argument("--seed", type=int, default=0)
    compile_parser.add_argument("--start-index", type=int, default=1)
    compile_parser.add_argument("--output-dir")
    compile_parser.add_argument("--clear", action="store_true")
    compile_parser.set_defaults(func=cmd_compile_simulator)

    calibrate_parser = subparsers.add_parser("calibrate-simulator")
    calibrate_parser.add_argument("--worlds", default="compiled")
    calibrate_parser.add_argument("--agents", default="oracle,scripted,naive,random")
    calibrate_parser.add_argument("--output")
    calibrate_parser.set_defaults(func=cmd_calibrate_simulator)

    study_parser = subparsers.add_parser("llm-study", help="Run a small LLM-agent study")
    study_parser.add_argument("--agent", choices=BUILTIN_AGENTS, default="llm_anthropic")
    study_parser.add_argument("--model", default=None, help="Model override for the LLM backend")
    study_parser.add_argument("--worlds", default=None, help="Comma-separated world IDs or 'all'")
    study_parser.add_argument("--count", type=int, default=10, help="Number of worlds when auto-selecting")
    study_parser.add_argument("--output", help="Path to save JSON results")
    study_parser.set_defaults(func=cmd_llm_study)

    phase4_parser = subparsers.add_parser("phase4-study", help="Run a packaged Phase 4 agent study")
    phase4_parser.add_argument("--pack", default="phase4_free_smoke", help="Study pack name or JSON path")
    phase4_parser.add_argument("--list-packs", action="store_true", help="List bundled study packs")
    phase4_parser.add_argument("--output-dir", default="results/phase4_run", help="Directory for study.json and report.md")
    phase4_parser.add_argument("--dry-run", action="store_true", help="Resolve worlds/runs and estimate cost without API calls")
    phase4_parser.add_argument("--include-disabled", action="store_true", help="Run disabled/template entries in a pack")
    phase4_parser.add_argument("--worlds", default=None, help="Comma-separated world IDs or 'all' override")
    phase4_parser.add_argument("--count", type=int, default=None, help="Override world count for selectors that support it")
    phase4_parser.add_argument("--repeats", type=int, default=None, help="Override repeat count")
    phase4_parser.add_argument("--max-cost-usd", type=float, default=None, help="Override pack max API spend")
    phase4_parser.add_argument("--allow-unknown-cost", action="store_true", help="Allow paid runs without known price estimates")
    phase4_parser.add_argument("--skip-missing-credentials", action="store_true", help="Skip paid runs whose API key/base URL env vars are missing")
    phase4_parser.set_defaults(func=cmd_phase4_study)

    phase4_report_parser = subparsers.add_parser("phase4-report", help="Render a Phase 4 JSON study report")
    phase4_report_parser.add_argument("--input", default="results/phase4_run/study.json")
    phase4_report_parser.add_argument("--output", default="results/phase4_run/report.md")
    phase4_report_parser.set_defaults(func=cmd_phase4_report)

    report_parser = subparsers.add_parser("report-calibration")
    report_parser.add_argument("--input", default="results/simulator_calibration.json")
    report_parser.add_argument("--output", default="results/simulator_calibration_report.md")
    report_parser.add_argument("--title", default="MechanismBench Calibration Report")
    report_parser.set_defaults(func=cmd_report_calibration)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
