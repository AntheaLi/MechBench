"""Markdown reporting for Phase 4 study outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_phase4_report(input_path: str | Path, output_path: str | Path) -> str:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    markdown = render_phase4_report(payload)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(markdown, encoding="utf-8")
    return markdown


def render_phase4_report(payload: dict[str, Any]) -> str:
    pack = payload.get("pack", {})
    resolved = payload.get("resolved", {})
    preflight = payload.get("preflight", {})
    summary = payload.get("summary", {})
    lines: list[str] = []
    title = str(pack.get("name", "Phase 4 Study"))
    lines.append(f"# {title} Report")
    lines.append("")
    if pack.get("description"):
        lines.append(str(pack["description"]))
        lines.append("")
    lines.append("## Run Overview")
    lines.append("")
    lines.append(f"- Created: `{payload.get('created_at', '')}`")
    lines.append(f"- Dry run: `{bool(payload.get('dry_run', False))}`")
    lines.append(f"- Worlds: `{len(resolved.get('world_ids', []))}`")
    lines.append(f"- Repeats: `{resolved.get('repeats', 1)}`")
    lines.append(f"- Completed episodes: `{summary.get('completed_episodes', 0)}`")
    lines.append(f"- Errors: `{summary.get('errors', 0)}`")
    lines.append(f"- Estimated API cost used: `${float(summary.get('estimated_api_cost_usd', 0.0)):.4f}`")
    if payload.get("stopped_reason"):
        lines.append(f"- Stopped reason: `{payload['stopped_reason']}`")
    lines.append("")

    lines.append("## Preflight Cost")
    lines.append("")
    lines.append(
        f"Estimate before execution: `${float(preflight.get('estimated_total_cost_usd', 0.0)):.4f}` "
        f"for `{preflight.get('episodes_per_run', 0)}` episodes per run."
    )
    lines.append("")
    lines.append("| Run | Input Tokens | Output Tokens | Estimated Cost | Price Known |")
    lines.append("|---|---:|---:|---:|---|")
    for row in preflight.get("runs", []):
        lines.append(
            f"| {row.get('run_id', '')} | {int(row.get('input_tokens', 0))} | "
            f"{int(row.get('output_tokens', 0))} | "
            f"${float(row.get('cost_usd', 0.0)):.4f} | {bool(row.get('known_price', False))} |"
        )
    lines.append("")

    runs = summary.get("runs", {})
    lines.append("## Agent Scores")
    lines.append("")
    if runs:
        lines.append(
            "| Run | Episodes | Mean | Median | Std | Min | Max | API Cost | Input Tok | Output Tok |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for run_id, row in runs.items():
            lines.append(
                f"| {run_id} | {row.get('episodes', 0)} | "
                f"{float(row.get('mean_score', 0.0)):.2f} | "
                f"{float(row.get('median_score', 0.0)):.2f} | "
                f"{float(row.get('std_score', 0.0)):.2f} | "
                f"{float(row.get('min_score', 0.0)):.2f} | "
                f"{float(row.get('max_score', 0.0)):.2f} | "
                f"${float(row.get('api_cost_usd', 0.0)):.4f} | "
                f"{int(row.get('input_tokens', 0))} | {int(row.get('output_tokens', 0))} |"
            )
    else:
        lines.append("No completed episodes yet.")
    lines.append("")

    lines.append("## Component Scores")
    lines.append("")
    component_keys = sorted(
        {
            component
            for row in runs.values()
            for component in row.get("components", {})
        }
    )
    if component_keys:
        lines.append("| Run | " + " | ".join(component_keys) + " |")
        lines.append("|---|" + "|".join("---:" for _ in component_keys) + "|")
        for run_id, row in runs.items():
            values = [
                f"{float(row.get('components', {}).get(component, 0.0)):.2f}"
                for component in component_keys
            ]
            lines.append(f"| {run_id} | " + " | ".join(values) + " |")
    else:
        lines.append("No component scores available.")
    lines.append("")

    lines.append("## World-Type Breakdown")
    lines.append("")
    for run_id, row in runs.items():
        lines.append(f"### {run_id}")
        lines.append("")
        lines.append("| World Type | Episodes | Mean | Std |")
        lines.append("|---|---:|---:|---:|")
        for label, label_row in row.get("per_world_type", {}).items():
            lines.append(
                f"| {label} | {label_row.get('episodes', 0)} | "
                f"{float(label_row.get('mean_score', 0.0)):.2f} | "
                f"{float(label_row.get('std_score', 0.0)):.2f} |"
            )
        lines.append("")

    lines.append("## Confusion Matrices")
    lines.append("")
    for run_id, row in runs.items():
        lines.append(f"### {run_id}")
        lines.append("")
        labels = list(row.get("confusion", {}).keys())
        if not labels:
            lines.append("No confusion data.")
            lines.append("")
            continue
        predictions = sorted({pred for vals in row["confusion"].values() for pred in vals})
        lines.append("| True \\ Pred | " + " | ".join(predictions) + " |")
        lines.append("|---|" + "|".join("---:" for _ in predictions) + "|")
        for label in labels:
            values = [str(row["confusion"].get(label, {}).get(pred, 0)) for pred in predictions]
            lines.append(f"| {label} | " + " | ".join(values) + " |")
        lines.append("")

    lines.append("## Intervention Prediction")
    lines.append("")
    lines.append("| Run | Heldout Component Mean | Qualitative Accuracy |")
    lines.append("|---|---:|---:|")
    for run_id, row in runs.items():
        lines.append(
            f"| {run_id} | {float(row.get('mean_intervention_prediction', 0.0)):.2f} | "
            f"{float(row.get('mean_intervention_qualitative_accuracy', 0.0)):.2f} |"
        )
    lines.append("")

    lines.append("## Budget Use")
    lines.append("")
    lines.append("| Run | Mean Experiment Cost | Mean Experiment Count |")
    lines.append("|---|---:|---:|")
    for run_id, row in runs.items():
        lines.append(
            f"| {run_id} | {float(row.get('mean_budget_spent', 0.0)):.2f} | "
            f"{float(row.get('mean_experiment_count', 0.0)):.2f} |"
        )
    lines.append("")

    lines.append("## Experiment Sequences")
    lines.append("")
    for run_id, row in runs.items():
        lines.append(f"### {run_id}")
        lines.append("")
        sequences = row.get("common_experiment_sequences", [])
        if not sequences:
            lines.append("No experiment sequences recorded.")
            lines.append("")
            continue
        lines.append("| Count | Sequence |")
        lines.append("|---:|---|")
        for seq in sequences:
            lines.append(f"| {seq.get('count', 0)} | `{' -> '.join(seq.get('sequence', []))}` |")
        lines.append("")

    lines.append("## Repeated-Run Variance")
    lines.append("")
    any_variance = False
    for run_id, row in runs.items():
        variance = row.get("variance_by_world", [])
        if not variance:
            continue
        any_variance = True
        lines.append(f"### {run_id}")
        lines.append("")
        lines.append("| World | Runs | Mean | Std | Min | Max |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for item in variance[:20]:
            lines.append(
                f"| {item.get('world_id', '')} | {item.get('runs', 0)} | "
                f"{float(item.get('mean_score', 0.0)):.2f} | "
                f"{float(item.get('std_score', 0.0)):.2f} | "
                f"{float(item.get('min_score', 0.0)):.2f} | "
                f"{float(item.get('max_score', 0.0)):.2f} |"
            )
        lines.append("")
    if not any_variance:
        lines.append("No repeated-run variance data in this payload.")
        lines.append("")

    if summary.get("error_examples"):
        lines.append("## Error Examples")
        lines.append("")
        for error in summary["error_examples"]:
            lines.append(
                f"- `{error.get('run_id', '')}` on `{error.get('world_id', '')}`: "
                f"{error.get('error', '')}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
