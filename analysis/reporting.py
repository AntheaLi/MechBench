"""Markdown reporting helpers for MechanismBench result payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_calibration_report(payload: dict[str, Any], title: str = "MechanismBench Calibration Report") -> str:
    """Render a calibration payload as a human-readable Markdown report."""

    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- World count: {payload.get('world_count', 'n/a')}",
        f"- Public-only world classifier accuracy: {_fmt(payload.get('public_only_world_classification_accuracy'))}",
        f"- Gates passing: {_gate_count(payload.get('gates', {}))}",
        "",
        "## Gates",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    for gate, passed in sorted(payload.get("gates", {}).items()):
        lines.append(f"| `{gate}` | {'PASS' if passed else 'FAIL'} |")

    lines.extend(["", "## Agent Scores", "", "| Agent | Mean | Median | Min | Max |", "|---|---:|---:|---:|---:|"])
    for agent, summary in sorted(payload.get("agents", {}).items()):
        lines.append(
            f"| {agent} | {_fmt(summary.get('mean'))} | {_fmt(summary.get('median'))} | "
            f"{_fmt(summary.get('min'))} | {_fmt(summary.get('max'))} |"
        )

    per_component = payload.get("per_component", {})
    if per_component:
        components = sorted({component for values in per_component.values() for component in values})
        lines.extend(["", "## Components", "", _component_header(components), _component_rule(components)])
        for agent, values in sorted(per_component.items()):
            cells = " | ".join(_fmt(values.get(component)) for component in components)
            lines.append(f"| {agent} | {cells} |")

    per_world = payload.get("per_world_type", {})
    if per_world:
        lines.extend(["", "## World Types", ""])
        for agent, values in sorted(per_world.items()):
            lines.extend([f"### {agent}", "", "| World Type | Mean | Median | Min | Max |", "|---|---:|---:|---:|---:|"])
            for world_type, summary in sorted(values.items()):
                lines.append(
                    f"| {world_type} | {_fmt(summary.get('mean'))} | {_fmt(summary.get('median'))} | "
                    f"{_fmt(summary.get('min'))} | {_fmt(summary.get('max'))} |"
                )
            lines.append("")

    weakest = _weakest_episodes(payload.get("per_episode", []))
    if weakest:
        agent_names = sorted({
            key for row in weakest for key, value in row.items()
            if isinstance(value, (int, float)) and key not in {"world_count"}
        })
        lines.extend(["## Weakest Episodes", "", _episode_header(agent_names), _episode_rule(agent_names)])
        for row in weakest:
            cells = " | ".join(_fmt(row.get(agent)) for agent in agent_names)
            lines.append(f"| {row.get('world_id')} | {row.get('causal_label')} | {cells} |")

    return "\n".join(lines).rstrip() + "\n"


def write_calibration_report(
    input_path: str | Path,
    output_path: str | Path,
    title: str = "MechanismBench Calibration Report",
) -> str:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    markdown = render_calibration_report(payload, title=title)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(markdown, encoding="utf-8")
    return markdown


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _gate_count(gates: dict[str, Any]) -> str:
    if not gates:
        return "n/a"
    passed = sum(1 for value in gates.values() if value)
    return f"{passed}/{len(gates)}"


def _component_header(components: list[str]) -> str:
    return "| Agent | " + " | ".join(components) + " |"


def _component_rule(components: list[str]) -> str:
    return "|---|" + "|".join("---:" for _ in components) + "|"


def _weakest_episodes(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    def minimum_score(row: dict[str, Any]) -> float:
        scores = [value for value in row.values() if isinstance(value, (int, float))]
        return min(scores) if scores else 0.0

    return sorted(rows, key=minimum_score)[:limit]


def _episode_header(agent_names: list[str]) -> str:
    return "| World | Label | " + " | ".join(agent_names) + " |"


def _episode_rule(agent_names: list[str]) -> str:
    return "|---|---|" + "|".join("---:" for _ in agent_names) + "|"
