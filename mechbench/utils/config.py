"""Config loading helpers.

The repository uses `.yaml` names for contributor-facing files, but the generic
bootstrap keeps dependencies optional. JSON-compatible YAML is supported through
the standard library; full YAML is accepted when PyYAML is installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a config file cannot be parsed."""


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as json_error:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as yaml_error:
            raise ConfigError(
                f"{path} is not JSON-compatible and PyYAML is not installed"
            ) from yaml_error
        try:
            data = yaml.safe_load(text)
        except Exception as yaml_parse_error:  # pragma: no cover - depends on PyYAML
            raise ConfigError(f"Could not parse {path}: {yaml_parse_error}") from json_error

    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain an object at the top level")
    return data


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

