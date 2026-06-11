"""Thin action helpers for agent adapters."""

from __future__ import annotations

from typing import Any

from mechbench.interface.schemas import ExperimentRequest


def make_experiment_request(config: ExperimentRequest | dict[str, Any]) -> ExperimentRequest:
    if isinstance(config, ExperimentRequest):
        return config
    if not isinstance(config, dict):
        raise TypeError("experiment config must be a dict or ExperimentRequest")
    return ExperimentRequest.from_dict(config)

