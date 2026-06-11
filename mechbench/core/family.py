"""Family and world abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mechbench.interface.schemas import (
    ActionSpec,
    BudgetConfig,
    ExperimentRequest,
    ExperimentResult,
    Headline,
    InterventionSpec,
)


@dataclass(frozen=True)
class World:
    """A frozen causal episode configuration discovered by the registry."""

    name: str
    family: str
    root: Path
    config: dict[str, Any]
    headline: Headline
    budget: BudgetConfig
    interventions: list[InterventionSpec]
    causal_label: str
    certificate: dict[str, Any] = field(default_factory=dict)

    @property
    def world_id(self) -> str:
        return str(self.config.get("world_id", f"{self.family}/{self.name}"))

    @property
    def public_id(self) -> str:
        return self.world_id


class FamilyInterface(ABC):
    """Interface implemented once per benchmark family."""

    def __init__(self, root: Path, config: dict[str, Any]):
        self.root = root
        self.config = config
        self.name = str(config.get("name", root.name))

    def public_paths(self) -> list[str]:
        """Family paths copied into an episode workspace."""

        return ["base_model", "modification", "training", "evaluation", "actions"]

    def setup_workspace(self, world: World, workspace_root: Path) -> None:
        """Optional family hook for adding public files to the workspace."""

    @abstractmethod
    def get_available_actions(self, world: World) -> list[ActionSpec]:
        """Return experiment actions available to the agent."""

    def estimate_cost(self, world: World, request: ExperimentRequest) -> float:
        for action in self.get_available_actions(world):
            if action.name == request.variant:
                return action.default_cost
        return 1.0

    @abstractmethod
    def run_experiment(self, world: World, request: ExperimentRequest) -> ExperimentResult:
        """Execute one controlled experiment."""

    def analyze_artifact(self, world: World, experiment_id: str, analysis_spec: dict[str, Any]) -> dict[str, Any]:
        """Return cached artifact analysis. Families can override this."""

        return {
            "experiment_id": experiment_id,
            "analysis_spec": analysis_spec,
            "status": "not_available",
        }


class TableFamily(FamilyInterface):
    """Simple family backed by precomputed experiment tables.

    This is useful for tests and for the upcoming factorized simulator adapter.
    """

    def get_available_actions(self, world: World) -> list[ActionSpec]:
        actions = self.config.get("actions", [])
        if actions:
            return [ActionSpec.from_dict(item) for item in actions]
        table = world.config.get("experiment_table", {})
        return [ActionSpec(name=name, description=f"Run {name}") for name in table]

    def run_experiment(self, world: World, request: ExperimentRequest) -> ExperimentResult:
        table = world.config.get("experiment_table", {})
        if request.variant not in table:
            raise ValueError(f"unknown experiment variant: {request.variant}")
        row = dict(table[request.variant])
        return ExperimentResult.from_dict(
            {
                "experiment_id": "",
                "status": row.get("status", "completed"),
                "cost": row.get("cost", self.estimate_cost(world, request)),
                "config_hash": "",
                "model_stats": row.get("model_stats", {}),
                "metrics": row.get("metrics", {}),
                "artifacts": row.get("artifacts", {}),
            }
        )

