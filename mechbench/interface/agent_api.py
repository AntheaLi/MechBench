"""Controlled API surface exposed to agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mechbench.core.budget import Budget
from mechbench.core.evidence_ledger import EvidenceLedger
from mechbench.core.experiment_runner import ExperimentRunner
from mechbench.core.family import FamilyInterface, World
from mechbench.core.workspace import Workspace
from mechbench.interface.actions import make_experiment_request
from mechbench.interface.schemas import (
    FinalReport,
    InterventionPrediction,
    to_plain_dict,
)


class ReportLocked(RuntimeError):
    """Raised when an agent tries to continue experimenting after submission."""


@dataclass
class AgentAPI:
    workspace: Workspace
    family: FamilyInterface
    world: World
    ledger: EvidenceLedger
    budget: Budget

    def __post_init__(self) -> None:
        self.runner = ExperimentRunner(self.family, self.world, self.budget)
        self._report: FinalReport | None = None
        self._predictions: list[InterventionPrediction] | None = None

    @property
    def episode_id(self) -> str:
        return self.world.world_id

    @property
    def headline(self) -> dict[str, Any]:
        return self.world.headline.to_dict()

    @property
    def description(self) -> str:
        return self.world.headline.description

    @property
    def budget_remaining(self) -> float:
        return self.budget.remaining

    @property
    def submitted_report(self) -> FinalReport | None:
        return self._report

    @property
    def submitted_predictions(self) -> list[InterventionPrediction] | None:
        return self._predictions

    def inspect(self, path: str) -> str:
        content = self.workspace.inspect(path)
        self.ledger.record_inspect(path)
        return content

    def list_files(self) -> list[str]:
        return self.workspace.list_files()

    def list_experiments(self) -> list[dict[str, Any]]:
        return [action.to_dict() for action in self.family.get_available_actions(self.world)]

    def list_actions(self) -> list[dict[str, Any]]:
        return self.list_experiments()

    def run_experiment(self, config: dict[str, Any]) -> dict[str, Any]:
        if self._report is not None or self._predictions is not None:
            raise ReportLocked("experiments are locked after report or prediction submission")
        request = make_experiment_request(config)
        result = self.runner.run(request)
        self.ledger.record_experiment(request, result, self.budget.remaining)
        return result.to_dict()

    def analyze_artifact(self, experiment_id: str, analysis_spec: dict[str, Any]) -> dict[str, Any]:
        result = self.family.analyze_artifact(self.world, experiment_id, analysis_spec)
        self.ledger.record_analysis(experiment_id, analysis_spec, result)
        return result

    def analyze(self, experiment_id: str, analysis_spec: dict[str, Any]) -> dict[str, Any]:
        return self.analyze_artifact(experiment_id, analysis_spec)

    def get_interventions(self) -> list[dict[str, Any]]:
        return [intervention.public_dict() for intervention in self.world.interventions]

    def submit_predictions(self, predictions: list[dict[str, Any] | InterventionPrediction]) -> list[dict[str, Any]]:
        parsed = [
            item if isinstance(item, InterventionPrediction) else InterventionPrediction.from_dict(item)
            for item in predictions
        ]
        self._predictions = parsed
        payload = [prediction.to_dict() for prediction in parsed]
        self.ledger.record_submission("submit_intervention_predictions", payload)
        return payload

    def submit_report(self, report: dict[str, Any] | FinalReport) -> dict[str, Any]:
        parsed = report if isinstance(report, FinalReport) else FinalReport.from_dict(report)
        self._report = parsed
        payload = parsed.to_dict()
        self.ledger.record_submission("submit_report", payload)
        return payload

    def evidence_summary(self) -> dict[str, Any]:
        records = self.ledger.export()
        return {
            "budget": self.budget.to_dict(),
            "experiment_ids": sorted(self.ledger.experiment_ids()),
            "actions": [to_plain_dict(record) for record in records],
        }

