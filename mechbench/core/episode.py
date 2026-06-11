"""Episode lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mechbench.core.budget import Budget
from mechbench.core.evidence_ledger import EvidenceLedger
from mechbench.core.family import FamilyInterface, World
from mechbench.core.workspace import Workspace
from mechbench.interface.agent_api import AgentAPI
from mechbench.interface.schemas import EpisodeResult, FinalReport, InterventionPrediction
from mechbench.scoring import score_episode


class AgentLike(Protocol):
    def investigate(self, api: AgentAPI) -> tuple[FinalReport, list[InterventionPrediction]]:
        ...


@dataclass
class Episode:
    family: FamilyInterface
    world: World
    workspace_root: Path | None = None

    def run(self, agent: AgentLike) -> EpisodeResult:
        workspace = Workspace(self.family, self.world, self.workspace_root).create()
        ledger = EvidenceLedger(workspace.root / "evidence", self.world.world_id)  # type: ignore[operator]
        budget = Budget(self.world.budget)
        api = AgentAPI(
            workspace=workspace,
            family=self.family,
            world=self.world,
            ledger=ledger,
            budget=budget,
        )

        returned_report, returned_predictions = agent.investigate(api)
        predictions = api.submitted_predictions or returned_predictions
        report = api.submitted_report or returned_report

        if api.submitted_predictions is None:
            api.submit_predictions(predictions)
        if api.submitted_report is None:
            api.submit_report(report)

        score = score_episode(self.world, report, predictions, ledger)
        return EpisodeResult(
            episode_id=self.world.world_id,
            world_id=self.world.world_id,
            report=report,
            predictions=predictions,
            score=score,
            ledger=ledger.export(),
            workspace=str(workspace.root),
        )

