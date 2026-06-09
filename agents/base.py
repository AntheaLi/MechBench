"""Base class for agent adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mechbench.interface.agent_api import AgentAPI
from mechbench.interface.schemas import FinalReport, InterventionPrediction


class MechBenchAgent(ABC):
    @abstractmethod
    def investigate(self, api: AgentAPI) -> tuple[FinalReport, list[InterventionPrediction]]:
        """Run one full investigation episode."""
