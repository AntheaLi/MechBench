"""Controlled experiment execution."""

from __future__ import annotations

from dataclasses import dataclass

from mechbench.core.budget import Budget
from mechbench.core.family import FamilyInterface, World
from mechbench.interface.schemas import ExperimentRequest, ExperimentResult
from mechbench.utils.hashing import hash_payload


@dataclass
class ExperimentRunner:
    family: FamilyInterface
    world: World
    budget: Budget
    _counter: int = 0

    def run(self, request: ExperimentRequest) -> ExperimentResult:
        estimated_cost = self.family.estimate_cost(self.world, request)
        self.budget.consume(estimated_cost)
        self._counter += 1
        experiment_id = f"exp_{self._counter:04d}"
        config_hash = hash_payload(request.to_dict())
        raw_result = self.family.run_experiment(self.world, request)
        return raw_result.with_identity(
            experiment_id=experiment_id,
            config_hash=config_hash,
            cost=estimated_cost,
        )

