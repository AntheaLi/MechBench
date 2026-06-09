from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mechbench.core.budget import Budget, BudgetExceeded
from mechbench.core.evidence_ledger import EvidenceLedger
from mechbench.core.workspace import Workspace, WorkspaceAccessError
from mechbench.interface.schemas import BudgetConfig, ExperimentRequest, ExperimentResult
from mechbench.registry import Registry


class BudgetLedgerWorkspaceTest(unittest.TestCase):
    def test_budget_enforces_limit(self) -> None:
        budget = Budget(BudgetConfig(max_cost=1.0))
        budget.consume(0.75)
        with self.assertRaises(BudgetExceeded):
            budget.consume(0.26)

    def test_ledger_records_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = EvidenceLedger(Path(temp_dir), "episode_x")
            request = ExperimentRequest.from_dict({"variant": "baseline"})
            result = ExperimentResult.from_dict(
                {
                    "experiment_id": "exp_0001",
                    "status": "completed",
                    "cost": 1.0,
                    "config_hash": "sha256:test",
                    "metrics": {"accuracy": 0.7},
                }
            )
            ledger.record_experiment(request, result, remaining_budget=2.0)
            self.assertEqual(ledger.experiment_ids(), {"exp_0001"})
            self.assertEqual(ledger.total_experiment_cost(), 1.0)

    def test_workspace_inspect_cannot_escape(self) -> None:
        registry = Registry("families")
        episode = registry.get_episode("fixture/parameter_laundering")
        workspace = Workspace(episode.family, episode.world).create()
        with self.assertRaises(WorkspaceAccessError):
            workspace.inspect("../MechanismBench_Codebase.md")


if __name__ == "__main__":
    unittest.main()
