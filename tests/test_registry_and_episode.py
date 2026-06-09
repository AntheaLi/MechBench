from __future__ import annotations

import unittest

from agents.oracle import OracleAgent
from mechbench.registry import Registry


class RegistryAndEpisodeTest(unittest.TestCase):
    def test_discovers_fixture_world(self) -> None:
        registry = Registry("families")
        self.assertIn("fixture/parameter_laundering", registry.list_worlds())

    def test_oracle_episode_runs_and_scores(self) -> None:
        registry = Registry("families")
        result = registry.get_episode("fixture/parameter_laundering").run(OracleAgent())
        self.assertEqual(result.world_id, "fixture/parameter_laundering")
        self.assertGreater(result.score.total, 90.0)
        experiment_records = [
            record for record in result.ledger if record.get("action_type") == "run_experiment"
        ]
        self.assertEqual(len(experiment_records), 3)


if __name__ == "__main__":
    unittest.main()
