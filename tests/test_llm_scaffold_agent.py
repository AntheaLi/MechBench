from __future__ import annotations

import unittest

from agents.llm_scaffold import LLMScaffoldAgent
from mechbench.interface.schemas import FinalReport, InterventionPrediction
from mechbench.registry import Registry


class InvalidBackend:
    def complete(self, messages, *, purpose: str) -> str:
        return "not json"


class LLMScaffoldAgentTest(unittest.TestCase):
    def test_mock_llm_scaffold_runs_and_records_trace(self) -> None:
        agent = LLMScaffoldAgent(max_experiments=4)
        result = Registry("families").get_episode("simulator/seed_hacking_001").run(agent)

        self.assertIsInstance(result.report, FinalReport)
        self.assertEqual(result.report.episode_id, "simulator/seed_hacking_001")
        self.assertEqual(len(result.predictions), len(Registry("families").worlds["simulator/seed_hacking_001"].interventions))
        self.assertGreater(result.report.causal_probabilities["seed_hacking"], 0.5)
        self.assertTrue(any(item["type"] == "llm_request" for item in agent.last_trace))
        self.assertTrue(any(item["type"] == "tool_call" for item in agent.last_trace))
        self.assertTrue(any(item["type"] == "tool_result" for item in agent.last_trace))

    def test_invalid_backend_falls_back_to_valid_payload(self) -> None:
        agent = LLMScaffoldAgent(backend=InvalidBackend(), max_experiments=2)
        result = Registry("families").get_episode("simulator/true_mechanism_001").run(agent)

        self.assertIsInstance(result.report, FinalReport)
        self.assertTrue(all(isinstance(item, InterventionPrediction) for item in result.predictions))
        self.assertEqual(len(result.predictions), len(Registry("families").worlds["simulator/true_mechanism_001"].interventions))
        self.assertTrue(any(item["type"] == "llm_parse_error" for item in agent.last_trace))


if __name__ == "__main__":
    unittest.main()
