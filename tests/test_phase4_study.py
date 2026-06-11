from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mechbench.phase4.reporting import render_phase4_report, write_phase4_report
from mechbench.phase4.study import (
    estimate_study_cost,
    list_study_packs,
    load_study_pack,
    run_phase4_study,
)
from mechbench.utils.config import dump_json


class Phase4StudyTest(unittest.TestCase):
    def test_lists_bundled_packs(self) -> None:
        names = {pack["name"] for pack in list_study_packs()}
        self.assertIn("phase4_free_smoke", names)
        self.assertIn("phase4_frontier_template", names)

    def test_dry_run_resolves_pack_and_cost(self) -> None:
        payload = run_phase4_study("phase4_free_smoke", dry_run=True)
        self.assertTrue(payload["dry_run"])
        self.assertEqual(len(payload["resolved"]["world_ids"]), 10)
        self.assertEqual(payload["preflight"]["estimated_total_cost_usd"], 0.0)

    def test_free_pack_runs_and_renders_report(self) -> None:
        payload = run_phase4_study(
            {
                "name": "tiny_free",
                "worlds": {"selector": "simulator_compiled_balanced", "count": 2},
                "repeats": 1,
                "max_cost_usd": 0.0,
                "include_traces": False,
                "runs": [
                    {"id": "naive", "agent": "naive"},
                    {"id": "llm_mock", "agent": "llm_mock"},
                ],
            }
        )
        self.assertEqual(payload["summary"]["completed_episodes"], 4)
        self.assertIn("naive", payload["summary"]["runs"])
        report = render_phase4_report(payload)
        self.assertIn("# tiny_free Report", report)
        self.assertIn("## Confusion Matrices", report)
        self.assertIn("## Experiment Sequences", report)

    def test_cost_guard_blocks_expensive_pack(self) -> None:
        pack = load_study_pack("phase4_frontier_template")
        with self.assertRaises(ValueError):
            run_phase4_study(pack, dry_run=True, max_cost_usd_override=0.01)

    def test_unknown_price_requires_override(self) -> None:
        pack = {
            "name": "unknown_price",
            "worlds": {"selector": "simulator_compiled_balanced", "count": 1},
            "repeats": 1,
            "runs": [
                {
                    "id": "mystery",
                    "agent": "llm_scaffold",
                    "provider": "openai",
                    "model": "mystery-model",
                    "api_key_env": "OPENAI_API_KEY",
                }
            ],
        }
        with self.assertRaises(ValueError):
            run_phase4_study(pack, dry_run=True)
        payload = run_phase4_study(pack, dry_run=True, allow_unknown_cost=True)
        self.assertFalse(payload["preflight"]["known_cost"])

    def test_paid_run_requires_credentials_before_execution(self) -> None:
        pack = {
            "name": "credential_guard",
            "worlds": {"selector": "simulator_compiled_balanced", "count": 1},
            "repeats": 1,
            "max_cost_usd": 1.0,
            "runs": [
                {"id": "naive", "agent": "naive"},
                {
                    "id": "paid",
                    "agent": "llm_scaffold",
                    "provider": "openai",
                    "model": "gpt-5.4-mini",
                    "api_key_env": "OPENAI_API_KEY",
                    "price_per_million_tokens": {"input": 1.0, "output": 1.0},
                },
            ],
        }
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                run_phase4_study(pack)
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_skip_missing_credentials_keeps_free_runs(self) -> None:
        pack = {
            "name": "credential_skip",
            "worlds": {"selector": "simulator_compiled_balanced", "count": 1},
            "repeats": 1,
            "max_cost_usd": 1.0,
            "include_traces": False,
            "runs": [
                {"id": "naive", "agent": "naive"},
                {
                    "id": "paid",
                    "agent": "llm_scaffold",
                    "provider": "openai",
                    "model": "gpt-5.4-mini",
                    "api_key_env": "OPENAI_API_KEY",
                    "price_per_million_tokens": {"input": 1.0, "output": 1.0},
                },
            ],
        }
        with patch.dict("os.environ", {}, clear=True):
            payload = run_phase4_study(pack, skip_missing_credentials=True)
        self.assertEqual(payload["summary"]["completed_episodes"], 1)
        self.assertEqual(payload["resolved"]["skipped_runs"][0]["run_id"], "paid")

    def test_estimate_study_cost_uses_explicit_price(self) -> None:
        cost = estimate_study_cost(
            [
                {
                    "id": "paid",
                    "agent": "llm_scaffold",
                    "provider": "openai",
                    "model": "custom",
                    "price_per_million_tokens": {"input": 1.0, "output": 2.0},
                }
            ],
            world_count=2,
            repeats=3,
            token_estimate={"input_tokens_per_episode": 1000, "output_tokens_per_episode": 500},
        )
        self.assertEqual(cost["episodes_per_run"], 6)
        self.assertAlmostEqual(cost["estimated_total_cost_usd"], 0.012)

    def test_write_phase4_report(self) -> None:
        payload = run_phase4_study(
            {
                "name": "report_smoke",
                "worlds": {"selector": "simulator_compiled_balanced", "count": 1},
                "repeats": 1,
                "include_traces": False,
                "runs": [{"id": "naive", "agent": "naive"}],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "study.json"
            output_path = root / "report.md"
            dump_json(input_path, payload)
            write_phase4_report(input_path, output_path)
            self.assertIn("report_smoke", output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
