"""Immutable JSONL evidence ledger."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mechbench.interface.schemas import ExperimentRequest, ExperimentResult
from mechbench.utils.config import dump_json
from mechbench.utils.hashing import hash_payload


@dataclass
class EvidenceLedger:
    root: Path
    episode_id: str

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "ledger.jsonl"
        self._action_index = self._load_existing_count()

    def _load_existing_count(self) -> int:
        if not self.path.exists():
            return 0
        return sum(1 for _ in self.path.open("r", encoding="utf-8"))

    def _append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._action_index += 1

    def record_inspect(self, path: str) -> None:
        self._append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "episode_id": self.episode_id,
                "action_index": self._action_index,
                "action_type": "inspect",
                "path": path,
                "cost": 0.0,
            }
        )

    def record_analysis(self, experiment_id: str, analysis_spec: dict[str, Any], result: dict[str, Any]) -> None:
        self._append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "episode_id": self.episode_id,
                "action_index": self._action_index,
                "action_type": "analyze_artifact",
                "experiment_id": experiment_id,
                "request_hash": hash_payload(analysis_spec),
                "response_hash": hash_payload(result),
                "cost": 0.0,
            }
        )

    def record_experiment(
        self,
        request: ExperimentRequest,
        result: ExperimentResult,
        remaining_budget: float,
    ) -> None:
        result_dir = self.root / result.experiment_id
        dump_json(result_dir / "request.json", request.to_dict())
        dump_json(result_dir / "result.json", result.to_dict())
        self._append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "episode_id": self.episode_id,
                "action_index": self._action_index,
                "action_type": "run_experiment",
                "experiment_id": result.experiment_id,
                "request_hash": hash_payload(request.to_dict()),
                "response_hash": hash_payload(result.to_dict()),
                "cost": result.cost,
                "remaining_budget": remaining_budget,
            }
        )

    def record_submission(self, action_type: str, payload: Any) -> None:
        self._append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "episode_id": self.episode_id,
                "action_index": self._action_index,
                "action_type": action_type,
                "payload_hash": hash_payload(payload),
                "cost": 0.0,
            }
        )

    def export(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def experiment_ids(self) -> set[str]:
        return {
            record["experiment_id"]
            for record in self.export()
            if record.get("action_type") == "run_experiment" and "experiment_id" in record
        }

    def total_experiment_cost(self) -> float:
        return sum(
            float(record.get("cost", 0.0))
            for record in self.export()
            if record.get("action_type") == "run_experiment"
        )

