"""Objective protocol-integrity checks."""

from __future__ import annotations

from mechbench.core.evidence_ledger import EvidenceLedger
from mechbench.interface.schemas import FinalReport, WORLD_LABELS


def score_protocol_integrity(
    report: FinalReport,
    ledger: EvidenceLedger,
    labels: tuple[str, ...] = WORLD_LABELS,
    weight: float = 5.0,
) -> float:
    probabilities = report.causal_probabilities
    if any(value < 0 for value in probabilities.values()):
        return 0.0
    total = sum(probabilities.get(label, 0.0) for label in labels)
    if total <= 0:
        return 0.0

    existing = ledger.experiment_ids()
    for item in report.evidence:
        for experiment_id in item.experiment_ids:
            if experiment_id not in existing:
                return 0.0
    return weight

