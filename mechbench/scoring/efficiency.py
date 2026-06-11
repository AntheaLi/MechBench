"""Experimental efficiency scoring."""

from __future__ import annotations

from mechbench.core.evidence_ledger import EvidenceLedger


def score_efficiency(
    ledger: EvidenceLedger,
    certificate: dict,
    weight: float = 5.0,
) -> float:
    spent = ledger.total_experiment_cost()
    oracle = float(certificate.get("oracle_minimum_cost", certificate.get("oracle_cost", 0.0)) or 0.0)
    exhaustive = float(certificate.get("exhaustive_cost", 0.0) or 0.0)
    if spent <= 0:
        return weight
    if oracle <= 0:
        return weight * 0.5
    if spent <= oracle:
        return weight
    ceiling = exhaustive if exhaustive > oracle else oracle * 2.0
    if spent >= ceiling:
        return 0.0
    return weight * (1.0 - (spent - oracle) / (ceiling - oracle))

