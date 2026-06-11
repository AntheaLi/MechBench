"""Evidence grounding and claim citation scoring."""

from __future__ import annotations

import json

from mechbench.core.evidence_ledger import EvidenceLedger
from mechbench.interface.schemas import FinalReport


def score_evidence_grounding(report: FinalReport, ledger: EvidenceLedger, weight: float = 10.0) -> float:
    cited_ids = [experiment_id for item in report.evidence for experiment_id in item.experiment_ids]
    if not cited_ids:
        return 0.0
    existing = ledger.experiment_ids()
    valid = sum(1 for experiment_id in cited_ids if experiment_id in existing)
    validity_score = valid / len(cited_ids)
    return weight * validity_score * _claim_strength_multiplier(report, ledger)


def _claim_strength_multiplier(report: FinalReport, ledger: EvidenceLedger) -> float:
    variants = _cited_variants(report, ledger)
    control_variants = {
        variant for variant in variants if variant not in {"baseline", "proposed"}
    }
    if report.mechanism_support == "unknown":
        return 0.5
    if report.mechanism_support in {
        "claimed_mechanism_supported",
        "claimed_mechanism_not_supported",
    } and not control_variants:
        return 0.35
    return 1.0


def _cited_variants(report: FinalReport, ledger: EvidenceLedger) -> set[str]:
    variants = set()
    existing = ledger.experiment_ids()
    for item in report.evidence:
        for experiment_id in item.experiment_ids:
            if experiment_id not in existing:
                continue
            request_path = ledger.root / experiment_id / "request.json"
            if not request_path.exists():
                continue
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            variant = request.get("variant")
            if variant:
                variants.add(str(variant))
    return variants
