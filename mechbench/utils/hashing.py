"""Stable hashing utilities."""

from __future__ import annotations

import hashlib
from typing import Any

from mechbench.utils.config import stable_json


def hash_payload(payload: Any) -> str:
    digest = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"

