# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared canonical-JSON hashing used by every perf-smoke contract.

All contract hashes (``runtime_contract_hash``, ``launch_config_hash``,
``benchmark_contract_hash``, ``era_key``) must agree byte-for-byte across the
producer and consumer sides of the gate, so the hashing primitive lives in one
place. Changing :func:`canonical_json` or :func:`stable_hash` shifts every
stored hash and invalidates the baseline branch, so treat this module as frozen.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(data: dict[str, Any]) -> str:
    """Return a deterministic JSON encoding with sorted keys and no whitespace."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def stable_hash(data: dict[str, Any]) -> str:
    """Return the first 16 hex chars of the SHA-256 over the canonical JSON."""
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()[:16]
