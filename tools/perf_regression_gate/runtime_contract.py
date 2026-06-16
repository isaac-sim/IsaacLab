# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime compatibility contract construction for baseline matching"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

try:
    from .backend_identity import BackendIdentity
except ImportError:  # pragma: no cover - supports direct script imports
    from backend_identity import BackendIdentity


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def stable_hash(data: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()[:16]


def _get_path(data: Mapping[str, Any], dotted_path: str) -> Any:
    cur: Any = data
    for part in dotted_path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _field_list(policy: Mapping[str, Any], backend: BackendIdentity) -> list[str]:
    fields: list[str] = []

    def add_many(values: Any) -> None:
        for value in values or []:
            field = str(value)
            if field not in fields:
                fields.append(field)

    add_many(policy.get("always"))
    by_physics = policy.get("by_physics_backend") or {}
    if isinstance(by_physics, Mapping):
        add_many(by_physics.get(backend.physics_backend))
    by_render = policy.get("by_render_backend") or {}
    if isinstance(by_render, Mapping) and backend.render_backend:
        add_many(by_render.get(backend.render_backend))
    return fields


def runtime_source(provenance: dict[str, Any] | None, gpu_diag: dict[str, Any] | None) -> dict[str, Any]:
    provenance = provenance or {}
    return {
        "software": provenance.get("software") or {},
        "hardware": provenance.get("hardware") or {},
        "gpu_diag": gpu_diag or {},
    }


def build_runtime_contract(
    *,
    provenance: dict[str, Any] | None,
    gpu_diag: dict[str, Any] | None,
    backend: BackendIdentity,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Build the compatibility contract and hash used for baseline matching."""
    source = runtime_source(provenance, gpu_diag)
    selected = {field: _get_path(source, field) for field in _field_list(policy, backend)}
    contract = {
        "runtime_contract_version": int(policy.get("contract_version", 1)),
        "fields": selected,
    }
    return contract, stable_hash(contract)


def build_runtime_publish_info(
    *,
    provenance: dict[str, Any] | None,
    gpu_diag: dict[str, Any] | None,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Return debug/runtime fields published for humans but not used for matching."""
    source = runtime_source(provenance, gpu_diag)
    publish_only = {
        str(field): _get_path(source, str(field))
        for field in policy.get("publish_only", [])
    }
    publish_only = {key: value for key, value in publish_only.items() if value is not None}
    return {
        "software": source["software"],
        "publish_only": publish_only,
    }
