# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Comparability contract: decides which runs may be pooled into one baseline.

Two runs are comparable when they measured the same workload on the same hardware
with the same externally pinned runtime. Pooling across a Newton or Isaac Sim
bump would hide or invent regressions, so those versions are part of the key.

The dependency set is conditional on the backends actually in use, so a Newton
bump invalidates only Newton baselines and leaves PhysX baselines usable.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .metrics import PerfSmokeError, mapping, positive_int

#: Externally pinned versions that affect measured performance, keyed by where they
#: apply. Keys are the values a bundle reports in ``run.config``, which are NOT the
#: matrix names. An unrecognised value raises.
#: ``version`` is a manual invalidation switch. It is part of the hashed contract, so
#: raising it retires every baseline recorded so far, and the gate reports SKIP until
#: history rebuilds.
RUNTIME_COMPATIBILITY: dict[str, Any] = {
    "version": 3,
    "always": ("torch", "warp"),
    "by_physics_backend": {
        "physx": ("ovphysx", "isaacsim"),
        "newton_mjwarp": ("newton", "mujoco", "mjwarp"),
    },
    "by_render_backend": {
        "newton": (),
        # Kit's RTX renderer, versioned with the Isaac Sim build it ships in.
        "isaacsim_rtx": ("isaacsim",),
        # Standalone OV RTX renderer, shipped as its own package.
        "ovrtx": ("ovrtx",),
    },
}

#: Versions whose semver build metadata is rebuild noise rather than a real variant.
#: e.g. Isaac Sim reports ``6.0.1-rc.7+release.42383.32955d8d.gl``.
#: This should NOT be applied to torch e.g. ``+cu128`` vs ``+cu130``.
_STRIP_BUILD_METADATA = ("isaacsim",)

_UNKNOWN_GPU = "unknown-gpu"


def normalize_gpu_model(value: Any) -> str:
    """Return a canonical slug for a raw GPU name."""
    raw = value.strip() if isinstance(value, str) else ""
    if not raw:
        return _UNKNOWN_GPU
    without_vendor = re.sub(r"^nvidia\s+", "", raw, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "_", without_vendor.lower()).strip("_") or _UNKNOWN_GPU


def strip_build_metadata(version: str) -> str:
    """Drop semver build metadata so a rebuild of the same release still matches."""
    return version.split("+", 1)[0].strip()


@dataclass(frozen=True)
class Contract:
    """The full baseline match key for one benchmark run.

    Attributes:
        workload: Task identity i.e. what was measured.
        runtime: Externally pinned versions and hardware i.e. what it was measured on.
    """

    workload: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the contract as a plain, JSON-serialisable dict."""
        return {"workload": dict(self.workload), "runtime": dict(self.runtime)}

    @property
    def hash(self) -> str:
        """Return a stable 16-hex-character digest of the whole contract."""
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def matches(self, other: Contract) -> bool:
        """Return whether ``other`` may be pooled with this run."""
        return self.as_dict() == other.as_dict()


def _required_version_names(physics_backend: str, render_backend: str | None) -> tuple[str, ...]:
    """Return the version keys that matter for these backends.

    Raises:
        PerfSmokeError: If either backend is not in :data:`RUNTIME_COMPATIBILITY`.
    """
    names: list[str] = list(RUNTIME_COMPATIBILITY["always"])

    physics_map = RUNTIME_COMPATIBILITY["by_physics_backend"]
    if physics_backend not in physics_map:
        raise PerfSmokeError(
            f"Unknown physics backend {physics_backend!r}; add it to"
            f" RUNTIME_COMPATIBILITY['by_physics_backend'] (known: {sorted(physics_map)})"
        )
    names.extend(physics_map[physics_backend])

    if render_backend:
        render_map = RUNTIME_COMPATIBILITY["by_render_backend"]
        if render_backend not in render_map:
            raise PerfSmokeError(
                f"Unknown render backend {render_backend!r}; add it to"
                f" RUNTIME_COMPATIBILITY['by_render_backend'] (known: {sorted(render_map)})"
            )
        names.extend(render_map[render_backend])

    return tuple(sorted(set(names)))


def build(bundle: dict[str, Any]) -> Contract:
    """Build the comparability contract for a runtime bundle.

    Args:
        bundle: A schema-v1 runtime bundle.

    Returns:
        The contract used to select poolable baseline rows.

    Raises:
        PerfSmokeError: If required identity or hardware fields are missing.
    """
    run = mapping(bundle.get("run"), "run")
    config = mapping(run.get("config"), "run.config")
    versions = mapping(bundle.get("versions"), "versions")
    hardware = mapping(bundle.get("hardware"), "hardware")

    task = run.get("task")
    if not isinstance(task, str) or not task.strip():
        raise PerfSmokeError("run.task must be a non-empty string")

    physics_backend = config.get("physics_backend")
    if not isinstance(physics_backend, str) or not physics_backend.strip():
        raise PerfSmokeError("run.config.physics_backend must be a non-empty string")

    raw_render = config.get("rendering_backend")
    if not isinstance(raw_render, str) or not raw_render.strip():
        raise PerfSmokeError("run.config.rendering_backend must be a non-empty string")
    render_backend = None if raw_render == "none" else raw_render

    presets = config.get("presets", [])
    if not isinstance(presets, list) or not all(isinstance(item, str) for item in presets):
        raise PerfSmokeError("run.config.presets must be a list of strings")

    gpu_devices = hardware.get("gpu_devices")
    if not isinstance(gpu_devices, list) or not gpu_devices:
        raise PerfSmokeError("hardware.gpu_devices must contain at least one device")
    gpu_name = mapping(gpu_devices[0], "hardware.gpu_devices[0]").get("name")
    if not isinstance(gpu_name, str) or not gpu_name.strip():
        raise PerfSmokeError("hardware.gpu_devices[0].name must be a non-empty string")

    cpu_name = hardware.get("cpu_name")
    if not isinstance(cpu_name, str) or not cpu_name.strip():
        raise PerfSmokeError("hardware.cpu_name must be a non-empty string")

    pinned: dict[str, str] = {}
    for name in _required_version_names(physics_backend, render_backend):
        value = versions.get(name)
        if value is None:
            raise PerfSmokeError(f"versions.{name} is required to compare {physics_backend} runs but is missing")
        if not isinstance(value, str) or not value.strip():
            raise PerfSmokeError(f"versions.{name} must be a non-empty string")
        pinned[name] = strip_build_metadata(value) if name in _STRIP_BUILD_METADATA else value.strip()

    return Contract(
        workload={
            "task": task,
            "physics_backend": physics_backend,
            "render_backend": render_backend,
            "presets": sorted(presets),
            "num_envs": positive_int(run.get("num_envs"), "run.num_envs"),
        },
        runtime={
            "versions": pinned,
            "gpu_model": normalize_gpu_model(gpu_name),
            "cpu_name": cpu_name.strip(),
            "contract_version": RUNTIME_COMPATIBILITY["version"],
        },
    )


def from_dict(data: Any) -> Contract:
    """Rebuild a :class:`Contract` from its serialised form (a stored baseline row)."""
    payload = mapping(data, "contract")
    return Contract(
        workload=mapping(payload.get("workload"), "contract.workload"),
        runtime=mapping(payload.get("runtime"), "contract.runtime"),
    )


def valid_backend_keys() -> frozenset[str]:
    """Return every backend key :func:`backend_key` can produce."""
    physics = RUNTIME_COMPATIBILITY["by_physics_backend"]
    renderers = RUNTIME_COMPATIBILITY["by_render_backend"]
    return frozenset({*physics, *(f"{p}_{r}" for p in physics for r in renderers)})


def backend_key(contract: Contract) -> str:
    """Return the ``{physics}`` or ``{physics}_{render}`` key used by the threshold config."""
    physics = contract.workload.get("physics_backend")
    render = contract.workload.get("render_backend")
    return f"{physics}_{render}" if render else str(physics)
