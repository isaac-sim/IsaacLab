# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed contracts for the perf-gate's own artifacts (schema v1).

The gate's analogue of :mod:`isaaclab.test.benchmark.schema`: the fields the gate
computes on are typed dataclass attributes, while open pass-through sub-structures
(``benchmark_info``, ``provenance``, ``launch_config``, the compatibility contracts)
stay as ``dict`` fields — mirroring ``RuntimeBundle``'s typed fields + ``extra: dict``.

:class:`RuntimeSample` is the adapter's projection of a benchmark ``RuntimeBundle``.
:class:`BenchResult` is the typed shape of ``perf_smoke_test_result.json``; it
serializes (:meth:`BenchResult.to_dict`) to the same wire format the gate has always
written (plus an additive ``schema_version``), and reconstructs strictly via
:meth:`BenchResult.from_dict`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

CONTRACT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class RuntimeSample:
    """Adapter projection of a schema-v1 ``RuntimeBundle`` into the gate's fields.

    Aggregates (fps/startup) and run identity are typed; ``provenance`` and
    ``gpu_diag`` remain dicts (open provenance payloads the gate carries through).
    """

    fps_mean: float | None = None
    fps_std: float | None = None
    fps_min: float | None = None
    fps_max: float | None = None
    startup_time_s: float | None = None
    task: str | None = None
    num_envs: int | None = None
    seed: int | None = None
    num_frames: int | None = None
    warmup_frames: int | None = None
    status: str | None = None
    physics_backend: str | None = None
    render_backend: str | None = None
    presets: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    gpu_diag: dict[str, Any] = field(default_factory=dict)

    def benchmark_info(self) -> dict[str, Any]:
        """Return the run's self-reported identity as a dict.

        Feeds the dict-based drift / backend-identity helpers; keys whose value is
        ``None`` are omitted (matching the historical ``benchmark_info`` shape).
        """
        info = {
            "task": self.task,
            "num_envs": self.num_envs,
            "seed": self.seed,
            "num_frames": self.num_frames,
            "warmup_frames": self.warmup_frames,
            "status": self.status,
            "physics_backend": self.physics_backend,
            "render_backend": self.render_backend,
            "presets": self.presets or None,
        }
        return {k: v for k, v in info.items() if v is not None}


@dataclass(frozen=True)
class BenchResult:
    """Typed shape of ``perf_smoke_test_result.json`` (schema v1).

    Field order matches the historical JSON so serialization is wire-stable; the
    only addition is a trailing ``schema_version``. The leading identity fields are
    required (a build that forgets one fails at construction); the rest default so a
    HARD_FAILURE result (no benchmark output) still constructs cleanly.
    """

    # --- identity (required) ---
    task_id: str
    backend: str
    physics_backend: str
    render_backend: str | None
    backend_key: str
    preset: str
    # --- run status ---
    attempt: int = 1
    was_retried: bool = False
    exit_code: int = 0
    failure_phase: str | None = None
    stdout_tail: str = ""
    wall_time_s: float | None = None
    startup_time_s: float | None = None
    perf_smoke_test_info_present: bool = False
    # --- perf (steady-state aggregates) ---
    raw_fps_mean: float | None = None
    raw_fps_std: float | None = None
    raw_fps_min: float | None = None
    raw_fps_max: float | None = None
    # --- open pass-through payloads + matching/contracts ---
    benchmark_info: dict[str, Any] = field(default_factory=dict)
    observed_backend: dict[str, Any] | None = None
    config_mismatch: str | None = None
    runtime_contract: dict[str, Any] | None = None
    runtime_contract_hash: str | None = None
    runtime_info: dict[str, Any] | None = None
    gpu_diag: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    launch_config: dict[str, Any] = field(default_factory=dict)
    launch_config_hash: str | None = None
    benchmark_contract_hash: str | None = None
    baseline_epoch: int = 1
    task_config_snapshot: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the on-disk JSON shape (plain dicts, wire-stable order)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchResult:
        """Reconstruct from a parsed ``perf_smoke_test_result.json``.

        Strict on this version's own artifacts: unknown keys are dropped, and
        missing required identity fields raise (there is no legacy-tolerance layer).
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
