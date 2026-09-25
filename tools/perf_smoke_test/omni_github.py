# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Emit the omni-github test-result artifact for the perf-smoke gate.

Writes the artifact directory omni-github ingests (a manifest plus a result JSON)
directly from the aggregate's scored rows, so per-task perf diagnostics -- FPS,
regression, hardware, software, and contract hashes -- are indexed as
``custom.perf_smoke.*`` fields instead of being flattened into a lossy pass/fail
message. This is the perf gate's analogue of the shared JUnit upload action; it is
kept separate because that action's JUnit contract cannot carry these structured
custom fields.

See ``.github/actions/upload-omni-github-test-results/result-json.schema.json`` for
the schema the emitted result JSON validates against.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    from .gate_types import OracleVerdict
except ImportError:  # executed as a script, not a package
    from gate_types import OracleVerdict

MANIFEST_NAME = "omni-github-test-results-upload.json"
RESULT_REL_PATH = "_testoutput/test_results.json"
RESULT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
CUSTOM_NAMESPACE = "perf_smoke"
TEST_TOOL_ID = "perf-smoke"

_PASSING_VERDICTS = (OracleVerdict.PASS, OracleVerdict.WARN)


def _number(value: Any) -> float | int | None:
    """Return a finite number for storage, or ``None`` so the field is dropped.

    Booleans are rejected here because they are handled as booleans elsewhere.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def _drop_none(fields: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is ``None`` (omni-github stores omitted, not null)."""
    return {key: value for key, value in fields.items() if value is not None}


def _launch_value(bench_result: Any, key: str) -> Any:
    """Read a run-shape value, preferring the requested launch_config over the ran info."""
    launch_config = getattr(bench_result, "launch_config", None) or {}
    if launch_config.get(key) is not None:
        return launch_config.get(key)
    benchmark_info = getattr(bench_result, "benchmark_info", None) or {}
    return benchmark_info.get(key)


def _custom_fields(result: Any, bench_result: Any) -> dict[str, Any]:
    """Project the oracle verdict and bench result into ``custom.perf_smoke`` fields.

    Intentionally a compact "dashboard core": identity, run shape, verdict, timing,
    the headline FPS/regression, and GPU/host memory. Deeper diagnostics (software
    versions, contract hashes, raw fps spread, threshold provenance) stay in the
    uploaded bench artifacts and can be promoted here later if a dashboard needs them.
    """
    runtime_resources = getattr(bench_result, "runtime_resources", None) or {}
    fps_mean = result.measured_fps if result.measured_fps is not None else getattr(bench_result, "raw_fps_mean", None)
    return _drop_none(
        {
            "task_id": result.task_id,
            "physics_backend": getattr(bench_result, "physics_backend", None),
            "render_backend": getattr(bench_result, "render_backend", None),
            "num_envs": _number(_launch_value(bench_result, "num_envs")),
            "num_frames": _number(_launch_value(bench_result, "num_frames")),
            "verdict": result.verdict.value,
            "failure_phase": result.failure_phase,
            "wall_time_s": _number(getattr(bench_result, "wall_time_s", None)),
            "startup_time_s": _number(getattr(bench_result, "startup_time_s", None)),
            "fps_mean": _number(fps_mean),
            "regression_pct": _number(result.regression_pct),
            "vram_mb": _number(runtime_resources.get("gpu_mem_used_mb")),
            "sysram_mb": _number(runtime_resources.get("system_ram_used_mb")),
        }
    )


def _message(result: Any) -> str:
    parts = [result.verdict.value]
    if result.measured_fps is not None:
        parts.append(f"fps={result.measured_fps:.1f}")
    if result.regression_pct is not None:
        parts.append(f"regression={result.regression_pct:.2f}%")
    if result.failure_phase:
        parts.append(f"phase={result.failure_phase}")
    return " ".join(parts)


def _row(result: Any, bench_result: Any) -> dict[str, Any]:
    duration = max(float(getattr(bench_result, "wall_time_s", None) or 0.0), 0.0)
    passed = result.verdict in _PASSING_VERDICTS
    row: dict[str, Any] = {
        "test_id": f"perf-smoke.{result.task_id}::{result.backend}",
        "test_name": result.backend,
        "test_type": "performance",
        "passed": passed,
        "duration": duration,
        "custom": {CUSTOM_NAMESPACE: _custom_fields(result, bench_result)},
    }
    if not passed:
        row["message"] = _message(result)
    return row


def build_result(rows, *, platform: str, app_config: str, test_tool_id: str = TEST_TOOL_ID) -> dict[str, Any]:
    """Build the omni-github result payload from ``(oracle_result, bench_result)`` rows."""
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "test_tool_id": test_tool_id,
        "app": {"platform": platform, "config": app_config},
        "tests": [_row(result, bench_result) for result, bench_result in rows],
    }


def write_artifact(rows, output_dir, *, platform: str, app_config: str, test_tool_id: str = TEST_TOOL_ID) -> Path:
    """Write the manifest and result JSON omni-github ingests; returns the artifact root."""
    output_dir = Path(output_dir)
    result_path = output_dir / RESULT_REL_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(build_result(rows, platform=platform, app_config=app_config, test_tool_id=test_tool_id)),
        encoding="utf-8",
    )
    manifest = {"schema_version": MANIFEST_SCHEMA_VERSION, "result_paths": [RESULT_REL_PATH]}
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return output_dir
