# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixture builders for perf-gate unit tests: synthetic info artifacts + bench results."""

import json
from pathlib import Path


def write_info(
    artifact_dir: Path,
    fps_series: list[float],
    step_times: list[float] | None = None,
    benchmark_info: dict | None = None,
) -> Path:
    """Write a minimal ``perf_regression_gate_info.json`` matching the json benchmark backend layout."""
    phases: list[dict] = []
    if benchmark_info is not None:
        phases.append(
            {
                "phase_name": "benchmark_info",
                "metadata": [{"name": f"benchmark_non_rl benchmark_info {k}", "data": v} for k, v in benchmark_info.items()],
            }
        )
    value: dict = {"Environment step effective FPS": fps_series}
    if step_times is not None:
        value["Environment step times"] = step_times
    phases.append(
        {
            "phase_name": "runtime",
            "measurements": [{"name": "benchmark_non_rl runtime Step Frametimes", "value": value}],
        }
    )
    path = artifact_dir / "perf_regression_gate_info.json"
    path.write_text(json.dumps(phases))
    return path


def make_bench_result(
    task_id: str = "Isaac-Cartpole",
    backend: str = "physx",
    present: bool = True,
    failure_phase: str | None = None,
    was_retried: bool = False,
    **extra,
) -> dict:
    """Build the subset of perf_regression_gate_result.json fields the oracle reads."""
    br = {
        "task_id": task_id,
        "backend": backend,
        "backend_key": backend,
        "failure_phase": failure_phase,
        "was_retried": was_retried,
        "perf_regression_gate_info_present": present,
        "startup_time_s": 1.0,
        "wall_time_s": 10.0,
        "gpu_diag": {"gpu_mem_used_mb": 1234.0},
    }
    br.update(extra)
    return br
