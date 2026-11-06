# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark modes shared by the rendering benchmark tasks."""

from __future__ import annotations

import os
from typing import Any, Literal, cast

BenchmarkMode = Literal["render", "physics_render"]
"""How a rendering benchmark animates its scene.

``"render"`` writes joint poses directly after physics and requires ``scene.lazy_sensor_update=True``, so the
camera renders the written pose. Isaac RTX direct posing also requires no Kit app-pumping visualizer (for example,
omit ``--visualizer``). ``"physics_render"`` sends actuator targets before physics and renders the resulting state.
Both modes still step physics. A task with a non-``None`` mode also enables the runtime's scope profiling.
"""

BENCHMARK_MODES: tuple[BenchmarkMode, ...] = ("render", "physics_render")
"""Explicit benchmark modes; ``None`` disables benchmark animation and scope profiling."""


def read_benchmark_mode() -> BenchmarkMode | None:
    """Read the benchmark mode from ``BENCHMARK_MODE``, rejecting unknown values.

    Returns:
        The configured mode, or ``None`` when the variable is unset.

    Raises:
        ValueError: If ``BENCHMARK_MODE`` is set to a value outside :data:`BENCHMARK_MODES`.
    """
    mode = os.getenv("BENCHMARK_MODE")
    if mode is not None and mode not in BENCHMARK_MODES:
        raise ValueError(f"Unknown BENCHMARK_MODE '{mode}'. Expected one of {list(BENCHMARK_MODES)}.")
    return cast(BenchmarkMode | None, mode)


BENCHMARK_MODE: BenchmarkMode | None = read_benchmark_mode()
"""The ``BENCHMARK_MODE`` environment variable, read once at import."""


def check_direct_pose_cfg(benchmark_mode: BenchmarkMode | None, lazy_sensor_update: bool) -> None:
    """Reject a ``"render"`` mode whose sensors would update before the direct pose is written.

    Raises:
        ValueError: If the mode is ``"render"`` and sensors do not update lazily.
    """
    if benchmark_mode == "render" and not lazy_sensor_update:
        raise ValueError("Render benchmark mode requires scene.lazy_sensor_update=True to render the direct pose.")


def check_direct_pose_rendering(env: Any, benchmark_mode: BenchmarkMode | None) -> None:
    """Close the environment and fail if Isaac RTX would render through a Kit app-pumping visualizer before the
    ``"render"`` pose write.

    Raises:
        ValueError: If the environment cannot render its direct pose.
    """
    if (
        benchmark_mode == "render"
        and "isaac_rtx" in env.sim.render_context.renderer_types
        and any(visualizer.pumps_app_update() for visualizer in env.sim.visualizers)
    ):
        env.close()
        raise ValueError(
            "Isaac RTX direct posing requires no visualizer, or only visualizers that do not pump the Kit app"
            " loop. Alternatively, use benchmark_mode=physics_render."
        )
