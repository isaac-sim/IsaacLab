# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video recording helper for the standalone tablecloth comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs.utils.video_recorder import VideoRecorder
    from isaaclab.sim import SimulationContext


class _StandaloneVideoTarget:
    """Expose the simulation fields expected by Isaac Lab's step-driven recorder."""

    def __init__(self, sim: SimulationContext, fps: int):
        self.sim = sim
        self.step_dt = sim.get_physics_dt()
        self.metadata = {"render_fps": fps}


def create_video_recorder(
    sim: SimulationContext,
    *,
    enabled: bool,
    output_dir: str,
    filename_prefix: str,
    video_length: int,
    fps: int,
) -> VideoRecorder | None:
    """Create a step-driven viewport recorder when requested."""
    if not enabled:
        return None

    from isaaclab.envs.utils.video_recorder import VideoRecorder  # noqa: PLC0415
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg  # noqa: PLC0415

    print(f"[INFO]: Recording {video_length / fps:.1f} s to {output_dir}/", flush=True)
    return VideoRecorder(
        VideoRecorderCfg(
            source="visualizer",
            output_dir=output_dir,
            output_filename_prefix=filename_prefix,
            fps=fps,
            video_length=video_length,
        ),
        _StandaloneVideoTarget(sim, fps),
    )
