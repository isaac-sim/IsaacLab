# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video recording on the Warp environments.

``DirectRLEnvWarp`` and the manager-based Warp envs are separate implementations of the recorder
lifecycle, so both are built through :meth:`WarpFrontend.build_env`, the adapter the CLI uses, and
record from a headless Newton GL visualizer without Kit.
"""

# Check for moviepy before launching so a missing dependency produces a clean collection-time skip.
import pytest

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    pytest.skip("moviepy is not installed; install with: pip install 'moviepy<2'", allow_module_level=True)

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())))

import os
import tempfile

import torch
from isaaclab_experimental.envs.frontend import WarpFrontend
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg as CartpoleDirectEnvCfg
from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import CartpoleEnvCfg as CartpoleManagerEnvCfg

_CLIP = 20  # frames per clip
_STEPS = _CLIP // 2  # shorter than one clip, so only close() can write it


@pytest.mark.parametrize(
    ("cfg_class", "task_id"),
    [(CartpoleDirectEnvCfg, "Isaac-Cartpole-Direct"), (CartpoleManagerEnvCfg, "Isaac-Cartpole")],
    ids=["direct", "manager"],
)
def test_warp_env_records_and_flushes_clip(cfg_class: type, task_id: str):
    """A Warp env records one frame per step and flushes the partial clip on close."""
    env_cfg = cfg_class()
    env_cfg.seed = 42
    env_cfg.scene.num_envs = 1
    env_cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg())
    env_cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(headless=True, window_width=320, window_height=240)]
    with tempfile.TemporaryDirectory() as output_dir:
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:newton_gl", output_dir=output_dir, video_length=_CLIP, video_interval=0, fps=10
            )
        ]
        sim_utils.create_new_stage()
        env = WarpFrontend.build_env(env_cfg, task_id)
        try:
            env.reset()
            actions = torch.zeros(env.num_envs, *env.action_space.shape[1:], device=env.device)
            for _ in range(_STEPS):
                env.step(actions)
        finally:
            env.close()

        clip_path = os.path.join(output_dir, "clip_0000.mp4")
        assert os.path.isfile(clip_path), "close() did not flush the partial clip"
        clip = VideoFileClip(clip_path)
        frames = list(clip.iter_frames())
        clip.close()

    # decoded frame counts round by one against the clip duration; a missing or doubled tick lands outside
    assert _STEPS <= len(frames) <= _STEPS + 1, f"expected one frame per env step, got {len(frames)}"
