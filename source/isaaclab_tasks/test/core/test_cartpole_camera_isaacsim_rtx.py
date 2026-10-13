# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Cartpole camera task renders through Kit's RTX renderer on Windows.

Linux CI compares this scene against golden images in ``test_rendering_cartpole.py``.
"""

import sys

import pytest

if sys.platform != "win32":
    pytest.skip("Linux CI covers this scene with golden images", allow_module_level=True)

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True)

import torch  # noqa: E402

from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env import CartpoleCameraEnv  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

pytestmark = pytest.mark.windows_ci


def test_cartpole_camera_renders_and_steps():
    env_cfg, _ = resolve_task_config(
        "Isaac-Cartpole-Camera-Direct", "", overrides=("physics=isaacsim_physx", "renderer=isaacsim_rtx")
    )
    env_cfg.scene.num_envs = 2
    env_cfg.frame_stack = 1
    env = CartpoleCameraEnv(cfg=env_cfg)
    try:
        obs, _ = env.reset()
        image = obs["policy"]
        assert torch.isfinite(image).all(), "Camera observation contains NaN or infinity"
        assert image.amax() > image.amin(), "Camera observation is constant"
        _, reward, _, _, _ = env.step(torch.zeros(env.num_envs, 1, device=env.device))
        assert torch.isfinite(reward).all(), "Cartpole reward contains NaN or infinity"
    finally:
        env.close()
