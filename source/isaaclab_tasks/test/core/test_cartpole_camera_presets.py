# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer/data-type preset compatibility for the cartpole camera tasks.

The Newton Warp renderer cannot produce the ``simple_shading_*`` data types, which the tasks
offer as an independent ``presets=`` selector. Preset resolution and ``validate()`` are pure
config operations, so no simulator is launched here.
"""

import pytest

from isaaclab.utils import validate

from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env_cfg import (
    CartpoleCameraEnvCfg as CartpoleDirectCameraEnvCfg,
)
from isaaclab_tasks.core.cartpole.cartpole_manager_camera_env_cfg import (
    CartpoleCameraEnvCfg as CartpoleManagerCameraEnvCfg,
)
from isaaclab_tasks.utils.hydra import resolve_presets

_ENV_CFG_CLASSES = [
    pytest.param(CartpoleDirectCameraEnvCfg, id="direct"),
    pytest.param(CartpoleManagerCameraEnvCfg, id="manager"),
]


@pytest.mark.parametrize("cfg_cls", _ENV_CFG_CLASSES)
def test_newton_renderer_rejects_simple_shading(cfg_cls):
    """The Newton Warp renderer cannot shade, so this combination must not resolve."""
    cfg = resolve_presets(cfg_cls(), {"newton_renderer", "simple_shading_full_mdl"})
    with pytest.raises(ValueError, match="simple_shading_full_mdl"):
        validate(cfg)


@pytest.mark.parametrize("cfg_cls", _ENV_CFG_CLASSES)
@pytest.mark.parametrize("renderer_preset", ["isaacsim_rtx", "ovrtx"])
def test_rtx_renderers_accept_simple_shading(cfg_cls, renderer_preset):
    """The RTX backends render every data type and must stay unaffected by the guard."""
    cfg = resolve_presets(cfg_cls(), {renderer_preset, "simple_shading_full_mdl"})
    validate(cfg)  # must not raise


@pytest.mark.parametrize("cfg_cls", _ENV_CFG_CLASSES)
def test_newton_renderer_accepts_supported_data_types(cfg_cls):
    """The guard must not reject data types the Newton Warp renderer does publish."""
    cfg = resolve_presets(cfg_cls(), {"newton_renderer", "rgb"})
    validate(cfg)  # must not raise


def test_direct_terminal_camera_read_preserves_survivor_history():
    """The direct task forwards the preview flag and keeps terminal image storage owned."""
    from types import SimpleNamespace

    import torch

    from isaaclab.utils.images import CameraFrameStack

    from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env import CartpoleCameraEnv

    env = object.__new__(CartpoleCameraEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        scene=SimpleNamespace(tiled_camera=SimpleNamespace(data_types=["rgb"])),
        write_image_to_file=False,
        observation_noise_model=None,
    )
    env.scene = SimpleNamespace(num_envs=2)
    env.extras = {}
    camera = torch.randint(0, 255, (2, 4, 4, 3), dtype=torch.uint8)
    env._tiled_camera = SimpleNamespace(data=SimpleNamespace(output={"rgb": SimpleNamespace(torch=camera)}))
    env._frames = CameraFrameStack(2, "cpu", 2, channel_first=True)
    env.joint_pos = torch.zeros(2, 2)
    env.joint_vel = torch.zeros(2, 2)
    env.cartpole = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=SimpleNamespace(torch=env.joint_pos.clone()),
            default_joint_vel=SimpleNamespace(torch=env.joint_vel.clone()),
        )
    )
    env._cart_dof_idx, env._pole_dof_idx = [0], [1]
    previous = camera.float() / 255
    previous = (previous - previous.mean((1, 2), keepdim=True)).permute(0, 3, 1, 2)
    env._compute_observations()
    camera.copy_(torch.randint_like(camera, 0, 255))
    successor = camera.float() / 255
    successor = (successor - successor.mean((1, 2), keepdim=True)).permute(0, 3, 1, 2)
    env._capture_final_obs()
    terminal = env.extras["final_obs"]["policy"]
    env._frames.reset([0])
    live = env._compute_observations()["policy"]
    expected = torch.cat((previous, successor), dim=1)
    torch.testing.assert_close(terminal, expected)
    torch.testing.assert_close(live[1], expected[1])
    torch.testing.assert_close(live[0], successor[0].repeat(2, 1, 1))
    camera.zero_()
    env._compute_observations()
    torch.testing.assert_close(terminal, expected)
