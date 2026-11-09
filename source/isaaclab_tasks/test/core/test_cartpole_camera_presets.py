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


def test_direct_camera_observation_forwards_preview(monkeypatch):
    """A task preview must leave the next live camera frame stack unchanged."""
    from types import SimpleNamespace

    import torch

    from isaaclab.utils.images import CameraFrameStack

    from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env import CartpoleCameraEnv
    from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv

    monkeypatch.setattr(CartpoleEnv, "_get_observations", lambda self: {"policy": torch.empty(0)})
    env = object.__new__(CartpoleCameraEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        scene=SimpleNamespace(tiled_camera=SimpleNamespace(data_types=["depth"])),
        write_image_to_file=False,
    )
    camera = torch.ones(1, 1, 1, 1)
    env._tiled_camera = SimpleNamespace(data=SimpleNamespace(output={"depth": SimpleNamespace(torch=camera)}))
    env._frames = CameraFrameStack(1, "cpu", 2, channel_first=True)
    env._get_observations()
    camera.fill_(2)
    env._is_observation_preview = True
    env._get_observations()
    env._is_observation_preview = False
    assert env._get_observations()["policy"].flatten().tolist() == [1, 2]
