# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for rendering correctness."""

# Launch Isaac Sim Simulator first.
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import copy  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.direct.cartpole.cartpole_camera_env import (  # noqa: E402
    CartpoleCameraEnv,
)
from isaaclab_tasks.direct.cartpole.cartpole_camera_presets_env_cfg import (  # noqa: E402
    CartpoleCameraPresetsEnvCfg,
)
from isaaclab_tasks.direct.shadow_hand.shadow_hand_vision_env import (  # noqa: E402
    ShadowHandVisionEnv,
)
from isaaclab_tasks.direct.shadow_hand.shadow_hand_vision_env_cfg import (  # noqa: E402
    ShadowHandVisionBenchmarkEnvCfg,
    ShadowHandVisionEnvCfg,
)
from isaaclab_tasks.utils.hydra import collect_presets, resolve_preset_defaults  # noqa: E402


@pytest.fixture(scope="module")
def shadow_hand_vision_presets():
    """Collect all presets from ShadowHandVisionEnvCfg once for the module."""
    return collect_presets(ShadowHandVisionEnvCfg())


@pytest.fixture(scope="module")
def cartpole_presets():
    """Collect all presets from CartpoleCameraPresetsEnvCfg once for the module."""
    return collect_presets(CartpoleCameraPresetsEnvCfg())


# Skip reason for ovrtx_renderer
_OVRTX_SKIP = "OVRTX testing disabled"

# (physics_backend, renderer, data_type) shared by both envs
_SHARED_RENDER_CORRECTNESS_CASES = [
    # physx + isaacsim_rtx_renderer
    pytest.param(("physx", "isaacsim_rtx_renderer", "rgb"), id="physx-isaacsim_rtx-rgb"),
    pytest.param(("physx", "isaacsim_rtx_renderer", "albedo"), id="physx-isaacsim_rtx-albedo"),
    pytest.param(("physx", "isaacsim_rtx_renderer", "depth"), id="physx-isaacsim_rtx-depth"),
    pytest.param(
        ("physx", "isaacsim_rtx_renderer", "simple_shading_constant_diffuse"),
        id="physx-isaacsim_rtx-simple_shading_constant_diffuse",
    ),
    pytest.param(
        ("physx", "isaacsim_rtx_renderer", "simple_shading_diffuse_mdl"),
        id="physx-isaacsim_rtx-simple_shading_diffuse_mdl",
    ),
    pytest.param(
        ("physx", "isaacsim_rtx_renderer", "simple_shading_full_mdl"),
        id="physx-isaacsim_rtx-simple_shading_full_mdl",
    ),
    # physx + newton_renderer (warp)
    pytest.param(("physx", "newton_renderer", "rgb"), id="physx-newton_warp-rgb"),
    pytest.param(("physx", "newton_renderer", "depth"), id="physx-newton_warp-depth"),
    # newton + isaacsim_rtx_renderer
    pytest.param(("newton", "isaacsim_rtx_renderer", "rgb"), id="newton-isaacsim_rtx-rgb"),
    pytest.param(("newton", "isaacsim_rtx_renderer", "albedo"), id="newton-isaacsim_rtx-albedo"),
    pytest.param(
        ("newton", "isaacsim_rtx_renderer", "depth"),
        id="newton-isaacsim_rtx-depth",
    ),
    pytest.param(
        ("newton", "isaacsim_rtx_renderer", "simple_shading_constant_diffuse"),
        id="newton-isaacsim_rtx-simple_shading_constant_diffuse",
    ),
    pytest.param(
        ("newton", "isaacsim_rtx_renderer", "simple_shading_diffuse_mdl"),
        id="newton-isaacsim_rtx-simple_shading_diffuse_mdl",
    ),
    pytest.param(
        ("newton", "isaacsim_rtx_renderer", "simple_shading_full_mdl"),
        id="newton-isaacsim_rtx-simple_shading_full_mdl",
    ),
    # newton + newton_renderer (warp)
    pytest.param(("newton", "newton_renderer", "rgb"), id="newton-newton_warp-rgb"),
    pytest.param(
        ("newton", "newton_renderer", "depth"),
        id="newton-newton_warp-depth",
    ),
    # newton + ovrtx_renderer
    pytest.param(
        ("newton", "ovrtx_renderer", "rgb"),
        id="newton-ovrtx-rgb",
        marks=pytest.mark.skip(reason=_OVRTX_SKIP),
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "albedo"),
        id="newton-ovrtx-albedo",
        marks=pytest.mark.skip(reason=_OVRTX_SKIP),
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "depth"),
        id="newton-ovrtx-depth",
        marks=pytest.mark.skip(reason=_OVRTX_SKIP),
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "simple_shading_constant_diffuse"),
        id="newton-ovrtx-simple_shading_constant_diffuse",
        marks=pytest.mark.skip(reason=_OVRTX_SKIP),
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "simple_shading_diffuse_mdl"),
        id="newton-ovrtx-simple_shading_diffuse_mdl",
        marks=pytest.mark.skip(reason=_OVRTX_SKIP),
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "simple_shading_full_mdl"),
        id="newton-ovrtx-simple_shading_full_mdl",
        marks=pytest.mark.skip(reason=_OVRTX_SKIP),
    ),
]


def _assert_camera_renders_not_empty(env_name, physics_backend, renderer, data_type, env):
    """Shared assertion: camera output has at least one non-zero pixel per data type."""
    label = f"{env_name}-{physics_backend}-{renderer}+{data_type}"
    camera_output = env._tiled_camera.data.output
    assert len(camera_output) > 0, f"[{label}] Camera produced no output tensors at all."
    for dt, tensor in camera_output.items():
        finite = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
        assert finite.max() > 0, (
            f"[{label}] Camera output '{dt}' is all zeros or all inf "
            f"after stepping. Tensor shape: {tensor.shape}, dtype: {tensor.dtype}."
        )


@pytest.fixture(params=_SHARED_RENDER_CORRECTNESS_CASES)
def shadow_hand_env(request, shadow_hand_vision_presets):
    """Build Shadow Hand vision env for (physics_backend, renderer, data_type), step once, yield, close.

    Function-scoped so each parametrized case creates and closes its own env sequentially.
    """
    physics_backend, renderer, data_type = request.param
    presets = shadow_hand_vision_presets
    camera_cfg = copy.deepcopy(presets["tiled_camera"][data_type])
    camera_cfg.renderer_cfg = copy.deepcopy(presets["tiled_camera.renderer_cfg"][renderer])
    env_cfg = ShadowHandVisionBenchmarkEnvCfg()  # HDC_TODO: use ShadowHandVisionEnvCfg with feature_extractor disabled explicitly
    env_cfg.tiled_camera = camera_cfg
    if physics_backend == "newton":
        env_cfg.sim.physics = copy.deepcopy(presets["sim.physics"]["newton"])
        env_cfg.robot_cfg = copy.deepcopy(presets["robot_cfg"]["newton"])
        env_cfg.object_cfg = copy.deepcopy(presets["object_cfg"]["newton"])
        if "events" in presets:
            env_cfg.events = copy.deepcopy(presets["events"]["newton"])
    env_cfg = resolve_preset_defaults(env_cfg)
    env_cfg.scene.num_envs = 4
    env = ShadowHandVisionEnv(env_cfg)
    env.reset()
    actions = torch.zeros(env_cfg.scene.num_envs, env.action_space.shape[-1], device=env.device)
    env.step(actions)
    yield physics_backend, renderer, data_type, env
    env.close()


@pytest.fixture(params=_SHARED_RENDER_CORRECTNESS_CASES)
def cartpole_camera_env(request, cartpole_presets):
    """Build Cartpole camera env for (physics_backend, renderer, data_type), step once, yield, close.

    Function-scoped so each parametrized case creates and closes its own env sequentially.
    """
    physics_backend, renderer, data_type = request.param
    presets = cartpole_presets
    camera_cfg = copy.deepcopy(presets["tiled_camera"][data_type])
    camera_cfg.renderer_cfg = copy.deepcopy(presets["tiled_camera.renderer_cfg"][renderer])
    env_cfg = CartpoleCameraPresetsEnvCfg()
    env_cfg.tiled_camera = camera_cfg
    if physics_backend == "newton":
        if "robot_cfg" in presets:
            env_cfg.robot_cfg = copy.deepcopy(presets["robot_cfg"]["newton"])
        if "object_cfg" in presets:
            env_cfg.object_cfg = copy.deepcopy(presets["object_cfg"]["newton"])
        if "events" in presets:
            env_cfg.events = copy.deepcopy(presets["events"]["newton"])
    env_cfg = resolve_preset_defaults(env_cfg)
    env_cfg.scene.num_envs = 4
    env = CartpoleCameraEnv(env_cfg)
    env.reset()
    actions = torch.zeros(env_cfg.scene.num_envs, env.action_space.shape[-1], device=env.device)
    env.step(actions)
    yield physics_backend, renderer, data_type, env
    env.close()


def test_camera_renders_not_empty_shadow_hand(shadow_hand_env):
    """Camera output must contain at least one non-zero pixel (Shadow Hand vision env)."""
    physics_backend, renderer, data_type, env = shadow_hand_env
    _assert_camera_renders_not_empty("shadow_hand", physics_backend, renderer, data_type, env)


def test_camera_renders_not_empty_cartpole_camera(cartpole_camera_env):
    """Camera output must contain at least one non-zero pixel (Cartpole camera env)."""
    physics_backend, renderer, data_type, env = cartpole_camera_env
    _assert_camera_renders_not_empty("cartpole", physics_backend, renderer, data_type, env)
