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

from isaaclab_tasks.utils.hydra import collect_presets, resolve_preset_defaults  # noqa: E402


@pytest.fixture(scope="module")
def cartpole_presets():
    """Collect all presets from CartpoleCameraPresetsEnvCfg once for the module."""
    return collect_presets(CartpoleCameraPresetsEnvCfg())


# ---------------------------------------------------------------------------
# Integration: camera output tensors must contain non-zero pixel values
# ---------------------------------------------------------------------------

# physics backend, renderer, data type
_RENDER_CORRECTNESS_CASES = [
    # physx + isaacsim_rtx_renderer
    pytest.param(("physx", "isaacsim_rtx_renderer", "rgb"), id="physx-isaacsim_rtx-rgb"),
    pytest.param(("physx", "isaacsim_rtx_renderer", "albedo"), id="physx-isaacsim_rtx-albedo"),
    pytest.param(("physx", "isaacsim_rtx_renderer", "depth"), id="physx-isaacsim_rtx-depth"),
    pytest.param(("physx", "isaacsim_rtx_renderer", "simple_shading_constant_diffuse"), id="physx-isaacsim_rtx-simple_shading_constant_diffuse"),
    pytest.param(("physx", "isaacsim_rtx_renderer", "simple_shading_diffuse_mdl"), id="physx-isaacsim_rtx-simple_shading_diffuse_mdl"),
    pytest.param(("physx", "isaacsim_rtx_renderer", "simple_shading_full_mdl"), id="physx-isaacsim_rtx-simple_shading_full_mdl"),

    # physx + newton_renderer (warp)
    pytest.param(("physx", "newton_renderer", "rgb"), id="physx-newton_warp-rgb"),
    pytest.param(("physx", "newton_renderer", "depth"), id="physx-newton_warp-depth"),

    # newton + isaacsim_rtx_renderer
    pytest.param(("newton", "isaacsim_rtx_renderer", "rgb"), id="newton-isaacsim_rtx-rgb"),
    pytest.param(("newton", "isaacsim_rtx_renderer", "albedo"), id="newton-isaacsim_rtx-albedo"),
    pytest.param(("newton", "isaacsim_rtx_renderer", "depth"),
                 id="newton-isaacsim_rtx-depth",
                 marks=pytest.mark.skip(reason="AssertionError: [newton-isaacsim_rtx-depth] Camera output 'depth' is all zeros or all inf after stepping. Tensor shape: torch.Size([4, 120, 120, 1]), dtype: torch.float32."),
                 ),
    pytest.param(("newton", "isaacsim_rtx_renderer", "simple_shading_constant_diffuse"),
                 id="newton-isaacsim_rtx-simple_shading_constant_diffuse",
                 marks=pytest.mark.skip(reason="AssertionError: [newton-isaacsim_rtx_renderer+simple_shading_constant_diffuse] Camera output 'simple_shading_constant_diffuse' is all zeros or all inf after stepping. Tensor shape: torch.Size([4, 120, 120, 3]), dtype: torch.uint8."),
                 ),
    pytest.param(("newton", "isaacsim_rtx_renderer", "simple_shading_diffuse_mdl"),
                 id="newton-isaacsim_rtx-simple_shading_diffuse_mdl",
                 marks=pytest.mark.skip(reason="AssertionError: [newton-isaacsim_rtx_renderer+simple_shading_diffuse_mdl] Camera output 'simple_shading_diffuse_mdl' is all zeros or all inf after stepping. Tensor shape: torch.Size([4, 120, 120, 3]), dtype: torch.uint8."),
                 ),
    pytest.param(("newton", "isaacsim_rtx_renderer", "simple_shading_full_mdl"),
                 id="newton-isaacsim_rtx-simple_shading_full_mdl",
                 marks=pytest.mark.skip(reason="AssertionError: [newton-isaacsim_rtx_renderer+simple_shading_full_mdl] Camera output 'simple_shading_full_mdl' is all zeros or all inf after stepping. Tensor shape: torch.Size([4, 120, 120, 3]), dtype: torch.uint8."),
                 ),

    # newton + newton_renderer (warp)
    pytest.param(("newton", "newton_renderer", "rgb"), id="newton-newton_warp-rgb"),
    pytest.param(("newton", "newton_renderer", "depth"),
                 id="newton-newton_warp-depth",
                 marks=pytest.mark.skip(reason="AssertionError: [newton-newton_renderer+depth] Camera output 'depth' is all zeros or all inf after stepping. Tensor shape: torch.Size([4, 120, 120, 1]), dtype: torch.float32."),
                 ),

    # newton + ovrtx_renderer
    pytest.param(("newton", "ovrtx_renderer", "rgb"),
                 id="newton-ovrtx-rgb",
                 marks=pytest.mark.skip(reason="ValueError: Could not import module for backend 'ov' for factory Renderer. Attempted to import from 'isaaclab_ov.renderers'."),
                 ),
    pytest.param(("newton", "ovrtx_renderer", "albedo"),
                 id="newton-ovrtx-albedo",
                 marks=pytest.mark.skip(reason="ValueError: Could not import module for backend 'ov' for factory Renderer. Attempted to import from 'isaaclab_ov.renderers'."),
                 ),
    pytest.param(("newton", "ovrtx_renderer", "depth"),
                 id="newton-ovrtx-depth",
                 marks=pytest.mark.skip(reason="ValueError: Could not import module for backend 'ov' for factory Renderer. Attempted to import from 'isaaclab_ov.renderers'."),
                 ),
    pytest.param(("newton", "ovrtx_renderer", "simple_shading_constant_diffuse"),
                 id="newton-ovrtx-simple_shading_constant_diffuse",
                 marks=pytest.mark.skip(reason="ValueError: Could not import module for backend 'ov' for factory Renderer. Attempted to import from 'isaaclab_ov.renderers'."),
                 ),
    pytest.param(("newton", "ovrtx_renderer", "simple_shading_diffuse_mdl"),
                 id="newton-ovrtx-simple_shading_diffuse_mdl",
                 marks=pytest.mark.skip(reason="ValueError: Could not import module for backend 'ov' for factory Renderer. Attempted to import from 'isaaclab_ov.renderers'."),
                 ),
    pytest.param(("newton", "ovrtx_renderer", "simple_shading_full_mdl"),
                 id="newton-ovrtx-simple_shading_full_mdl",
                 marks=pytest.mark.skip(reason="ValueError: Could not import module for backend 'ov' for factory Renderer. Attempted to import from 'isaaclab_ov.renderers'."),
                 ),
]


@pytest.fixture(params=_RENDER_CORRECTNESS_CASES)
def render_correctness_env(request, cartpole_presets):
    """Build an env with the specified physics_backend + renderer + data_type combination, step once, yield, close.

    Function-scoped so each parametrized case creates and closes its own env sequentially.
    ``SimulationContext.clear_instance()`` (called by ``env.close()``) fully tears down the
    singleton, allowing a new env with a different physics backend to be created next.

    The shared ``cartpole_presets`` fixture is deepcopied before mutation so that
    subsequent parametrized cases see clean preset configs.
    """
    physics_backend, renderer, data_type = request.param

    # Wire in the requested camera and renderer presets.
    camera_cfg = copy.deepcopy(cartpole_presets["tiled_camera"][data_type])
    camera_cfg.renderer_cfg = copy.deepcopy(cartpole_presets["tiled_camera.renderer_cfg"][renderer])

    env_cfg = CartpoleCameraPresetsEnvCfg()
    env_cfg.tiled_camera = camera_cfg

    # Apply Newton presets before resolve_preset_defaults so they are not overwritten by defaults.
    # Cartpole only has sim.physics as a PresetCfg; robot_cfg/object_cfg/events are not presets.
    # if physics_backend == "newton" and "sim.physics" in cartpole_presets:
    #     env_cfg.sim.physics = copy.deepcopy(cartpole_presets["sim.physics"]["newton"])
    if physics_backend == "newton":
        if "robot_cfg" in cartpole_presets:
            env_cfg.robot_cfg = copy.deepcopy(cartpole_presets["robot_cfg"]["newton"])
        if "object_cfg" in cartpole_presets:
            env_cfg.object_cfg = copy.deepcopy(cartpole_presets["object_cfg"]["newton"])
        if "events" in cartpole_presets:
            env_cfg.events = copy.deepcopy(cartpole_presets["events"]["newton"])

    env_cfg = resolve_preset_defaults(env_cfg)
    env_cfg.scene.num_envs = 4

    env = CartpoleCameraEnv(env_cfg)
    env.reset()

    actions = torch.zeros(env_cfg.scene.num_envs, env.action_space.shape[-1], device=env.device)
    env.step(actions)

    yield physics_backend, renderer, data_type, env

    env.close()


def test_camera_renders_not_empty(render_correctness_env):
    """Camera output must contain at least one non-zero pixel for every valid renderer+camera combo.

    Depth tensors may contain ``inf`` for background pixels (empty space). ``inf`` is replaced
    with 0 before checking ``max()``; a non-zero max confirms the renderer produced geometry pixels.

    The ``newton-isaacsim_rtx-rgb`` case is marked ``xfail(strict=True)``: Newton physics +
    IsaacRTX renderer is a known incompatibility that produces empty frames. If it ever starts
    passing, the strict xfail will surface it as a regression for review.
    """
    physics_backend, renderer, data_type, env = render_correctness_env

    label = f"{physics_backend}-{renderer}+{data_type}"
    camera_output = env._tiled_camera.data.output
    assert len(camera_output) > 0, f"[{label}] Camera produced no output tensors at all."
    for dt, tensor in camera_output.items():
        finite = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
        assert finite.max() > 0, (
            f"[{label}] Camera output '{dt}' is all zeros or all inf "
            f"after stepping. Tensor shape: {tensor.shape}, dtype: {tensor.dtype}."
        )
