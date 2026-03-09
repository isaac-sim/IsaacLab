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

from isaaclab_tasks.direct.shadow_hand.shadow_hand_vision_env import (  # noqa: E402
    ShadowHandVisionEnv,
)
from isaaclab_tasks.direct.shadow_hand.shadow_hand_vision_env_cfg import (  # noqa: E402
    ShadowHandVisionBenchmarkEnvCfg,
    ShadowHandVisionEnvCfg,
)
from isaaclab_tasks.utils.hydra import collect_presets, resolve_preset_defaults  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Preset resolution — camera data types
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shadow_hand_vision_presets():
    """Collect all presets from ShadowHandVisionEnvCfg once for the module."""
    return collect_presets(ShadowHandVisionEnvCfg())


# ---------------------------------------------------------------------------
# Integration: camera output tensors must contain non-zero pixel values
# ---------------------------------------------------------------------------

_RENDER_CORRECTNESS_CASES = [
    # (renderer_preset, camera_preset, physics) — physics is "physx" or "newton"
    # ── PhysX physics (default) + IsaacRTX: supports all data types ──
    pytest.param(("isaacsim_rtx_renderer", "rgb", "physx"), id="physx-isaacsim_rtx-rgb"),
    pytest.param(("isaacsim_rtx_renderer", "depth", "physx"), id="physx-isaacsim_rtx-depth"),
    pytest.param(("isaacsim_rtx_renderer", "albedo", "physx"), id="physx-isaacsim_rtx-albedo"),
    pytest.param(
        ("isaacsim_rtx_renderer", "simple_shading_constant_diffuse", "physx"),
        id="physx-isaacsim_rtx-simple_shading_constant_diffuse",
    ),
    pytest.param(
        ("isaacsim_rtx_renderer", "simple_shading_diffuse_mdl", "physx"),
        id="physx-isaacsim_rtx-simple_shading_diffuse_mdl",
    ),
    pytest.param(
        ("isaacsim_rtx_renderer", "simple_shading_full_mdl", "physx"),
        id="physx-isaacsim_rtx-simple_shading_full_mdl",
    ),
    # ── PhysX physics + Warp: only rgb and depth are supported ──
    pytest.param(
        ("newton_renderer", "rgb", "physx"),
        id="physx-warp-rgb",
        marks=pytest.mark.xfail(strict=True, reason="Expected failure: known issue with Warp renderer + PhysX backend (see #TODO: update when fixed)"),
    ),
    pytest.param(
        ("newton_renderer", "depth", "physx"),
        id="physx-warp-depth",
        marks=pytest.mark.xfail(strict=True, reason="Expected failure: known issue with Warp renderer + PhysX backend (see #TODO: update when fixed)"),
    ),
    # ── Newton physics + Warp: Warp renderer is physics-backend agnostic ──
    pytest.param(
        ("newton_renderer", "rgb", "newton"),
        id="newton-warp-rgb",
        marks=pytest.mark.xfail(strict=True, reason="Expected failure: known issue with Warp renderer + Newton backend (see #TODO: update when fixed)"),
    ),
    pytest.param(
        ("newton_renderer", "depth", "newton"),
        id="newton-warp-depth",
        marks=pytest.mark.xfail(strict=True, reason="Expected failure: known issue with Warp renderer + Newton backend (see #TODO: update when fixed)"),
    ),
    # ── Newton physics + IsaacRTX: known incompatibility — produces empty frames ──
    # xfail(strict=True): if these ever pass the mark becomes a hard failure, prompting review.
    pytest.param(
        ("isaacsim_rtx_renderer", "rgb", "newton"),
        id="newton-isaacsim_rtx-rgb",
        marks=pytest.mark.xfail(strict=True, reason="Newton physics + IsaacRTX renderer produces empty frames"),
    ),
    pytest.param(
        ("isaacsim_rtx_renderer", "depth", "newton"),
        id="newton-isaacsim_rtx-depth",
        marks=pytest.mark.xfail(strict=True, reason="Newton physics + IsaacRTX renderer produces empty frames"),
    ),
    pytest.param(
        ("isaacsim_rtx_renderer", "albedo", "newton"),
        id="newton-isaacsim_rtx-albedo",
        marks=pytest.mark.xfail(strict=True, reason="Newton physics + IsaacRTX renderer produces empty frames"),
    ),
    pytest.param(
        ("isaacsim_rtx_renderer", "simple_shading_constant_diffuse", "newton"),
        id="newton-isaacsim_rtx-simple_shading_constant_diffuse",
        marks=pytest.mark.xfail(strict=True, reason="Newton physics + IsaacRTX renderer produces empty frames"),
    ),
    pytest.param(
        ("isaacsim_rtx_renderer", "simple_shading_diffuse_mdl", "newton"),
        id="newton-isaacsim_rtx-simple_shading_diffuse_mdl",
        marks=pytest.mark.xfail(strict=True, reason="Newton physics + IsaacRTX renderer produces empty frames"),
    ),
    pytest.param(
        ("isaacsim_rtx_renderer", "simple_shading_full_mdl", "newton"),
        id="newton-isaacsim_rtx-simple_shading_full_mdl",
        marks=pytest.mark.xfail(strict=True, reason="Newton physics + IsaacRTX renderer produces empty frames"),
    ),
    # ── OVRTX: disabled ── HDC_TODO: OVRTX test can be enabled?
    pytest.param(
        ("ovrtx_renderer", "rgb", "physx"),
        id="physx-ovrtx-rgb",
        marks=pytest.mark.skip(reason="OVRTX testing disabled"),
    ),
]


@pytest.fixture(params=_RENDER_CORRECTNESS_CASES)
def render_correctness_env(request, shadow_hand_vision_presets):
    """Build an env with the specified renderer+camera+physics combination, step once, yield, close.

    Function-scoped so each parametrized case creates and closes its own env sequentially.
    ``SimulationContext.clear_instance()`` (called by ``env.close()``) fully tears down the
    singleton, allowing a new env with a different physics backend to be created next.

    The shared ``shadow_hand_vision_presets`` fixture is deepcopied before mutation so that
    subsequent parametrized cases see clean preset configs.
    """
    renderer_preset, camera_preset, physics = request.param
    cfg = ShadowHandVisionBenchmarkEnvCfg()
    # Wire in the requested camera and renderer presets.
    camera_cfg = copy.deepcopy(shadow_hand_vision_presets["tiled_camera"][camera_preset])
    camera_cfg.renderer_cfg = copy.deepcopy(shadow_hand_vision_presets["tiled_camera.renderer_cfg"][renderer_preset])
    cfg.tiled_camera = camera_cfg
    # Apply Newton presets before resolve_preset_defaults so they are not overwritten by defaults.
    # Newton needs a specific solver config, a different robot USD, an articulation-based object,
    # and a stripped-down event cfg (no PhysX-specific material randomization).
    if physics == "newton":
        cfg.sim.physics = copy.deepcopy(shadow_hand_vision_presets["sim.physics"]["newton"])
        cfg.robot_cfg = copy.deepcopy(shadow_hand_vision_presets["robot_cfg"]["newton"])
        cfg.object_cfg = copy.deepcopy(shadow_hand_vision_presets["object_cfg"]["newton"])
        if "events" in shadow_hand_vision_presets:
            cfg.events = copy.deepcopy(shadow_hand_vision_presets["events"]["newton"])
    cfg = resolve_preset_defaults(cfg)
    cfg.scene.num_envs = 4
    cfg.feature_extractor.write_image_to_file = False
    env = ShadowHandVisionEnv(cfg)
    env.reset()
    actions = torch.zeros(cfg.scene.num_envs, env.action_space.shape[-1], device=env.device)
    env.step(actions)
    yield renderer_preset, camera_preset, physics, env
    env.close()


def test_camera_renders_not_empty(render_correctness_env):
    """Camera output must contain at least one non-zero pixel for every valid renderer+camera combo.

    Depth tensors may contain ``inf`` for background pixels (empty space). ``inf`` is replaced
    with 0 before checking ``max()``; a non-zero max confirms the renderer produced geometry pixels.

    The ``newton-isaacsim_rtx-rgb`` case is marked ``xfail(strict=True)``: Newton physics +
    IsaacRTX renderer is a known incompatibility that produces empty frames. If it ever starts
    passing, the strict xfail will surface it as a regression for review.
    """
    renderer_preset, camera_preset, physics, env = render_correctness_env
    print(f"[HDC] test_camera_renders_not_empty: renderer_preset: {renderer_preset}, camera_preset: {camera_preset}, physics: {physics}")
    label = f"{physics}-{renderer_preset}+{camera_preset}"
    camera_output = env._tiled_camera.data.output
    assert len(camera_output) > 0, f"[{label}] Camera produced no output tensors at all."
    for dt, tensor in camera_output.items():
        finite = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
        assert finite.max() > 0, (
            f"[{label}] Camera output '{dt}' is all zeros or all inf "
            f"after stepping. Tensor shape: {tensor.shape}, dtype: {tensor.dtype}."
        )
