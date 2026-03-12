# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for rendering correctness.

Each test builds an environment with a given (physics_backend, renderer, data_type),
steps once, then asserts that camera outputs are not blank (at least one non-zero
pixel). Env-specific fixtures use parametrized combinations; a separate test
covers a list of registered task IDs that use camera-based observations.
"""

# Launch Isaac Sim Simulator first.
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

from typing import Any

import gymnasium as gym  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs.utils.spaces import sample_space  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from isaaclab_tasks.utils.hydra import (  # noqa: E402
    apply_overrides,
    collect_presets,
    parse_overrides,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def cleanup_simulation_context():
    """Fixture to clear SimulationContext after each test.

    SimulationContext is a singleton; tests that create envs leave it set. Without
    cleanup, later tests can see stale context or fail when the instance is
    reused. The fixture runs after every test and calls clear_instance() so each
    test runs with a clean simulation context and tests stay isolated.
    """
    yield

    SimulationContext.clear_instance()


# ---------------------------------------------------------------------------
# Parametrization: (physics_backend, renderer, data_type)
# ---------------------------------------------------------------------------

_PHYSICS_RENDERER_AOV_COMBINATIONS = [
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
    # physx + newton_renderer (warp) — skipped due to known issues
    pytest.param(
        ("physx", "newton_renderer", "rgb"),
        id="physx-newton_warp-rgb",
        marks=pytest.mark.skip(reason="physx + newton_renderer has known issues"),
    ),
    pytest.param(
        ("physx", "newton_renderer", "depth"),
        id="physx-newton_warp-depth",
        marks=pytest.mark.skip(reason="physx + newton_renderer has known issues"),
    ),
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
        marks=pytest.mark.skip(reason="OVRTX testing disabled"),
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "albedo"),
        id="newton-ovrtx-albedo",
        marks=pytest.mark.skip(reason="OVRTX testing disabled"),
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "depth"),
        id="newton-ovrtx-depth",
        marks=pytest.mark.skip(reason="OVRTX testing disabled"),
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "simple_shading_constant_diffuse"),
        id="newton-ovrtx-simple_shading_constant_diffuse",
        marks=pytest.mark.skip(reason="OVRTX testing disabled"),
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "simple_shading_diffuse_mdl"),
        id="newton-ovrtx-simple_shading_diffuse_mdl",
        marks=pytest.mark.skip(reason="OVRTX testing disabled"),
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "simple_shading_full_mdl"),
        id="newton-ovrtx-simple_shading_full_mdl",
        marks=pytest.mark.skip(reason="OVRTX testing disabled"),
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_overrides_to_env_cfg(env_cfg: Any, override_args: list[str]) -> Any:
    """Apply override args to env_cfg using parse_overrides and apply_overrides.

    Args:
        env_cfg: Environment config to mutate (supports :meth:`to_dict`).
        override_args: List of override strings (e.g. ``["presets=physx,isaacsim_rtx_renderer,rgb"]``).

    Returns:
        The resolved env_cfg (possibly a different object if root preset was applied).
    """
    presets = {"env": collect_presets(env_cfg)}
    global_presets, preset_sel, preset_scalar, _ = parse_overrides(override_args, presets)
    hydra_cfg = {"env": env_cfg.to_dict()}
    env_cfg, _ = apply_overrides(env_cfg, None, hydra_cfg, global_presets, preset_sel, preset_scalar, presets)
    return env_cfg


def _assert_camera_outputs_are_not_blank(label: str, camera_outputs: dict[str, dict[str, torch.Tensor]]) -> None:
    """Assert each camera output has at least one non-zero pixel.

    Args:
        label: Label for error messages.
        camera_outputs: Nested dict: sensor name -> {data_type -> tensor}.
    """
    assert len(camera_outputs) > 0, f"[{label}] No camera outputs; env may have no cameras or wrong structure."
    for sensor_name, output in camera_outputs.items():
        for data_type, tensor in output.items():
            invalid = torch.logical_or(torch.isinf(tensor), torch.isnan(tensor))
            corrected = torch.where(invalid, torch.zeros_like(tensor), tensor)
            assert corrected.max() > 0, (
                f"[{label}] Sensor '{sensor_name}' output '{data_type}' has no non-zero pixels. "
                f"Shape: {corrected.shape}, dtype: {corrected.dtype}."
            )


def _collect_camera_outputs(env: object) -> dict[str, dict[str, torch.Tensor]]:
    """Collect camera outputs from env.scene.sensors.

    Args:
        env: Gymnasium env (or any object with optional unwrapped.scene.sensors).

    Returns:
        Nested dict: sensor name -> {data_type -> tensor} for non-empty tensor outputs.
    """
    base = getattr(env, "unwrapped", env)
    out = {}

    scene = getattr(base, "scene", None)
    if scene is not None:
        sensors = getattr(scene, "sensors", None)
        if sensors is not None:
            for name, sensor in sensors.items():
                data = getattr(sensor, "data", None)
                output = getattr(data, "output", None) if data is not None else None
                if not isinstance(output, dict):
                    continue

                # Collect only tensor entries (ignore empty or lazy-unfilled)
                tensor_output = {k: v for k, v in output.items() if isinstance(v, torch.Tensor) and v.numel() > 0}
                if tensor_output:
                    out[name] = tensor_output

    return out


# ---------------------------------------------------------------------------
# Shadow Hand vision env
# ---------------------------------------------------------------------------


@pytest.fixture(params=_PHYSICS_RENDERER_AOV_COMBINATIONS)
def shadow_hand_env(request):
    """Build Shadow Hand vision env for (physics_backend, renderer, data_type); step once, yield, close."""
    from isaaclab_tasks.direct.shadow_hand.shadow_hand_vision_env import ShadowHandVisionEnv
    from isaaclab_tasks.direct.shadow_hand.shadow_hand_vision_env_cfg import ShadowHandVisionEnvCfg

    physics_backend, renderer, data_type = request.param

    override_args = [f"presets={physics_backend},{renderer},{data_type}"]

    env_cfg = ShadowHandVisionEnvCfg()
    env_cfg = _apply_overrides_to_env_cfg(env_cfg, override_args)

    env_cfg.scene.num_envs = 4

    if data_type == "depth":
        # Disable CNN forward pass as it cannot be meaningfully trained from depth alone and will raise a ValueError.
        env_cfg.feature_extractor.enabled = False

    env = None
    try:
        env = ShadowHandVisionEnv(env_cfg)
        env.reset()
        actions = torch.zeros(env_cfg.scene.num_envs, env.action_space.shape[-1], device=env.device)
        env.step(actions)
        yield physics_backend, renderer, data_type, env
    finally:
        if env is not None:
            env.close()


def test_camera_outputs_are_not_blank_for_shadow_hand(shadow_hand_env):
    """Camera output must contain at least one non-zero pixel (Shadow Hand vision env)."""
    physics_backend, renderer, data_type, env = shadow_hand_env
    label = f"shadow_hand-{physics_backend}-{renderer}+{data_type}"
    _assert_camera_outputs_are_not_blank(label, {"tiled_camera": env._tiled_camera.data.output})


# ---------------------------------------------------------------------------
# Cartpole camera env
# ---------------------------------------------------------------------------


@pytest.fixture(params=_PHYSICS_RENDERER_AOV_COMBINATIONS)
def cartpole_env(request):
    """Build Cartpole camera env for (physics_backend, renderer, data_type); step once, yield, close."""
    from isaaclab_tasks.direct.cartpole.cartpole_camera_env import CartpoleCameraEnv
    from isaaclab_tasks.direct.cartpole.cartpole_camera_presets_env_cfg import CartpoleCameraPresetsEnvCfg

    physics_backend, renderer, data_type = request.param

    override_args = [f"presets={physics_backend},{renderer},{data_type}"]

    env_cfg = CartpoleCameraPresetsEnvCfg()
    env_cfg = _apply_overrides_to_env_cfg(env_cfg, override_args)

    env_cfg.scene.num_envs = 4

    env = None
    try:
        env = CartpoleCameraEnv(env_cfg)
        env.reset()
        actions = torch.zeros(env_cfg.scene.num_envs, env.action_space.shape[-1], device=env.device)
        env.step(actions)
        yield physics_backend, renderer, data_type, env
    finally:
        if env is not None:
            env.close()


def test_camera_outputs_are_not_blank_for_cartpole(cartpole_env):
    """Camera output must contain at least one non-zero pixel (Cartpole camera env)."""
    physics_backend, renderer, data_type, env = cartpole_env
    label = f"cartpole-{physics_backend}-{renderer}+{data_type}"
    _assert_camera_outputs_are_not_blank(label, {"tiled_camera": env._tiled_camera.data.output})


# ---------------------------------------------------------------------------
# Dexsuite Kuka-Allegro Lift (single camera)
# ---------------------------------------------------------------------------


@pytest.fixture(params=_PHYSICS_RENDERER_AOV_COMBINATIONS)
def dexsuite_kuka_allegro_lift_env(request):
    """Build Dexsuite Kuka-Allegro Lift (single camera) for backend/renderer/data_type; step once, yield, close."""
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg import (
        DexsuiteKukaAllegroLiftEnvCfg,
    )

    physics_backend, renderer, data_type = request.param

    # Dexsuite data type has explicit resolution suffix (64, 128, 256). We only test 64x64.
    override_args = [f"presets={physics_backend},{renderer},{data_type}64,single_camera,cube"]

    env_cfg = DexsuiteKukaAllegroLiftEnvCfg()
    env_cfg = _apply_overrides_to_env_cfg(env_cfg, override_args)

    env_cfg.scene.num_envs = 4

    env = None
    try:
        env = ManagerBasedRLEnv(env_cfg)
        env.reset()
        actions = torch.zeros(env_cfg.scene.num_envs, env.action_space.shape[-1], device=env.device)
        env.step(actions)
        yield physics_backend, renderer, data_type, env
    finally:
        if env is not None:
            env.close()


def test_camera_outputs_are_not_blank_for_dexsuite_kuka_allegro_lift(dexsuite_kuka_allegro_lift_env):
    """Camera output must contain at least one non-zero pixel (Dexsuite Kuka-Allegro Lift, single camera)."""
    physics_backend, renderer, data_type, env = dexsuite_kuka_allegro_lift_env
    label = f"dexsuite_kuka_allegro_lift-{physics_backend}-{renderer}+{data_type}"
    _assert_camera_outputs_are_not_blank(label, {"base_camera": env.scene.sensors["base_camera"].data.output})


# ---------------------------------------------------------------------------
# Registered tasks (camera-based observations)
# ---------------------------------------------------------------------------

# Task IDs that expose camera/tiled_camera image observations; each is validated for non-blank rendering.
_RENDER_CORRECTNESS_TASK_IDS = [
    "Isaac-Cartpole-RGB-Camera-Direct-v0",
    "Isaac-Cartpole-Albedo-Camera-Direct-v0",
    "Isaac-Cartpole-SimpleShading-Constant-Camera-Direct-v0",
    "Isaac-Cartpole-SimpleShading-Diffuse-Camera-Direct-v0",
    "Isaac-Cartpole-SimpleShading-Full-Camera-Direct-v0",
    "Isaac-Cartpole-Depth-Camera-Direct-v0",
    "Isaac-Cartpole-Camera-Presets-Direct-v0",
    "Isaac-Cartpole-Camera-Showcase-Box-Box-Direct-v0",
    "Isaac-Cartpole-Camera-Showcase-Box-Discrete-Direct-v0",
    "Isaac-Cartpole-Camera-Showcase-Box-MultiDiscrete-Direct-v0",
    "Isaac-Cartpole-Camera-Showcase-Dict-Box-Direct-v0",
    "Isaac-Cartpole-Camera-Showcase-Dict-Discrete-Direct-v0",
    "Isaac-Cartpole-Camera-Showcase-Dict-MultiDiscrete-Direct-v0",
    "Isaac-Cartpole-Camera-Showcase-Tuple-Box-Direct-v0",
    "Isaac-Cartpole-Camera-Showcase-Tuple-Discrete-Direct-v0",
    "Isaac-Cartpole-Camera-Showcase-Tuple-MultiDiscrete-Direct-v0",
    "Isaac-Repose-Cube-Shadow-Vision-Direct-v0",
    "Isaac-Cartpole-RGB-v0",
    "Isaac-Cartpole-Depth-v0",
    "Isaac-Cartpole-RGB-ResNet18-v0",
    "Isaac-Cartpole-RGB-TheiaTiny-v0",
]


@pytest.mark.parametrize("task_id", _RENDER_CORRECTNESS_TASK_IDS)
def test_camera_outputs_are_not_blank_for_registered_task(task_id):
    """Camera output must be non-empty for each registered task with camera-based observations."""
    env = None
    try:
        env_cfg = parse_env_cfg(task_id, num_envs=4)
        env = gym.make(task_id, cfg=env_cfg)
        unwrapped: Any = env.unwrapped
        sim = getattr(unwrapped, "sim", None)
        if sim is not None:
            sim._app_control_on_stop_handle = None

        env.reset()

        num_envs = getattr(unwrapped, "num_envs", 4)
        device = getattr(unwrapped, "device", None)

        if getattr(unwrapped, "possible_agents", None):
            action_spaces = getattr(unwrapped, "action_spaces", {})
            actions = {
                agent: sample_space(
                    action_spaces[agent],
                    device=device,
                    batch_size=num_envs,
                    fill_value=0,
                )
                for agent in unwrapped.possible_agents
            }
        else:
            actions = sample_space(
                getattr(unwrapped, "single_action_space", None),
                device=device,
                batch_size=num_envs,
                fill_value=0,
            )

        env.step(actions)

        camera_outputs = _collect_camera_outputs(env)
        _assert_camera_outputs_are_not_blank(task_id, camera_outputs)
    finally:
        if env is not None:
            env.close()
