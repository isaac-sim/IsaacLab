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

import copy  # noqa: E402

import gymnasium as gym  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.envs.utils.spaces import sample_space  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

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
    ShadowHandVisionEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg import (  # noqa: E402
    DexsuiteKukaAllegroLiftEnvCfg,
)
from isaaclab_tasks.utils.hydra import collect_presets, resolve_preset_defaults  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OVRTX_SKIP = "OVRTX testing disabled"


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_camera_outputs_are_not_blank(label_prefix: str, camera_outputs: dict[str, dict[str, torch.Tensor]]) -> None:
    """Assert each camera output has at least one non-zero pixel (no all-zero or all-inf).

    Args:
        label_prefix: Prefix for error messages (e.g. task_id or "env-renderer+data_type").
        camera_outputs: Nested dict: sensor name -> {data_type -> tensor}.
    """
    assert len(camera_outputs) > 0, f"[{label_prefix}] No camera outputs; env may have no cameras or wrong structure."
    for sensor_name, output in camera_outputs.items():
        for data_type, tensor in output.items():
            finite = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
            assert finite.max() > 0, (
                f"[{label_prefix}] Sensor '{sensor_name}' output '{data_type}' is all zeros or all inf. "
                f"Shape: {tensor.shape}, dtype: {tensor.dtype}."
            )


def _collect_camera_outputs(env) -> dict[str, dict[str, torch.Tensor]]:
    """Collect camera outputs from env.scene.sensors (name -> {data_type: tensor})."""
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


@pytest.fixture(scope="module")
def shadow_hand_vision_presets():
    """Collect all presets from ShadowHandVisionEnvCfg once for the module."""
    return collect_presets(ShadowHandVisionEnvCfg())


@pytest.fixture(params=_PHYSICS_RENDERER_AOV_COMBINATIONS)
def shadow_hand_env(request, shadow_hand_vision_presets):
    """Build Shadow Hand vision env for (physics_backend, renderer, data_type); step once, yield, close."""
    physics_backend, renderer, data_type = request.param

    if physics_backend == "physx" and renderer == "newton_renderer":
        pytest.skip("The preset is not supported by Shadow Hand vision env")

    presets = shadow_hand_vision_presets
    camera_cfg = copy.deepcopy(presets["tiled_camera"][data_type])
    camera_cfg.renderer_cfg = copy.deepcopy(presets["tiled_camera.renderer_cfg"][renderer])
    env_cfg = ShadowHandVisionEnvCfg()
    env_cfg.tiled_camera = camera_cfg
    if physics_backend == "newton":
        env_cfg.sim.physics = copy.deepcopy(presets["sim.physics"]["newton"])
        env_cfg.robot_cfg = copy.deepcopy(presets["robot_cfg"]["newton"])
        env_cfg.object_cfg = copy.deepcopy(presets["object_cfg"]["newton"])
        if "events" in presets:
            env_cfg.events = copy.deepcopy(presets["events"]["newton"])

    env_cfg = resolve_preset_defaults(env_cfg)
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


@pytest.fixture(scope="module")
def cartpole_camera_presets():
    """Collect all presets from CartpoleCameraPresetsEnvCfg once for the module."""
    return collect_presets(CartpoleCameraPresetsEnvCfg())


@pytest.fixture(params=_PHYSICS_RENDERER_AOV_COMBINATIONS)
def cartpole_env(request, cartpole_camera_presets):
    """Build Cartpole camera env for (physics_backend, renderer, data_type); step once, yield, close."""
    physics_backend, renderer, data_type = request.param
    presets = cartpole_camera_presets
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


@pytest.fixture(scope="module")
def dexsuite_kuka_allegro_lift_presets():
    """Collect all presets from DexsuiteKukaAllegroLiftEnvCfg once for the module."""
    return collect_presets(DexsuiteKukaAllegroLiftEnvCfg())


@pytest.fixture(params=_PHYSICS_RENDERER_AOV_COMBINATIONS)
def dexsuite_kuka_allegro_lift_env(request, dexsuite_kuka_allegro_lift_presets):
    """Build Dexsuite Kuka-Allegro Lift (single camera) for backend/renderer/data_type; step once, yield, close."""
    physics_backend, renderer, data_type = request.param

    # Dexsuite data type has explicit resolution suffix (64, 128, 256). We only test 64x64 for now.
    dexsuite_data_type = f"{data_type}64"

    env_cfg = DexsuiteKukaAllegroLiftEnvCfg()
    env_cfg.scene = copy.deepcopy(dexsuite_kuka_allegro_lift_presets["scene"]["single_camera"])
    env_cfg.scene.base_camera = copy.deepcopy(
        dexsuite_kuka_allegro_lift_presets["scene.base_camera"][dexsuite_data_type]
    )
    env_cfg.scene.base_camera.renderer_cfg = copy.deepcopy(
        dexsuite_kuka_allegro_lift_presets["scene.base_camera.renderer_cfg"][renderer]
    )
    env_cfg.observations = copy.deepcopy(dexsuite_kuka_allegro_lift_presets["observations"]["single_camera"])
    env_cfg = resolve_preset_defaults(env_cfg)
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
    "Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0",
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
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "sim") and getattr(unwrapped, "sim", None) is not None:
            unwrapped.sim._app_control_on_stop_handle = None

        env.reset()
        num_envs = getattr(unwrapped, "num_envs", 4)
        if getattr(unwrapped, "possible_agents", None):
            actions = {
                agent: sample_space(
                    unwrapped.action_spaces[agent],
                    device=unwrapped.device,
                    batch_size=num_envs,
                    fill_value=0,
                )
                for agent in unwrapped.possible_agents
            }
        else:
            actions = sample_space(
                unwrapped.single_action_space,
                device=unwrapped.device,
                batch_size=num_envs,
                fill_value=0,
            )
        env.step(actions)
        camera_outputs = _collect_camera_outputs(env)
        _assert_camera_outputs_are_not_blank(task_id, camera_outputs)
    finally:
        if env is not None:
            env.close()
