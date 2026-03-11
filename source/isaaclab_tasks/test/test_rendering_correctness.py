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

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
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


# Skip reason for ovrtx_renderer
_OVRTX_SKIP = "OVRTX testing disabled"


@pytest.fixture(scope="module")
def shadow_hand_vision_presets():
    """Collect all presets from ShadowHandVisionEnvCfg once for the module."""
    return collect_presets(ShadowHandVisionEnvCfg())


@pytest.fixture(scope="module")
def cartpole_camera_presets():
    """Collect all presets from CartpoleCameraPresetsEnvCfg once for the module."""
    return collect_presets(CartpoleCameraPresetsEnvCfg())


@pytest.fixture(scope="module")
def dexsuite_kuka_allegro_lift_presets():
    """Collect all presets from DexsuiteKukaAllegroLiftEnvCfg once for the module."""
    return collect_presets(DexsuiteKukaAllegroLiftEnvCfg())


@pytest.fixture(autouse=True)
def cleanup_simulation_context():
    """Fixture to ensure SimulationContext is cleared after each test."""
    yield
    # Cleanup after test
    SimulationContext.clear_instance()


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


def _assert_camera_renders_not_empty(env_name, physics_backend, renderer, data_type, camera_output):
    """Shared assertion: camera output has at least one non-zero pixel per data type."""
    label = f"{env_name}-{physics_backend}-{renderer}+{data_type}"
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
    Uses try/finally so env.close() runs even when setup or test fails, ensuring simulation context is cleared.
    """
    physics_backend, renderer, data_type = request.param
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
    env_cfg.feature_extractor.write_image_to_file = True
    env_cfg.seed = 42

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


def test_camera_renders_not_empty_shadow_hand(shadow_hand_env):
    """Camera output must contain at least one non-zero pixel (Shadow Hand vision env)."""
    physics_backend, renderer, data_type, env = shadow_hand_env
    _assert_camera_renders_not_empty("shadow_hand", physics_backend, renderer, data_type, env._tiled_camera.data.output)


@pytest.fixture(params=_SHARED_RENDER_CORRECTNESS_CASES)
def cartpole_env(request, cartpole_camera_presets):
    """Build Cartpole camera env for (physics_backend, renderer, data_type), step once, yield, close.

    Function-scoped so each parametrized case creates and closes its own env sequentially.
    Uses try/finally so env.close() runs even when setup or test fails, ensuring simulation context is cleared.
    """
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
    env_cfg.write_image_to_file = True
    env_cfg.seed = 42
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


def test_camera_renders_not_empty_cartpole(cartpole_env):
    """Camera output must contain at least one non-zero pixel (Cartpole camera env)."""
    physics_backend, renderer, data_type, env = cartpole_env
    _assert_camera_renders_not_empty("cartpole", physics_backend, renderer, data_type, env._tiled_camera.data.output)


@pytest.fixture(params=_SHARED_RENDER_CORRECTNESS_CASES)
def dexsuite_kuka_allegro_lift_env(request, dexsuite_kuka_allegro_lift_presets, shadow_hand_vision_presets):
    """Build Dexsuite Kuka-Allegro Lift env (single camera) for (physics_backend, renderer, data_type).

    Uses scene.single_camera and scene.base_camera presets; renderer from Shadow Hand presets.
    Skips cases Dexsuite does not support (no simple_shading presets; PhysX only; newton+depth).
    Function-scoped so each parametrized case creates and closes its own env sequentially.
    Uses try/finally so env.close() runs even when setup or test fails, ensuring simulation context is cleared.
    """
    physics_backend, renderer, data_type = request.param

    # Map the presets to the 64x64 presets
    data_type_to_dexsuite = {
        "rgb": "rgb64",
        "albedo": "albedo64",
        "depth": "depth64",
        "simple_shading_constant_diffuse": "simple_shading_constant_diffuse64",
        "simple_shading_diffuse_mdl": "simple_shading_diffuse_mdl64",
        "simple_shading_full_mdl": "simple_shading_full_mdl64",
    }

    dexsuite_camera_key = data_type_to_dexsuite[data_type]
    env_cfg = DexsuiteKukaAllegroLiftEnvCfg()
    env_cfg.scene = copy.deepcopy(dexsuite_kuka_allegro_lift_presets["scene"]["single_camera"])
    env_cfg.scene.base_camera = copy.deepcopy(dexsuite_kuka_allegro_lift_presets["scene.base_camera"][dexsuite_camera_key])
    env_cfg.scene.base_camera.renderer_cfg = copy.deepcopy(
        shadow_hand_vision_presets["tiled_camera.renderer_cfg"][renderer]
    )
    env_cfg.observations = copy.deepcopy(dexsuite_kuka_allegro_lift_presets["observations"]["single_camera"])
    env_cfg = resolve_preset_defaults(env_cfg)
    env_cfg.scene.num_envs = 4
    env_cfg.seed = 42
    env_cfg.write_image_to_file = True
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


def test_camera_renders_not_empty_dexsuite_kuka_allegro_lift(dexsuite_kuka_allegro_lift_env):
    """Camera output must contain at least one non-zero pixel (Dexsuite Kuka-Allegro Lift, single camera)."""
    physics_backend, renderer, data_type, env = dexsuite_kuka_allegro_lift_env

    camera = env.scene.sensors["base_camera"]
    _assert_camera_renders_not_empty(
        "dexsuite_kuka_allegro_lift",
        physics_backend,
        renderer,
        data_type,
        camera.data.output,
    )
