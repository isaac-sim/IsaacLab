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
import json  # noqa: E402
import numpy as np  # noqa: E402
import os  # noqa: E402
from PIL import Image, ImageChops  # noqa: E402
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
# Constants
# ---------------------------------------------------------------------------

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_IMAGES_DIR = os.path.join(_TEST_DIR, "golden_images")

# Golden image comparison: max per-pixel channel difference (0-255) to count as "same"
_PIXEL_DIFF_THRESHOLD = 5
# Max fraction of pixels (in percent) allowed to exceed _PIXEL_DIFF_THRESHOLD
_MAX_DIFF_PERCENT = 5.0


_OVRTX_DISABLED = pytest.mark.skip(
    reason="OVRTX is optional and experimental feature and temporarily is excluded from testing."
)


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
        marks=_OVRTX_DISABLED,
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "albedo"),
        id="newton-ovrtx-albedo",
        marks=_OVRTX_DISABLED,
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "depth"),
        id="newton-ovrtx-depth",
        marks=_OVRTX_DISABLED,
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "simple_shading_constant_diffuse"),
        id="newton-ovrtx-simple_shading_constant_diffuse",
        marks=_OVRTX_DISABLED,
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "simple_shading_diffuse_mdl"),
        id="newton-ovrtx-simple_shading_diffuse_mdl",
        marks=_OVRTX_DISABLED,
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "simple_shading_full_mdl"),
        id="newton-ovrtx-simple_shading_full_mdl",
        marks=_OVRTX_DISABLED,
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


def _compare_images(
    result_image: Image.Image,
    golden_image_path: str,
) -> tuple[bool, str]:
    """Compare result and golden images; return (True, \"\") if deemed equal.

    Args:
        result_image: Current render as PIL Image.
        golden_image_path: Path to golden image (used in error messages).

    Returns:
        (True, \"\") if images are deemed equal, else (False, error_message).
    """
    try:
        golden_image = Image.open(golden_image_path)
    except Exception as e:
        return False, f"Error loading golden image from {golden_image_path}: {e}"

    if result_image.size != golden_image.size:
        return False, f"Size mismatch for {golden_image_path}: expected {golden_image.size}, got {result_image.size}."

    if result_image.mode != golden_image.mode:
        return False, f"Mode mismatch for {golden_image_path}: expected {golden_image.mode}, got {result_image.mode}."

    diff_image = ImageChops.difference(result_image, golden_image).convert(
        result_image.mode.replace("A", "")
    )
    diff_arr = np.array(diff_image)

    # Calculate the squared sum of each channel (shape [..., channels])
    # The sum is over the two first spatial dimensions
    channel_squared_sum = np.sum(diff_arr ** 2, axis=2)  # shape [H, W]
    print(f"[HDC] Channel squared sum: {channel_squared_sum}")
    # Count pixels (not channel elements) where any channel exceeds the threshold.
    pixels_over_threshold = (channel_squared_sum > _PIXEL_DIFF_THRESHOLD).any(axis=-1)
    num_diff_pixels = int(np.sum(pixels_over_threshold))
    total_pixels = diff_arr.shape[0] * diff_arr.shape[1]
    diff_percent = 100.0 * num_diff_pixels / total_pixels
    print(f"[HDC] diff_percent: {diff_percent} ({num_diff_pixels} / {total_pixels} pixels)")
    if diff_percent >= _MAX_DIFF_PERCENT:
        return False, f"Pixel difference for {golden_image_path} exceeds {_MAX_DIFF_PERCENT}%: got {diff_percent:.2f}% ({num_diff_pixels} / {total_pixels} pixels)."

    return True, ""


def _make_grid(images: torch.Tensor) -> torch.Tensor:
    """Make a grid of images from a tensor of shape (B, H, W, C).

    Args:
        images: A tensor of shape (B, H, W, C) containing the images.

    Returns:
        A tensor of shape (H, W, C) containing the grid of images.
    """
    from torchvision.utils import make_grid
    return make_grid(torch.swapaxes(images.unsqueeze(1), 1, -1).squeeze(-1), nrow=round(images.shape[0] ** 0.5))


def _check_camera_outputs(test_name: str, physics_backend: str, renderer: str, camera_outputs: dict[str, torch.Tensor]) -> None:
    """Check validity and consistency of camera outputs.

    Args:
        test_name: Test name.
        physics_backend: Physics backend.
        renderer: Renderer.
        camera_outputs: {data_type -> tensor}.
    """
    assert len(camera_outputs) > 0, f"[{test_name}] No camera outputs produced by {physics_backend} + {renderer}."

    for data_type, tensor in camera_outputs.items():
        invalid = torch.logical_or(torch.isinf(tensor), torch.isnan(tensor))
        corrected = torch.where(invalid, torch.zeros_like(tensor), tensor)
        assert corrected.max() > 0, (
            f"[{test_name}] Camera output '{data_type}' has no non-zero pixels. "
            f"Shape: {corrected.shape}, dtype: {corrected.dtype}."
        )

        normalized = _normalize_tensor(corrected, data_type)
        grid = _make_grid(normalized)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        result_image = Image.fromarray(ndarr)

        golden_images_dir = os.path.join(_GOLDEN_IMAGES_DIR, test_name)
        golden_image_path = f"{golden_images_dir}/{physics_backend}-{renderer}-{data_type}.png"

        success, message = _compare_images(result_image, golden_image_path)
        if not success:
            os.makedirs(os.path.dirname(golden_image_path), exist_ok=True)
            result_image.save(golden_image_path)

        assert success, f"Image mismatch for {golden_image_path}: {message}"


def _normalize_tensor(tensor: torch.Tensor, data_type: str) -> torch.Tensor:
    """Convert camera output tensor to [0, 1] float32 for conversion to image.

    Args:
        tensor: Camera output tensor.
        data_type: Data type of the camera output.

    Returns:
        Normalized tensor.
    """
    normalized = tensor.float()

    if data_type == "depth":
        max_val = normalized.max()
        if max_val > 0:
            normalized = normalized / max_val
    elif data_type == "rgba":
        # Keep 4 channels so tensor -> PIL produces RGBA.
        normalized = normalized[..., :4] / 255.0
    else:
        # rgb, semantic_segmentation, albedo, and simple_shading_* are uint8 [0, 255]
        normalized = normalized[..., :3] / 255.0

    return normalized


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


def test_shadow_hand(shadow_hand_env):
    """Camera output must contain at least one non-zero pixel (Shadow Hand vision env)."""
    physics_backend, renderer, _, env = shadow_hand_env
    _check_camera_outputs("shadow_hand", physics_backend, renderer, env._tiled_camera.data.output)


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


def test_cartpole(cartpole_env):
    """Camera output must contain at least one non-zero pixel (Cartpole camera env)."""
    physics_backend, renderer, _, env = cartpole_env
    _check_camera_outputs("cartpole", physics_backend, renderer, env._tiled_camera.data.output)


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
    env_cfg.seed = 42

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


def test_dexsuite_kuka_allegro_lift(dexsuite_kuka_allegro_lift_env):
    """Camera output must contain at least one non-zero pixel (Dexsuite Kuka-Allegro Lift, single camera)."""
    physics_backend, renderer, _, env = dexsuite_kuka_allegro_lift_env
    _check_camera_outputs("dexsuite_kuka", physics_backend, renderer, env.scene.sensors["base_camera"].data.output)


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
]


@pytest.mark.parametrize("task_id", _RENDER_CORRECTNESS_TASK_IDS)
def test_registered_tasks(task_id):
    """Camera output must be non-empty for each registered task with camera-based observations."""
    env = None
    try:
        env_cfg = parse_env_cfg(task_id, num_envs=4)
        env_cfg.seed = 42

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
        assert len(camera_outputs) == 1, f"[{task_id}] Expected 1 camera output, got {len(camera_outputs)}."
        assert "tiled_camera" in camera_outputs, f"[{task_id}] Expected 'tiled_camera' output, got {camera_outputs.keys()}."

        _check_camera_outputs(f"registered_tasks/{task_id}", "default_physics", "default_renderer", camera_outputs["tiled_camera"])
    finally:
        if env is not None:
            env.close()
