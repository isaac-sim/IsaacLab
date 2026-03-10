# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shadow hand vision environment preset combinations.

Two test suites are provided:

1. **Validation unit tests** — use lightweight ``types.SimpleNamespace`` mocks.
   These exercise :func:`_validate_cfg` directly and do not require Isaac Sim.

2. **Preset resolution tests** — verify that each named preset in
   :class:`ShadowHandVisionTiledCameraCfg` and
   :class:`~isaaclab_tasks.utils.renderer_cfg.RendererPresetCfg` resolves to the expected
   concrete config class and data types, using the real config classes.
   These require Isaac Sim to be launched so that the renderer cfg imports
   are available.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


import copy  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from isaaclab_newton.renderers import NewtonWarpRendererCfg  # noqa: E402
from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
from isaaclab_physx.renderers import IsaacRtxRendererCfg  # noqa: E402

from isaaclab_tasks.direct.shadow_hand.shadow_hand_vision_env import (  # noqa: E402
    _WARP_SUPPORTED_DATA_TYPES,
    ShadowHandVisionEnv,
    _validate_cfg,
)
from isaaclab_tasks.direct.shadow_hand.shadow_hand_vision_env_cfg import (  # noqa: E402
    ShadowHandVisionBenchmarkEnvCfg,
    ShadowHandVisionEnvCfg,
)
from isaaclab_tasks.utils.hydra import collect_presets, resolve_preset_defaults  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(renderer_type: str | None, data_types: list[str], feature_extractor_enabled: bool = True):
    """Build a minimal mock cfg accepted by :func:`_validate_cfg`."""
    cfg = types.SimpleNamespace()
    cfg.tiled_camera = types.SimpleNamespace(
        renderer_cfg=types.SimpleNamespace(renderer_type=renderer_type),
        data_types=data_types,
    )
    cfg.feature_extractor = types.SimpleNamespace(enabled=feature_extractor_enabled)
    return cfg


# ---------------------------------------------------------------------------
# Valid combinations — must not raise
# ---------------------------------------------------------------------------

_VALID_COMBOS = [
    # renderer_type, data_types, feature_extractor_enabled
    # ── Non-warp renderers accept every data type ──
    (None, ["rgb"], True),
    (None, ["rgb", "depth", "semantic_segmentation"], True),
    (None, ["albedo"], True),
    (None, ["simple_shading_constant_diffuse"], True),
    (None, ["simple_shading_diffuse_mdl"], True),
    (None, ["simple_shading_full_mdl"], True),
    (None, ["depth"], False),  # depth-only OK when CNN disabled
    ("isaac_rtx", ["rgb"], True),
    ("isaac_rtx", ["albedo"], True),
    ("isaac_rtx", ["simple_shading_full_mdl"], True),
    ("isaac_rtx", ["rgb", "depth", "semantic_segmentation"], True),
    ("isaac_rtx", ["depth"], False),
    pytest.param("ovrtx", ["rgb"], True, marks=pytest.mark.skip(reason="OVRTX testing disabled")),
    pytest.param("ovrtx", ["albedo"], True, marks=pytest.mark.skip(reason="OVRTX testing disabled")),
    pytest.param("ovrtx", ["depth"], False, marks=pytest.mark.skip(reason="OVRTX testing disabled")),
    # ── Warp renderer: only rgb and depth are supported ──
    ("newton_warp", ["rgb"], True),
    ("newton_warp", ["depth"], False),  # depth-only OK when CNN disabled
    ("newton_warp", ["rgb", "depth"], True),  # multiple supported types
]


@pytest.mark.parametrize("renderer_type,data_types,enabled", _VALID_COMBOS)
def test_valid_combinations_do_not_raise(renderer_type, data_types, enabled):
    cfg = _make_cfg(renderer_type, data_types, enabled)
    _validate_cfg(cfg)  # must not raise


# ---------------------------------------------------------------------------
# Invalid combinations — must raise ValueError with a descriptive message
# ---------------------------------------------------------------------------

_INVALID_COMBOS = [
    # renderer_type, data_types, enabled, substring expected in error message
    # ── Warp does not support colour-space data types ──
    (
        "newton_warp",
        ["albedo"],
        True,
        "albedo",
    ),
    (
        "newton_warp",
        ["simple_shading_constant_diffuse"],
        True,
        "simple_shading_constant_diffuse",
    ),
    (
        "newton_warp",
        ["simple_shading_diffuse_mdl"],
        True,
        "simple_shading_diffuse_mdl",
    ),
    (
        "newton_warp",
        ["simple_shading_full_mdl"],
        True,
        "simple_shading_full_mdl",
    ),
    (
        "newton_warp",
        ["rgb", "depth", "semantic_segmentation"],
        True,
        "semantic_segmentation",
    ),
    # ── Depth-only with CNN enabled is not valid for training ──
    (
        None,
        ["depth"],
        True,
        "Depth-only",
    ),
    (
        "isaac_rtx",
        ["depth"],
        True,
        "Depth-only",
    ),
    pytest.param(
        "ovrtx",
        ["depth"],
        True,
        "Depth-only",
        marks=pytest.mark.skip(reason="OVRTX testing disabled"),
    ),
    (
        "newton_warp",
        ["depth"],
        True,
        "Depth-only",  # depth is warp-supported but CNN can't train on it
    ),
]


@pytest.mark.parametrize("renderer_type,data_types,enabled,match", _INVALID_COMBOS)
def test_invalid_combinations_raise_value_error(renderer_type, data_types, enabled, match):
    cfg = _make_cfg(renderer_type, data_types, enabled)
    with pytest.raises(ValueError, match=match):
        _validate_cfg(cfg)


# ---------------------------------------------------------------------------
# Warp supported data types constant
# ---------------------------------------------------------------------------


def test_warp_supported_data_types():
    """_WARP_SUPPORTED_DATA_TYPES must contain exactly rgb and depth."""
    assert {"rgb", "depth"} == _WARP_SUPPORTED_DATA_TYPES


# ---------------------------------------------------------------------------
# Preset resolution — camera data types
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shadow_hand_vision_presets():
    """Collect all presets from ShadowHandVisionEnvCfg once for the module."""
    return collect_presets(ShadowHandVisionEnvCfg())


_CAMERA_DATA_TYPE_PRESETS = [
    # preset_name, expected_data_types
    ("default", ["rgb", "depth", "semantic_segmentation"]),
    ("full", ["rgb", "depth", "semantic_segmentation"]),
    ("rgb", ["rgb"]),
    ("albedo", ["albedo"]),
    ("simple_shading_constant_diffuse", ["simple_shading_constant_diffuse"]),
    ("simple_shading_diffuse_mdl", ["simple_shading_diffuse_mdl"]),
    ("simple_shading_full_mdl", ["simple_shading_full_mdl"]),
    ("depth", ["depth"]),
]


@pytest.mark.parametrize("preset_name,expected_data_types", _CAMERA_DATA_TYPE_PRESETS)
def test_camera_preset_data_types(shadow_hand_vision_presets, preset_name, expected_data_types):
    camera_presets = shadow_hand_vision_presets["tiled_camera"]
    assert preset_name in camera_presets, f"Preset '{preset_name}' not found in tiled_camera presets"
    resolved = camera_presets[preset_name]
    assert resolved.data_types == expected_data_types, (
        f"Preset '{preset_name}': expected data_types={expected_data_types}, got {resolved.data_types}"
    )


@pytest.mark.parametrize("preset_name,_", _CAMERA_DATA_TYPE_PRESETS)
def test_camera_preset_cfg_is_valid(shadow_hand_vision_presets, preset_name, _):
    """Every camera preset config must request at least one data type and have valid dimensions.

    Note: this is a config-level check only. It verifies that the preset is correctly wired up
    (non-empty data_types, positive width/height) but does NOT run the renderer or inspect actual
    pixel values. Verifying that rendered frames are non-empty requires a full integration test
    that steps the simulation and checks the camera output tensors.
    """
    resolved = shadow_hand_vision_presets["tiled_camera"][preset_name]
    assert len(resolved.data_types) > 0, (
        f"Camera preset '{preset_name}' has an empty data_types list — nothing would be rendered."
    )
    assert resolved.width > 0, f"Camera preset '{preset_name}' has non-positive width: {resolved.width}"
    assert resolved.height > 0, f"Camera preset '{preset_name}' has non-positive height: {resolved.height}"


def test_all_camera_presets_present(shadow_hand_vision_presets):
    """Every preset defined in ShadowHandVisionTiledCameraCfg is discoverable."""
    camera_presets = shadow_hand_vision_presets["tiled_camera"]
    expected_names = {name for name, _ in _CAMERA_DATA_TYPE_PRESETS}
    missing = expected_names - set(camera_presets.keys())
    assert not missing, f"Camera presets missing from collected presets: {missing}"


# ---------------------------------------------------------------------------
# Preset resolution — renderer
# ---------------------------------------------------------------------------

_RENDERER_PRESETS = [
    # preset_name, expected_class
    ("default", IsaacRtxRendererCfg),
    ("isaacsim_rtx_renderer", IsaacRtxRendererCfg),
    ("newton_renderer", NewtonWarpRendererCfg),
    pytest.param("ovrtx_renderer", OVRTXRendererCfg, marks=pytest.mark.skip(reason="OVRTX testing disabled")),
]


@pytest.mark.parametrize("preset_name,expected_class", _RENDERER_PRESETS)
def test_renderer_preset_class(shadow_hand_vision_presets, preset_name, expected_class):
    renderer_presets = shadow_hand_vision_presets["tiled_camera.renderer_cfg"]
    assert preset_name in renderer_presets, f"Preset '{preset_name}' not found in renderer presets"
    resolved = renderer_presets[preset_name]
    assert isinstance(resolved, expected_class), (
        f"Renderer preset '{preset_name}': expected {expected_class.__name__}, got {type(resolved).__name__}"
    )


def test_warp_renderer_has_correct_renderer_type(shadow_hand_vision_presets):
    """NewtonWarpRendererCfg must expose renderer_type='newton_warp' for validation to work."""
    warp_cfg = shadow_hand_vision_presets["tiled_camera.renderer_cfg"]["newton_renderer"]
    assert warp_cfg.renderer_type == "newton_warp"


def test_all_renderer_presets_present(shadow_hand_vision_presets):
    """Every preset in MultiBackendRendererCfg is discoverable."""
    renderer_presets = shadow_hand_vision_presets["tiled_camera.renderer_cfg"]
    expected_names = {"default", "isaacsim_rtx_renderer", "newton_renderer", "ovrtx_renderer"}
    missing = expected_names - set(renderer_presets.keys())
    assert not missing, f"Renderer presets missing from collected presets: {missing}"


# ---------------------------------------------------------------------------
# Cross-validation: every camera preset resolves to a valid warp combination
# when paired with the warp renderer preset
# ---------------------------------------------------------------------------

_WARP_VALID_CAMERA_PRESETS = ["rgb", "depth"]
_WARP_INVALID_CAMERA_PRESETS = [
    "default",
    "full",
    "albedo",
    "simple_shading_constant_diffuse",
    "simple_shading_diffuse_mdl",
    "simple_shading_full_mdl",
]


@pytest.mark.parametrize("camera_preset", _WARP_VALID_CAMERA_PRESETS)
def test_warp_with_valid_camera_preset(shadow_hand_vision_presets, camera_preset):
    """Warp + {rgb, depth} camera presets must not raise (depth with CNN disabled)."""
    camera_cfg = shadow_hand_vision_presets["tiled_camera"][camera_preset]
    warp_cfg = shadow_hand_vision_presets["tiled_camera.renderer_cfg"]["newton_renderer"]
    enabled = camera_cfg.data_types != ["depth"]  # disable CNN for depth-only
    cfg = _make_cfg(warp_cfg.renderer_type, camera_cfg.data_types, enabled)
    _validate_cfg(cfg)  # must not raise


@pytest.mark.parametrize("camera_preset", _WARP_INVALID_CAMERA_PRESETS)
def test_warp_with_invalid_camera_preset(shadow_hand_vision_presets, camera_preset):
    """Warp + unsupported camera presets must raise ValueError."""
    camera_cfg = shadow_hand_vision_presets["tiled_camera"][camera_preset]
    warp_cfg = shadow_hand_vision_presets["tiled_camera.renderer_cfg"]["newton_renderer"]
    cfg = _make_cfg(warp_cfg.renderer_type, camera_cfg.data_types, True)
    with pytest.raises(ValueError):
        _validate_cfg(cfg)


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
    # xfail: standard Shadow Hand USD contains PhysX tendons that Newton's ModelBuilder cannot parse,
    # so the Newton model build fails and the Warp renderer cannot initialise.
    pytest.param(
        ("newton_renderer", "rgb", "physx"),
        id="physx-warp-rgb",
        marks=pytest.mark.xfail(raises=RuntimeError, reason="PhysX tendon schemas unsupported by Newton ModelBuilder"),
    ),
    pytest.param(
        ("newton_renderer", "depth", "physx"),
        id="physx-warp-depth",
        marks=pytest.mark.xfail(raises=RuntimeError, reason="PhysX tendon schemas unsupported by Newton ModelBuilder"),
    ),
    # ── Newton physics + Warp: Warp renderer is physics-backend agnostic ──
    pytest.param(("newton_renderer", "rgb", "newton"), id="newton-warp-rgb"),
    pytest.param(("newton_renderer", "depth", "newton"), id="newton-warp-depth"),
    # ── Newton physics + IsaacRTX ──
    pytest.param(("isaacsim_rtx_renderer", "rgb", "newton"), id="newton-isaacsim_rtx-rgb"),
    pytest.param(("isaacsim_rtx_renderer", "depth", "newton"), id="newton-isaacsim_rtx-depth"),
    pytest.param(("isaacsim_rtx_renderer", "albedo", "newton"), id="newton-isaacsim_rtx-albedo"),
    pytest.param(
        ("isaacsim_rtx_renderer", "simple_shading_constant_diffuse", "newton"),
        id="newton-isaacsim_rtx-simple_shading_constant_diffuse",
    ),
    pytest.param(
        ("isaacsim_rtx_renderer", "simple_shading_diffuse_mdl", "newton"),
        id="newton-isaacsim_rtx-simple_shading_diffuse_mdl",
    ),
    pytest.param(
        ("isaacsim_rtx_renderer", "simple_shading_full_mdl", "newton"),
        id="newton-isaacsim_rtx-simple_shading_full_mdl",
    ),
    # ── OVRTX: disabled ──
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

    All renderer+camera+physics combinations are expected to produce non-empty frames.
    """
    renderer_preset, camera_preset, physics, env = render_correctness_env
    label = f"{physics}-{renderer_preset}+{camera_preset}"
    camera_output = env._tiled_camera.data.output
    assert len(camera_output) > 0, f"[{label}] Camera produced no output tensors at all."
    for dt, tensor in camera_output.items():
        finite = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
        # import pdb; pdb.set_trace()
        assert finite.max() > 0.2, (
            f"[{label}] Camera output '{dt}' is all zeros or all inf "
            f"after stepping. Tensor shape: {tensor.shape}, dtype: {tensor.dtype}."
        )
