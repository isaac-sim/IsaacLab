# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shadow hand vision environment preset combinations.

Two test suites are provided:

1. **Validation unit tests** — use lightweight ``types.SimpleNamespace`` mocks.
   These exercise :meth:`ShadowHandCameraEnvCfg.validate_config` directly and
   do not require Isaac Sim.

2. **Preset resolution tests** — verify that each named preset in
   :class:`ShadowHandTiledCameraCfg` and
   :class:`~isaaclab_tasks.utils.renderer_cfg.RendererPresetCfg` resolves to the expected
   concrete config class and data types, using the real config classes.

3. **Feature extractor checkpoint tests** — exercise how :class:`FeatureExtractor` resolves the
   CNN checkpoint in its log directory.
"""

import fnmatch
import os
import types

import pytest
import torch
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from isaaclab.renderers import RendererCfg
from isaaclab.sensors import CameraCfg

from isaaclab_rl.utils.pretrained_checkpoint import get_auxiliary_checkpoints

from isaaclab_tasks.core.reorient.config.shadow_hand.feature_extractor import FeatureExtractor, FeatureExtractorCfg
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_camera_manager_env_cfg import (
    ShadowHandCameraManagerEnvCfg,
)
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_camera_env_cfg import (
    ShadowHandCameraEnvCfg,
)
from isaaclab_tasks.utils.hydra import collect_presets

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(renderer_type: str | None, data_types: list[str], feature_extractor_enabled: bool = True):
    """Build a minimal mock cfg with a :meth:`validate_config` method.

    The mock reuses the real validation logic from :class:`ShadowHandCameraEnvCfg`.
    """
    cfg = types.SimpleNamespace()
    cfg.tiled_camera = CameraCfg(
        prim_path="/Camera",
        renderer_cfg=RendererCfg(renderer_type=renderer_type) if renderer_type is not None else None,
        data_types=data_types,
    )
    cfg.feature_extractor = types.SimpleNamespace(enabled=feature_extractor_enabled)
    cfg.validate_config = lambda: ShadowHandCameraEnvCfg.validate_config(cfg)
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
    # ── Warp renderer: rgb, depth, and semantic_segmentation are supported ──
    ("newton_warp", ["rgb"], True),
    ("newton_warp", ["depth"], False),  # depth-only OK when CNN disabled
    ("newton_warp", ["rgb", "depth"], True),  # multiple supported types
    ("newton_warp", ["rgb", "depth", "semantic_segmentation"], True),
]


@pytest.mark.parametrize("renderer_type,data_types,enabled", _VALID_COMBOS)
def test_valid_combinations_do_not_raise(renderer_type, data_types, enabled):
    cfg = _make_cfg(renderer_type, data_types, enabled)
    cfg.validate_config()  # must not raise


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
        cfg.validate_config()


# ---------------------------------------------------------------------------
# Preset resolution — camera data types
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shadow_hand_camera_presets():
    """Collect all presets from ShadowHandCameraEnvCfg once for the module."""
    return collect_presets(ShadowHandCameraEnvCfg())


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
def test_camera_presets_resolve_to_valid_configs(shadow_hand_camera_presets, preset_name, expected_data_types):
    """Camera presets must be discoverable, request data, and have valid dimensions."""
    camera_presets = shadow_hand_camera_presets["tiled_camera"]
    assert preset_name in camera_presets, f"Preset '{preset_name}' not found in tiled_camera presets"
    resolved = camera_presets[preset_name]
    assert resolved.data_types == expected_data_types, (
        f"Preset '{preset_name}': expected data_types={expected_data_types}, got {resolved.data_types}"
    )
    assert len(resolved.data_types) > 0, (
        f"Camera preset '{preset_name}' has an empty data_types list — nothing would be rendered."
    )
    assert resolved.width > 0, f"Camera preset '{preset_name}' has non-positive width: {resolved.width}"
    assert resolved.height > 0, f"Camera preset '{preset_name}' has non-positive height: {resolved.height}"


# ---------------------------------------------------------------------------
# Preset resolution — renderer
# ---------------------------------------------------------------------------

_RENDERER_PRESETS = [
    # preset_name, expected_class
    ("default", NewtonWarpRendererCfg),
    ("isaacsim_rtx", IsaacRtxRendererCfg),
    ("newton_renderer", NewtonWarpRendererCfg),
]


@pytest.mark.parametrize("preset_name,expected_class", _RENDERER_PRESETS)
def test_renderer_presets_resolve_to_expected_configs(shadow_hand_camera_presets, preset_name, expected_class):
    """Renderer presets must resolve to the expected configuration and renderer type."""
    renderer_presets = shadow_hand_camera_presets["tiled_camera.renderer_cfg"]
    assert preset_name in renderer_presets, f"Preset '{preset_name}' not found in renderer presets"
    resolved = renderer_presets[preset_name]
    assert isinstance(resolved, expected_class), (
        f"Renderer preset '{preset_name}': expected {expected_class.__name__}, got {type(resolved).__name__}"
    )
    if preset_name == "newton_renderer":
        assert resolved.renderer_type == "newton_warp"

    rtx_cfg = renderer_presets["rtx"]
    assert isinstance(rtx_cfg, RendererCfg)
    assert rtx_cfg.renderer_type == "auto_rtx"


# ---------------------------------------------------------------------------
# Cross-validation: every camera preset resolves to a valid warp combination
# when paired with the warp renderer preset
# ---------------------------------------------------------------------------

_WARP_CAMERA_PRESETS = [
    ("rgb", False),
    ("depth", False),
    ("default", False),
    ("full", False),
    ("albedo", True),
    ("simple_shading_constant_diffuse", True),
    ("simple_shading_diffuse_mdl", True),
    ("simple_shading_full_mdl", True),
]


@pytest.mark.parametrize("camera_preset,raises", _WARP_CAMERA_PRESETS)
def test_warp_camera_preset_compatibility(shadow_hand_camera_presets, camera_preset, raises):
    """Warp support must match the camera preset's requested data types."""
    camera_cfg = shadow_hand_camera_presets["tiled_camera"][camera_preset]
    warp_cfg = shadow_hand_camera_presets["tiled_camera.renderer_cfg"]["newton_renderer"]
    enabled = camera_cfg.data_types != ["depth"]
    cfg = _make_cfg(warp_cfg.renderer_type, camera_cfg.data_types, enabled)
    if raises:
        with pytest.raises(ValueError):
            cfg.validate_config()
    else:
        cfg.validate_config()


# ---------------------------------------------------------------------------
# Feature extractor checkpoint loading
# ---------------------------------------------------------------------------


def _make_feature_extractor(log_dir: str, load_checkpoint: bool):
    """Build a CPU feature extractor for one log directory."""
    return FeatureExtractor(
        FeatureExtractorCfg(train=False, load_checkpoint=load_checkpoint),
        device="cpu",
        data_types=["rgb"],
        log_dir=log_dir,
    )


def test_missing_feature_extractor_checkpoint_names_the_log_directory(tmp_path):
    """Playing without a feature-extractor checkpoint must report which directory was searched."""
    log_dir = str(tmp_path / "run")

    with pytest.raises(FileNotFoundError, match=log_dir):
        _make_feature_extractor(log_dir, load_checkpoint=True)


def test_published_feature_extractor_checkpoint_is_preferred(monkeypatch, tmp_path):
    """A published companion checkpoint must win over the checkpoints a training run wrote."""
    monkeypatch.chdir(tmp_path)
    log_dir = "run"  # relative, as the pretrained-checkpoint cache directory is
    published = _make_feature_extractor(log_dir, load_checkpoint=False).feature_extractor.state_dict()
    stale = _make_feature_extractor(log_dir, load_checkpoint=False).feature_extractor.state_dict()
    torch.save(published, os.path.join(log_dir, "Isaac-Task_physx_rtx_rsl_rl_feature_extractor.pth"))
    torch.save(stale, os.path.join(log_dir, "cnn_100_0.1.pth"))

    loaded = _make_feature_extractor(log_dir, load_checkpoint=True).feature_extractor.state_dict()

    assert torch.equal(loaded["linear.0.weight"], published["linear.0.weight"])
    assert not torch.equal(published["linear.0.weight"], stale["linear.0.weight"])


def test_saved_checkpoint_name_matches_the_declared_glob():
    """The file a run writes must match the glob the task publishes, or nothing gets collected."""
    saved = "cnn_50000_0.001.pth"  # the name FeatureExtractor.step writes

    assert fnmatch.fnmatch(saved, FeatureExtractorCfg().checkpoint.run_glob)


def test_camera_task_publishes_the_feature_extractor_checkpoint():
    """Discovery must find the CNN checkpoint from the task config, which declares nothing itself."""
    expected = FeatureExtractorCfg().checkpoint

    for env_cfg in (ShadowHandCameraEnvCfg(), ShadowHandCameraManagerEnvCfg()):
        (found,) = get_auxiliary_checkpoints(env_cfg)
        assert (found.name, found.run_glob) == (expected.name, expected.run_glob)
