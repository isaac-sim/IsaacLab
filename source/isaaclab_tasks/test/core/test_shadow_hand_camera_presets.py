# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shadow hand vision environment preset combinations.

Two test suites are provided:

1. **Validation unit tests** — use lightweight ``types.SimpleNamespace`` mocks.
   These exercise generic camera validation followed by Shadow Hand's task-specific
   feature-extractor validation and do not require Isaac Sim.

2. **Preset resolution tests** — verify that each named preset in
   :class:`ShadowHandTiledCameraCfg` and
   :class:`~isaaclab_tasks.utils.renderer_cfg.RendererPresetCfg` resolves to the expected
   concrete config class and data types, using the real config classes.

3. **Checkpoint tests** — verify that published policies resolve to their matching
   feature-extractor checkpoints.
"""

import types
from pathlib import Path

import pytest
import torch
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from isaaclab.renderers import RendererCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.core.reorient.config.shadow_hand import feature_extractor as feature_extractor_module
from isaaclab_tasks.core.reorient.config.shadow_hand.feature_extractor import FeatureExtractor, FeatureExtractorCfg
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_camera_manager_env_cfg import (
    ShadowHandCameraManagerEnvCfg,
)
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_camera_env_cfg import (
    ShadowHandCameraEnvCfg,
)
from isaaclab_tasks.utils import parse_env_cfg, resolve_task_config
from isaaclab_tasks.utils.hydra import collect_presets, resolve_presets

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(renderer_type: str | None, data_types: list[str], feature_extractor_enabled: bool = True):
    """Build a minimal mock cfg with a :meth:`validate_config` method.

    The mock reuses the real validation logic from :class:`ShadowHandCameraEnvCfg`.
    """
    cfg = types.SimpleNamespace()
    if renderer_type == "newton_warp":
        renderer_cfg = NewtonWarpRendererCfg()
    elif renderer_type == "isaac_rtx":
        renderer_cfg = IsaacRtxRendererCfg()
    else:
        renderer_cfg = RendererCfg(renderer_type=renderer_type) if renderer_type is not None else None
    cfg.scene = types.SimpleNamespace(
        tiled_camera=CameraCfg(
            prim_path="/Camera",
            renderer_cfg=renderer_cfg,
            data_types=data_types,
        )
    )
    cfg.feature_extractor = types.SimpleNamespace(enabled=feature_extractor_enabled)
    cfg.validate_config = lambda: ShadowHandCameraEnvCfg.validate_config(cfg)
    return cfg


def _validate_cfg(cfg) -> None:
    """Run camera and task hooks for the intentionally incomplete lightweight mock."""
    cfg.scene.tiled_camera.validate_config()
    cfg.validate_config()


# ---------------------------------------------------------------------------
# Valid combinations — must not raise
# ---------------------------------------------------------------------------

_VALID_COMBOS = [
    # renderer_type, data_types, feature_extractor_enabled
    (None, ["rgb", "depth", "semantic_segmentation"], True),  # no renderer contract to check
    ("isaac_rtx", ["simple_shading_full_mdl"], True),  # RTX publishes simple-shading outputs
    ("newton_warp", ["rgb", "depth", "semantic_segmentation"], True),  # warp-published outputs
    ("newton_warp", ["depth"], False),  # depth-only OK when CNN disabled
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
    # ── Warp does not support RTX simple-shading outputs ──
    ("newton_warp", ["simple_shading_full_mdl"], True, "simple_shading_full_mdl"),
    # ── Depth-only with CNN enabled is not valid for training (renderer-independent) ──
    (None, ["depth"], True, "Depth-only"),
]


@pytest.mark.parametrize("renderer_type,data_types,enabled,match", _INVALID_COMBOS)
def test_invalid_combinations_raise_value_error(renderer_type, data_types, enabled, match):
    cfg = _make_cfg(renderer_type, data_types, enabled)
    with pytest.raises(ValueError, match=match):
        _validate_cfg(cfg)


# ---------------------------------------------------------------------------
# Preset resolution — camera data types
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shadow_hand_camera_presets():
    """Collect all presets from ShadowHandCameraEnvCfg once for the module."""
    return collect_presets(ShadowHandCameraEnvCfg())


def test_camera_presets_resolve_to_valid_configs(shadow_hand_camera_presets):
    """Camera presets must be discoverable, request their named data type, and have valid dimensions."""
    camera_presets = shadow_hand_camera_presets["scene.tiled_camera"]
    assert set(camera_presets) == {
        "default",
        "full",
        "rgb",
        "albedo",
        "simple_shading_constant_diffuse",
        "simple_shading_diffuse_mdl",
        "simple_shading_full_mdl",
        "depth",
        "semantic_segmentation",
    }
    for preset_name, resolved in camera_presets.items():
        if preset_name in ("default", "full"):
            assert resolved.data_types == ["rgb", "depth", "semantic_segmentation"], preset_name
        else:
            assert resolved.data_types == [preset_name]
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
    renderer_presets = shadow_hand_camera_presets["scene.tiled_camera.renderer_cfg"]
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
    ("albedo", False),
    ("simple_shading_constant_diffuse", True),
    ("simple_shading_diffuse_mdl", True),
    ("simple_shading_full_mdl", True),
]


@pytest.mark.parametrize("camera_preset,raises", _WARP_CAMERA_PRESETS)
def test_warp_camera_preset_compatibility(shadow_hand_camera_presets, camera_preset, raises):
    """Warp support must match the camera preset's requested data types."""
    camera_cfg = shadow_hand_camera_presets["scene.tiled_camera"][camera_preset]
    warp_cfg = shadow_hand_camera_presets["scene.tiled_camera.renderer_cfg"]["newton_renderer"]
    enabled = camera_cfg.data_types != ["depth"]
    cfg = _make_cfg(warp_cfg.renderer_type, camera_cfg.data_types, enabled)
    if raises:
        with pytest.raises(ValueError):
            _validate_cfg(cfg)
    else:
        _validate_cfg(cfg)


@pytest.mark.parametrize(
    "env_cfg_type,presets,checkpoint_filename",
    [
        (
            ShadowHandCameraManagerEnvCfg,
            ("newton_mjwarp", "newton_renderer"),
            "Isaac-Reorient-Cube-Shadow-Camera_newtonmjwarp_newton_rsl_rl_feature_extractor.pth",
        ),
        (
            ShadowHandCameraManagerEnvCfg,
            ("isaacsim_physx", "isaacsim_rtx"),
            "Isaac-Reorient-Cube-Shadow-Camera_physx_rtx_rsl_rl_feature_extractor.pth",
        ),
        (
            ShadowHandCameraManagerEnvCfg,
            ("ovphysx", "ovrtx"),
            "Isaac-Reorient-Cube-Shadow-Camera_physx_rtx_rsl_rl_feature_extractor.pth",
        ),
        (
            ShadowHandCameraEnvCfg,
            ("newton_mjwarp", "newton_renderer"),
            "Isaac-Reorient-Cube-Shadow-Camera-Direct_newtonmjwarp_newton_rsl_rl_feature_extractor.pth",
        ),
        (
            ShadowHandCameraEnvCfg,
            ("isaacsim_physx", "isaacsim_rtx"),
            "Isaac-Reorient-Cube-Shadow-Camera-Direct_physx_rtx_rsl_rl_feature_extractor.pth",
        ),
        (
            ShadowHandCameraEnvCfg,
            ("ovphysx", "ovrtx"),
            "Isaac-Reorient-Cube-Shadow-Camera-Direct_physx_rtx_rsl_rl_feature_extractor.pth",
        ),
    ],
)
def test_task_presets_select_published_feature_extractor_checkpoint(
    env_cfg_type: type,
    presets: tuple[str, str],
    checkpoint_filename: str,
) -> None:
    """Each task/backend combination must select its published feature-extractor checkpoint."""
    env_cfg = resolve_presets(env_cfg_type(), presets)

    expected_path = f"{ISAACLAB_NUCLEUS_DIR}/PretrainedCheckpoints/rsl_rl/{checkpoint_filename}"
    assert env_cfg.feature_extractor.pretrained_checkpoint == expected_path


def test_registered_cosmos_preset_composes_rgb_observations_and_local_checkpoint_playback():
    """The training task composes depth-guided RGB, and playback retains its own trained CNN."""
    task_name = "Isaac-Reorient-Cube-Shadow-Camera-Direct"
    env_cfg = parse_env_cfg(task_name, overrides=("presets=cosmos", "env.episode_length_s=20.0"))
    env_cfg.validate()
    camera = env_cfg.scene.tiled_camera
    transfer = camera.modifiers["distance_to_image_plane"][-1]

    assert env_cfg.scene.num_envs == 1
    assert camera.data_types == ["rgb"] and transfer.output == "rgb"
    assert transfer.backend.modality == "depth"
    assert env_cfg.feature_extractor.enabled and env_cfg.feature_extractor.train
    assert env_cfg.feature_extractor.pretrained_checkpoint is None

    play_cfg, _ = resolve_task_config(task_name, None, play_mode=True, overrides=("presets=cosmos",))
    play_cfg.validate()

    assert play_cfg.scene.num_envs == 1
    assert not play_cfg.feature_extractor.train
    assert play_cfg.feature_extractor.load_checkpoint
    assert play_cfg.feature_extractor.pretrained_checkpoint is None


def test_manager_camera_term_passes_capture_counters_and_resets_held_targets(monkeypatch):
    """The Manager term gives the feature extractor each capture counter and invalidates targets on reset."""
    from isaaclab_tasks.core.reorient import mdp

    calls = []

    class RecordingExtractor:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self, env_ids=None):
            calls.append(("reset", env_ids))

        def step(self, camera_output, gt_pose, *, camera_frame=None):
            calls.append(("step", camera_frame))
            return None, torch.zeros(gt_pose.shape[0], 27)

    monkeypatch.setattr(feature_extractor_module, "FeatureExtractor", RecordingExtractor)
    frame = torch.tensor([5])
    camera = types.SimpleNamespace(
        cfg=types.SimpleNamespace(data_types=["rgb"], height=4, width=4),
        data=types.SimpleNamespace(output={"rgb": torch.zeros(1, 4, 4, 3)}),
        frame=types.SimpleNamespace(torch=frame),
    )
    cube = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=types.SimpleNamespace(torch=torch.zeros(1, 3)),
            root_quat_w=types.SimpleNamespace(torch=torch.tensor([[1.0, 0.0, 0.0, 0.0]])),
        )
    )

    class Scene(types.SimpleNamespace):
        def __getitem__(self, name):
            return cube

    env = types.SimpleNamespace(
        device="cpu",
        num_envs=1,
        cfg=types.SimpleNamespace(log_dir=None),
        extras={},
        scene=Scene(sensors={"tiled_camera": camera}, env_origins=torch.zeros(1, 3)),
    )
    params = {"feature_extractor_cfg": FeatureExtractorCfg(), "sensor_cfg": types.SimpleNamespace(name="tiled_camera")}
    term = mdp.ShadowHandCameraFeatures(types.SimpleNamespace(params=params), env)
    term.reset([0])
    term(env, params["feature_extractor_cfg"], params["sensor_cfg"], types.SimpleNamespace(name="object"))

    assert calls[0] == ("reset", [0])
    assert calls[1][0] == "step" and calls[1][1] is frame


def test_manager_cosmos_preset_feeds_generated_rgb_to_the_camera_observation_term():
    """The Manager task uses the Direct task's Cosmos camera; its observation term reads the generated rgb."""
    env_cfg = parse_env_cfg("Isaac-Reorient-Cube-Shadow-Camera", overrides=("presets=cosmos",))
    env_cfg.validate()
    camera = env_cfg.scene.tiled_camera

    assert env_cfg.scene.num_envs == 1 and (camera.width, camera.height) == (640, 640)
    assert camera.modifier_outputs() == {"distance_to_image_plane": "rgb"}
    assert env_cfg.observations.policy.camera_features.params["feature_extractor_cfg"] == env_cfg.feature_extractor
    assert env_cfg.feature_extractor.image_update_frames == 4
    assert env_cfg.feature_extractor.pretrained_checkpoint is None

    too_many = parse_env_cfg("Isaac-Reorient-Cube-Shadow-Camera", overrides=("presets=cosmos", "env.scene.num_envs=2"))
    with pytest.raises(ValueError, match="requires one environment"):
        too_many.validate()


@pytest.mark.parametrize(
    "overrides,error",
    [
        (("env.scene.num_envs=2",), "requires one environment"),
        (("env.max_consecutive_success=1",), "max_consecutive_success=0"),
        (("env.episode_length_s=20.1",), "camera captures per episode.*frame budget"),
        (("env.scene.tiled_camera.update_period=0.01",), "camera captures per episode.*frame budget"),
        (("env.scene.lazy_sensor_update=False",), "lazy_sensor_update"),
        (("renderer=ovrtx", "env.scene.tiled_camera.renderer_cfg.async_rendering=True"), "synchronous"),
        (("env.feature_extractor.image_update_frames=1",), "image_update_frames"),
    ],
)
def test_cosmos_task_rejects_overrides_exceeding_service_limits_before_simulation(overrides, error):
    """CLI overrides must preserve the service limits and camera supervision timing."""
    env_cfg = parse_env_cfg("Isaac-Reorient-Cube-Shadow-Camera-Direct", overrides=("presets=cosmos", *overrides))

    with pytest.raises(ValueError, match=error):
        env_cfg.validate()


def test_feature_extractor_holds_capture_target_until_new_rgb_and_invalidates_it_on_reset(tmp_path):
    """Real CNN loss uses the held image's target, including when frame one repeats after reset."""
    extractor = FeatureExtractor(
        FeatureExtractorCfg(train=True, image_update_frames=4),
        device="cpu",
        data_types=["rgb"],
        log_dir=str(tmp_path),
        height=64,
        width=64,
    )
    for group in extractor.optimizer.param_groups:
        group["lr"] = 0.0
    rgb = {"rgb": torch.full((1, 64, 64, 3), 90, dtype=torch.uint8)}
    pose = torch.zeros(1, 27)
    first_loss, _ = extractor.step(rgb, pose, camera_frame=torch.tensor([1]))

    for frame in (1, 2, 3, 4):
        held_loss, _ = extractor.step(rgb, torch.full_like(pose, 10.0), camera_frame=torch.tensor([frame]))
        torch.testing.assert_close(held_loss, first_loss)

    updated_loss, _ = extractor.step(rgb, torch.full_like(pose, 10.0), camera_frame=torch.tensor([5]))
    assert updated_loss > first_loss

    extractor.reset(torch.tensor([0]))
    reset_loss, _ = extractor.step(rgb, torch.full_like(pose, 20.0), camera_frame=torch.tensor([1]))
    assert reset_loss > updated_loss

    extractor.reset(torch.tensor([0]))
    repeated_frame_loss, _ = extractor.step(rgb, torch.full_like(pose, 30.0), camera_frame=torch.tensor([1]))
    assert repeated_frame_loss > reset_loss


@pytest.fixture
def mocked_feature_extractor_loading(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list[str]]:
    """Replace the CNN and checkpoint deserializer with lightweight recording doubles."""
    loaded_paths: list[str] = []
    loaded_checkpoints: list[str] = []

    class _FeatureExtractorNetwork:
        def to(self, device: str) -> None:
            pass

        def load_state_dict(self, checkpoint) -> None:
            loaded_checkpoints.append(checkpoint)

        def eval(self) -> None:
            pass

    monkeypatch.setattr(
        feature_extractor_module, "FeatureExtractorNetwork", lambda **kwargs: _FeatureExtractorNetwork()
    )
    monkeypatch.setattr(
        feature_extractor_module.torch,
        "load",
        lambda path, weights_only: loaded_paths.append(path) or "feature extractor weights",
    )
    return loaded_paths, loaded_checkpoints


def test_feature_extractor_fetches_task_configured_pretrained_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mocked_feature_extractor_loading: tuple[list[str], list[str]],
) -> None:
    """The feature extractor must retrieve its own configured checkpoint."""
    retrieved_paths: list[str] = []
    checkpoint_path = tmp_path / "feature_extractor.pth"
    checkpoint_path.touch()

    def _retrieve_file_path(path: str) -> str:
        retrieved_paths.append(path)
        return str(checkpoint_path)

    monkeypatch.setattr(feature_extractor_module, "retrieve_file_path", _retrieve_file_path)
    published_checkpoint = "omniverse://IsaacLab/feature_extractor.pth"
    cfg = FeatureExtractorCfg(
        train=False,
        load_checkpoint=True,
        pretrained_checkpoint=published_checkpoint,
    )

    FeatureExtractor(cfg, "cpu", ["rgb"], str(tmp_path / "logs"))

    loaded_paths, loaded_checkpoints = mocked_feature_extractor_loading
    assert retrieved_paths == [published_checkpoint]
    assert loaded_paths == [str(checkpoint_path)]
    assert loaded_checkpoints == ["feature extractor weights"]


def test_feature_extractor_prefers_local_training_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mocked_feature_extractor_loading: tuple[list[str], list[str]],
) -> None:
    """Local playback must keep loading the CNN checkpoint saved with the training run."""
    local_checkpoint = tmp_path / "cnn_100_loss.pth"
    local_checkpoint.touch()
    monkeypatch.setattr(
        feature_extractor_module,
        "retrieve_file_path",
        lambda path: pytest.fail("The pretrained checkpoint must not be fetched when a local CNN checkpoint exists."),
    )
    cfg = FeatureExtractorCfg(
        train=False,
        load_checkpoint=True,
        pretrained_checkpoint="omniverse://IsaacLab/feature_extractor.pth",
    )

    FeatureExtractor(cfg, "cpu", ["rgb"], str(tmp_path))

    loaded_paths, loaded_checkpoints = mocked_feature_extractor_loading
    assert loaded_paths == [str(local_checkpoint)]
    assert loaded_checkpoints == ["feature extractor weights"]
