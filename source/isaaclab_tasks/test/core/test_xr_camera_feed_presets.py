# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_teleop import XrCameraFeedSession
from packaging.version import Version

from isaaclab_tasks.contrib.locomanip_pick_place.fixed_base_upper_body_ik_g1_env_cfg import (
    FixedBaseUpperBodyIKG1EnvCfg,
)
from isaaclab_tasks.contrib.locomanip_pick_place.locomanipulation_g1_env_cfg import LocomanipulationG1EnvCfg
from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import PickPlaceGR1T2EnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg


def test_xr_camera_reference_task_selects_recorded_camera():
    """Reference tasks use a camera beneath a prim that inherits physical-body motion."""
    cfg = PickPlaceGR1T2EnvCfg()

    assert cfg.isaac_teleop.xr_camera_feeds[0].camera_name == "robot_pov_cam"
    assert cfg.isaac_teleop.xr_camera_feed_layout.use_scene_partition is False
    assert cfg.scene.robot_pov_cam.renderer_cfg.default.enable_scene_partitioning == (
        IsaacRtxRendererCfg().enable_scene_partitioning
    )
    assert hasattr(cfg.observations.policy, "robot_pov_cam")
    assert cfg.scene.robot_pov_cam.prim_path == "{ENV_REGEX_NS}/Robot/base_link/RobotPOVCam"
    assert isinstance(cfg.scene.robot_pov_cam.renderer_cfg, MultiBackendRendererCfg)
    assert cfg.isaac_teleop.xr_camera_feeds[0].enable_dlss_ray_reconstruction is True
    assert cfg.isaac_teleop.xr_camera_feeds[0].dlss_exec_mode == "quality"
    assert cfg.num_rerenders_on_reset == 3


def test_locomanipulation_g1_xr_camera_uses_calibration_and_head_locked_panel():
    """Locomanipulation G1 presents its calibrated camera in a headset-following panel."""
    cfg = LocomanipulationG1EnvCfg()
    camera_cfg = cfg.scene.robot_pov_cam
    feed_cfg = cfg.isaac_teleop.xr_camera_feeds[0]

    assert feed_cfg.camera_name == "robot_pov_cam"
    assert camera_cfg.prim_path == "{ENV_REGEX_NS}/Robot/torso_link/head_link/RobotHeadCam"
    assert (camera_cfg.width, camera_cfg.height) == (640, 480)
    assert camera_cfg.spawn.focal_length == 15.0
    assert camera_cfg.spawn.horizontal_aperture == 20.955
    assert camera_cfg.spawn.clipping_range == (0.1, 5.0)
    assert camera_cfg.offset.pos == (0.04485, 0.0, 0.35325)
    assert camera_cfg.offset.rot == (-0.62721, 0.62721, -0.32651, 0.32651)
    assert camera_cfg.offset.convention == "ros"
    assert cfg.isaac_teleop.xr_camera_feed_layout.placement == "head_locked"
    assert cfg.isaac_teleop.xr_camera_feed_layout.use_scene_partition is True
    assert feed_cfg.offset_m == (0.0, -0.15)
    assert feed_cfg.enable_dlss_ray_reconstruction is True
    assert feed_cfg.dlss_exec_mode == "quality"
    assert feed_cfg.max_update_hz == 0.0
    assert camera_cfg.renderer_cfg.default == IsaacRtxRendererCfg()
    assert camera_cfg.renderer_cfg.isaacsim_rtx == IsaacRtxRendererCfg()


def test_g1_partition_overrides_apply_only_during_enabled_pip_preparation(monkeypatch):
    monkeypatch.setattr("isaaclab_teleop.camera_feed._load_kit_scene_ui_presenter", lambda: object())
    monkeypatch.setattr("isaaclab_teleop.camera_feed.get_isaac_sim_version", lambda: Version("6.1.0"))
    cfg = LocomanipulationG1EnvCfg()
    cfg.scene.num_envs = 1
    camera = cfg.scene.robot_pov_cam

    XrCameraFeedSession.prepare(cfg, enabled=False, camera_rendering_enabled=True)
    assert camera.renderer_cfg.default == IsaacRtxRendererCfg()
    assert camera.renderer_cfg.isaacsim_rtx == IsaacRtxRendererCfg()

    XrCameraFeedSession.prepare(cfg, enabled=True, camera_rendering_enabled=True)
    for renderer in (camera.renderer_cfg.default, camera.renderer_cfg.isaacsim_rtx):
        assert renderer.enable_scene_partitioning is False
        assert renderer.global_settings.show_all_partitions_by_default is False


def test_fixed_base_g1_does_not_enable_xr_camera_pip():
    """The fixed-base G1 task does not present a robot camera in XR."""
    cfg = FixedBaseUpperBodyIKG1EnvCfg()

    assert cfg.isaac_teleop.xr_camera_feeds == []


def test_locomanipulation_g1_retains_recorded_camera():
    """Locomanipulation retains its recorded camera while presenting it in XR."""
    cfg = LocomanipulationG1EnvCfg()

    assert hasattr(cfg.scene, "robot_pov_cam")
    assert hasattr(cfg.observations.policy, "robot_pov_cam")
    assert cfg.image_obs_list == ["robot_pov_cam"]


def test_xr_camera_reference_renderer_resolves_for_supported_backends():
    """Reference cameras retain Isaac RTX defaults and OVRTX compatibility."""
    camera_renderer_cfg = PickPlaceGR1T2EnvCfg().scene.robot_pov_cam.renderer_cfg
    default = resolve_presets(camera_renderer_cfg)
    isaacsim_rtx = resolve_presets(
        camera_renderer_cfg,
        selected=("isaacsim_rtx",),
    )
    ovrtx = resolve_presets(camera_renderer_cfg, selected=("ovrtx",))
    expected_rtx = IsaacRtxRendererCfg()

    assert default == expected_rtx
    assert isaacsim_rtx == expected_rtx
    assert isinstance(ovrtx, OVRTXRendererCfg)
