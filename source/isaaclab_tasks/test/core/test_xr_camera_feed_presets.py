# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import re

import numpy as np
import pytest
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_teleop import xr_anchor_manager

from isaaclab_tasks.contrib.locomanip_pick_place.fixed_base_upper_body_ik_g1_env_cfg import (
    FixedBaseUpperBodyIKG1EnvCfg,
)
from isaaclab_tasks.contrib.locomanip_pick_place.locomanipulation_g1_env_cfg import LocomanipulationG1EnvCfg
from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import PickPlaceGR1T2EnvCfg
from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_waist_enabled_env_cfg import PickPlaceGR1T2WaistEnabledEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets


def test_xr_camera_reference_task_selects_recorded_camera():
    """Reference tasks use a camera beneath a prim that inherits physical-body motion."""
    cfg = PickPlaceGR1T2EnvCfg()
    camera_name = cfg.isaac_teleop.xr_camera_feeds[0].camera_name

    assert hasattr(cfg.observations.policy, camera_name)
    assert getattr(cfg.scene, camera_name).prim_path.startswith("{ENV_REGEX_NS}/Robot/")
    assert cfg.isaac_teleop.xr_camera_feed_layout.use_scene_partition is False


@pytest.mark.parametrize("env_cfg_type", [LocomanipulationG1EnvCfg, FixedBaseUpperBodyIKG1EnvCfg])
def test_g1_xr_camera_uses_calibration_and_head_locked_panel(env_cfg_type):
    """Both G1 tasks present the calibrated camera in a headset-following panel."""
    cfg = env_cfg_type()
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
    assert feed_cfg.panel_width_m == 0.48
    assert feed_cfg.distance_m == 0.8
    assert feed_cfg.enable_dlss_ray_reconstruction is True
    assert feed_cfg.dlss_exec_mode == "quality"
    assert feed_cfg.max_update_hz == (20.0 if env_cfg_type is FixedBaseUpperBodyIKG1EnvCfg else 0.0)
    assert camera_cfg.renderer_cfg.default == IsaacRtxRendererCfg()
    assert camera_cfg.renderer_cfg.isaacsim_rtx == IsaacRtxRendererCfg()


def test_fixed_base_g1_pip_preserves_policy_observations_and_fixed_root():
    """Retain the existing three-camera observation contract and fixed-base root."""
    cfg = FixedBaseUpperBodyIKG1EnvCfg()

    assert hasattr(cfg.scene, "robot_pov_cam")
    for name in ("robot_pov_cam", "left_wrist_cam", "right_wrist_cam"):
        assert hasattr(cfg.observations.policy, name)
    assert cfg.scene.robot.spawn.fix_root_link is True
    assert [feed.camera_name for feed in cfg.isaac_teleop.xr_camera_feeds] == [
        "robot_pov_cam",
        "left_wrist_cam",
        "right_wrist_cam",
    ]
    assert [feed.max_update_hz for feed in cfg.isaac_teleop.xr_camera_feeds] == [20.0, 10.0, 10.0]
    assert cfg.xr.hidden_visual_prims == []
    assert cfg.xr.excluded_visual_meshes


def test_locomanipulation_g1_retains_recorded_camera():
    """Locomanipulation retains its recorded camera while presenting it in XR."""
    cfg = LocomanipulationG1EnvCfg()

    assert hasattr(cfg.scene, "robot_pov_cam")
    assert hasattr(cfg.observations.policy, "robot_pov_cam")
    assert cfg.image_obs_list == ["robot_pov_cam"]


@pytest.mark.parametrize("env_cfg_type", [PickPlaceGR1T2EnvCfg, PickPlaceGR1T2WaistEnabledEnvCfg])
def test_gr1t2_xr_anchor_preserves_operator_view_and_head_partition(monkeypatch, env_cfg_type):
    """The operator faces the table; head exclusion remains camera-selective."""
    cfg = env_cfg_type()
    assert cfg.xr.anchor_pos == (0.0, 0.0, 0.0)
    assert cfg.xr.render_anchor_pos is None
    assert cfg.xr.render_anchor_rot is None
    assert not cfg.xr.hidden_visual_prims
    assert any(
        re.fullmatch(pattern, "/World/envs/env_0/Robot/head_yaw_link/visuals/head_yaw_link/mesh")
        for pattern in cfg.xr.excluded_visual_meshes
    )
    monkeypatch.setattr(xr_anchor_manager, "XRCore", None)
    monkeypatch.setattr(xr_anchor_manager, "_xr_anchor_prim_exists", lambda _path: True)
    anchor = xr_anchor_manager.XrAnchorManager(cfg.xr)
    assert anchor.render_anchor_path == anchor.anchor_headset_path
    transform = anchor.get_world_matrix()
    eye = transform @ np.array([0.0, 1.6, 0.0, 1.0])
    forward = transform @ np.array([0.0, 0.0, -1.0, 0.0])
    np.testing.assert_allclose(eye, (0.0, 0.0, 1.6, 1.0))
    np.testing.assert_allclose(forward, (0.0, 1.0, 0.0, 0.0))
    assert np.dot(np.asarray(cfg.scene.packing_table.init_state.pos) - eye[:3], forward[:3]) > 0.0


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
