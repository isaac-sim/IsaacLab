# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from isaaclab_tasks.contrib.locomanip_pick_place.locomanipulation_g1_env_cfg import LocomanipulationG1EnvCfg
from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import PickPlaceGR1T2EnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg


@pytest.mark.parametrize(
    ("env_cfg_type", "camera_prim_path"),
    [
        (PickPlaceGR1T2EnvCfg, "{ENV_REGEX_NS}/Robot/base_link/RobotPOVCam"),
        (LocomanipulationG1EnvCfg, "{ENV_REGEX_NS}/Robot/torso_link/head_link/RobotHeadCam"),
    ],
)
def test_xr_camera_reference_tasks_select_recorded_camera(env_cfg_type, camera_prim_path):
    """Reference tasks use a camera beneath a prim that inherits physical-body motion."""
    cfg = env_cfg_type()

    assert cfg.isaac_teleop.xr_camera_feeds[0].camera_name == "robot_pov_cam"
    assert hasattr(cfg.observations.policy, "robot_pov_cam")
    assert cfg.scene.robot_pov_cam.prim_path == camera_prim_path
    assert isinstance(cfg.scene.robot_pov_cam.renderer_cfg, MultiBackendRendererCfg)
    assert cfg.isaac_teleop.xr_camera_feeds[0].enable_dlss_ray_reconstruction is True
    assert cfg.isaac_teleop.xr_camera_feeds[0].dlss_exec_mode == "quality"
    assert cfg.num_rerenders_on_reset == 3


def test_g1_xr_camera_uses_calibration_and_viewer_start_panel():
    """G1 uses its calibrated head camera and the GR1T2-style panel placement."""
    cfg = LocomanipulationG1EnvCfg()
    camera_cfg = cfg.scene.robot_pov_cam
    feed_cfg = cfg.isaac_teleop.xr_camera_feeds[0]

    assert camera_cfg.prim_path == "{ENV_REGEX_NS}/Robot/torso_link/head_link/RobotHeadCam"
    assert (camera_cfg.width, camera_cfg.height) == (640, 480)
    assert camera_cfg.spawn.focal_length == 15.0
    assert camera_cfg.spawn.horizontal_aperture == 20.955
    assert camera_cfg.spawn.clipping_range == (0.1, 5.0)
    assert camera_cfg.offset.pos == (0.04485, 0.0, 0.35325)
    assert camera_cfg.offset.rot == (-0.62721, 0.62721, -0.32651, 0.32651)
    assert camera_cfg.offset.convention == "ros"
    assert cfg.isaac_teleop.xr_camera_feed_layout.placement == "viewer_start"
    assert feed_cfg.offset_m == (0.0, -0.15)


@pytest.mark.parametrize("env_cfg_type", [PickPlaceGR1T2EnvCfg, LocomanipulationG1EnvCfg])
def test_xr_camera_reference_renderer_resolves_for_supported_backends(env_cfg_type):
    """Reference cameras retain Isaac RTX defaults and OVRTX compatibility."""
    default = resolve_presets(env_cfg_type().scene.robot_pov_cam.renderer_cfg)
    isaacsim_rtx = resolve_presets(
        env_cfg_type().scene.robot_pov_cam.renderer_cfg,
        selected=("isaacsim_rtx",),
    )
    ovrtx = resolve_presets(env_cfg_type().scene.robot_pov_cam.renderer_cfg, selected=("ovrtx",))
    expected_rtx = IsaacRtxRendererCfg()

    assert default == expected_rtx
    assert isaacsim_rtx == expected_rtx
    assert isinstance(ovrtx, OVRTXRendererCfg)
