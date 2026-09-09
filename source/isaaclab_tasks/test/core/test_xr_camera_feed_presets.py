# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

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
    assert hasattr(cfg.observations.policy, "robot_pov_cam")
    assert cfg.scene.robot_pov_cam.prim_path == "{ENV_REGEX_NS}/Robot/base_link/RobotPOVCam"
    assert isinstance(cfg.scene.robot_pov_cam.renderer_cfg, MultiBackendRendererCfg)
    assert cfg.isaac_teleop.xr_camera_feeds[0].enable_dlss_ray_reconstruction is True
    assert cfg.isaac_teleop.xr_camera_feeds[0].dlss_exec_mode == "quality"
    assert cfg.num_rerenders_on_reset == 3


def test_locomanipulation_g1_recorded_camera_uses_calibration():
    """Locomanipulation G1 retains its calibrated camera for recorded observations."""
    cfg = LocomanipulationG1EnvCfg()
    camera_cfg = cfg.scene.robot_pov_cam

    assert camera_cfg.prim_path == "{ENV_REGEX_NS}/Robot/torso_link/head_link/RobotHeadCam"
    assert (camera_cfg.width, camera_cfg.height) == (640, 480)
    assert camera_cfg.spawn.focal_length == 15.0
    assert camera_cfg.spawn.horizontal_aperture == 20.955
    assert camera_cfg.spawn.clipping_range == (0.1, 5.0)
    assert camera_cfg.offset.pos == (0.04485, 0.0, 0.35325)
    assert camera_cfg.offset.rot == (-0.62721, 0.62721, -0.32651, 0.32651)
    assert camera_cfg.offset.convention == "ros"


@pytest.mark.parametrize("env_cfg_type", [LocomanipulationG1EnvCfg, FixedBaseUpperBodyIKG1EnvCfg])
def test_g1_tasks_do_not_enable_xr_camera_pip(env_cfg_type):
    """G1 tasks do not present a robot camera in XR."""
    cfg = env_cfg_type()

    assert cfg.isaac_teleop.xr_camera_feeds == []


def test_locomanipulation_g1_retains_recorded_camera():
    """Locomanipulation retains its recorded camera without presenting it in XR."""
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
