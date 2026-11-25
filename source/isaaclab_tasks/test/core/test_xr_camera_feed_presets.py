# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
from isaaclab_isaacsim.renderers import IsaacRtxRendererCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.renderers import OVRTXRendererCfg

from isaaclab.app import validate_launch
from isaaclab.sim import SimulationCfg

from isaaclab_tasks.contrib.locomanip_pick_place.fixed_base_upper_body_ik_g1_env_cfg import (
    FixedBaseUpperBodyIKG1EnvCfg,
)
from isaaclab_tasks.contrib.locomanip_pick_place.locomanipulation_g1_env_cfg import LocomanipulationG1EnvCfg
from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import PickPlaceGR1T2EnvCfg
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
    assert cfg.scene.robot_pov_cam.renderer_cfg.isaacsim_rtx == IsaacRtxRendererCfg()
    cfg = resolve_presets(cfg)
    validate_launch(cfg, {"visualizer": []})
    camera_cfg = cfg.scene.robot_pov_cam
    # The head panel is part of every feed selection.
    feed_cfg = resolve_presets(cfg.isaac_teleop.xr_camera_feeds)[0]

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
    assert camera_cfg.renderer_cfg == OVRTXRendererCfg()


@pytest.mark.parametrize("selection", [[], ["presets=rgb"]])
def test_fixed_base_g1_wrist_cameras_and_policy_images_are_opt_in(selection):
    """The state-only default renders only the head camera; ``presets=rgb`` adds the wrist views."""
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(
        "IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs", device="cpu", num_envs=1, overrides=selection
    )

    rgb = bool(selection)
    cameras = ["robot_pov_cam", "left_wrist_cam", "right_wrist_cam"] if rgb else ["robot_pov_cam"]
    # Image consumers use only the names present in the policy observations.
    assert cfg.image_obs_list == ["robot_pov_cam", "left_wrist_cam", "right_wrist_cam"]
    assert cfg.scene.robot_pov_cam is not None
    for name in ("left_wrist_cam", "right_wrist_cam"):
        assert (getattr(cfg.scene, name) is not None) is rgb
    for name in ("robot_pov_cam", "left_wrist_cam", "right_wrist_cam"):
        assert hasattr(cfg.observations.policy, name) is rgb
    assert cfg.num_rerenders_on_reset == (3 if rgb else 0)
    assert cfg.scene.robot.spawn.fix_root_link is True
    assert [feed.camera_name for feed in cfg.isaac_teleop.xr_camera_feeds] == cameras
    assert [feed.max_update_hz for feed in cfg.isaac_teleop.xr_camera_feeds] == [20.0, 10.0, 10.0][: len(cameras)]
    assert cfg.xr.excluded_visual_meshes


def test_locomanipulation_g1_retains_recorded_camera():
    """Locomanipulation retains its recorded camera while presenting it in XR."""
    cfg = LocomanipulationG1EnvCfg()

    assert hasattr(cfg.scene, "robot_pov_cam")
    assert hasattr(cfg.observations.policy, "robot_pov_cam")
    assert cfg.image_obs_list == ["robot_pov_cam"]


def test_xr_camera_reference_renderer_resolves_for_supported_backends():
    """Reference cameras default to OVRTX and retain explicit Isaac RTX compatibility."""
    camera_renderer_cfg = PickPlaceGR1T2EnvCfg().scene.robot_pov_cam.renderer_cfg
    default = SimpleNamespace(sim=SimulationCfg(physics=OvPhysxCfg()), renderer=resolve_presets(camera_renderer_cfg))
    validate_launch(default)
    isaacsim_rtx = resolve_presets(
        camera_renderer_cfg,
        selected=("isaacsim_rtx",),
    )
    ovrtx = resolve_presets(camera_renderer_cfg, selected=("ovrtx",))
    expected_rtx = IsaacRtxRendererCfg()

    assert default.renderer == OVRTXRendererCfg()
    assert isaacsim_rtx == expected_rtx
    assert isinstance(ovrtx, OVRTXRendererCfg)


def test_fixed_base_g1_direct_construction_prepares_the_state_policy_default():
    """Registered construction without task resolution uses the state-only policy preset default."""
    from isaaclab.managers import ObservationGroupCfg

    from isaaclab_tasks.contrib.teleop_physics_cfg import prepare_humanoid_teleop_environment

    cfg = FixedBaseUpperBodyIKG1EnvCfg()
    prepare_humanoid_teleop_environment(cfg)
    assert isinstance(cfg.observations.policy, ObservationGroupCfg)
    assert not hasattr(cfg.observations.policy, "robot_pov_cam")
    assert hasattr(cfg.observations.policy, "robot_joint_pos")
