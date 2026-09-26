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


def test_xr_camera_reference_task_selects_recorded_camera():
    """Reference tasks use a camera beneath a prim that inherits physical-body motion."""
    cfg = PickPlaceGR1T2EnvCfg()
    camera_name = cfg.isaac_teleop.xr_camera_feeds[0].camera_name

    assert hasattr(cfg.observations.policy, camera_name)
    assert getattr(cfg.scene, camera_name).prim_path.startswith("{ENV_REGEX_NS}/Robot/")


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
