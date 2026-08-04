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


@pytest.mark.parametrize("env_cfg_type", [PickPlaceGR1T2EnvCfg, LocomanipulationG1EnvCfg])
def test_xr_camera_reference_tasks_select_recorded_camera(env_cfg_type):
    """Reference tasks record and present the same camera after a temporal reset refresh."""
    cfg = env_cfg_type()

    assert cfg.isaac_teleop.xr_camera_feeds[0].camera_name == "robot_pov_cam"
    assert hasattr(cfg.observations.policy, "robot_pov_cam")
    assert isinstance(cfg.scene.robot_pov_cam.renderer_cfg, MultiBackendRendererCfg)
    assert cfg.num_rerenders_on_reset == 3


@pytest.mark.parametrize("env_cfg_type", [PickPlaceGR1T2EnvCfg, LocomanipulationG1EnvCfg])
def test_xr_camera_reference_renderer_resolves_for_supported_backends(env_cfg_type):
    """Reference cameras retain Isaac RTX defaults and OVRTX compatibility."""
    default = resolve_presets(env_cfg_type().scene.robot_pov_cam.renderer_cfg)
    isaacsim_rtx = resolve_presets(
        env_cfg_type().scene.robot_pov_cam.renderer_cfg,
        selected=("isaacsim_rtx",),
    )
    ovrtx = resolve_presets(env_cfg_type().scene.robot_pov_cam.renderer_cfg, selected=("ovrtx",))
    expected_rtx = IsaacRtxRendererCfg(
        enable_dlss_ray_reconstruction=True,
        dlss_exec_mode="quality",
    )

    assert default == expected_rtx
    assert isaacsim_rtx == expected_rtx
    assert isinstance(ovrtx, OVRTXRendererCfg)
