# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused regressions for the G1 kitless OVPhysX packing-table contract."""

import gymnasium as gym
import pytest
import torch
from isaaclab_ov.physics import OvPhysxCfg

from isaaclab.app import launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.locomanip_pick_place.fixed_base_upper_body_ik_g1_env_cfg import (
    FixedBaseUpperBodyIKG1EnvCfg,
)
from isaaclab_tasks.contrib.locomanip_pick_place.locomanipulation_g1_env_cfg import LocomanipulationG1EnvCfg
from isaaclab_tasks.contrib.packing_table import PackingTableUsdFileCfg

_TASK = "IsaacContrib-PickPlace-Locomanipulation-G1-Abs"


def test_fixed_base_g1_ovphysx_conditions_assets_without_changing_other_backends() -> None:
    cfg = FixedBaseUpperBodyIKG1EnvCfg()
    default_cfg = FixedBaseUpperBodyIKG1EnvCfg()
    action_type = cfg.actions.upper_body_ik.class_type
    drop_term = cfg.terminations.object_dropping.func

    cfg.stabilize_for_kitless_ovphysx()
    table_spawn = cfg.scene.packing_table.spawn.func
    wheel_spawn = cfg.scene.object.spawn.func
    cfg.stabilize_for_kitless_ovphysx()

    assert isinstance(cfg.scene.packing_table.spawn, PackingTableUsdFileCfg)
    assert cfg.scene.object.spawn.func == default_cfg.scene.object.spawn.func
    assert cfg.scene.object.spawn.usd_path == default_cfg.scene.object.spawn.usd_path
    assert cfg.scene.packing_table.spawn.func is table_spawn
    assert cfg.scene.object.spawn.func is wheel_spawn
    assert cfg.actions.upper_body_ik.class_type is action_type
    assert cfg.terminations.object_dropping.func is drop_term
    assert not isinstance(default_cfg.scene.packing_table.spawn, PackingTableUsdFileCfg)
    assert not isinstance(default_cfg.scene.object.spawn, PackingTableUsdFileCfg)


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize(
    ("cfg_type", "task"),
    [
        (LocomanipulationG1EnvCfg, _TASK),
        (FixedBaseUpperBodyIKG1EnvCfg, "IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs"),
    ],
    ids=["locomanipulation", "fixed_base"],
)
def test_g1_wheel_remains_supported_on_ovphysx_cpu(cfg_type, task) -> None:
    pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

    env_cfg = cfg_type()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = "cpu"
    env_cfg.sim.physics = OvPhysxCfg()
    env_cfg.scene.robot_pov_cam = None
    env_cfg.observations.policy.robot_pov_cam = None
    if isinstance(env_cfg, FixedBaseUpperBodyIKG1EnvCfg):
        env_cfg.scene.left_wrist_cam = None
        env_cfg.scene.right_wrist_cam = None
        env_cfg.observations.policy.left_wrist_cam = None
        env_cfg.observations.policy.right_wrist_cam = None
    env_cfg.image_obs_list = []
    env_cfg.stabilize_for_kitless_ovphysx()

    env = None
    launcher_args = {
        "physics": None,
        "device": "cpu",
        "visualizer": None,
        "visualizer_explicit": True,
    }
    try:
        with launch_simulation(env_cfg, launcher_args):
            env = gym.make(task, cfg=env_cfg).unwrapped
            phases = ("table", "reset", "bin") if cfg_type is FixedBaseUpperBodyIKG1EnvCfg else ("table",)
            for phase in phases:
                env.reset()
                wheel = env.scene["object"]
                if phase == "bin":
                    from pxr import Usd, UsdGeom

                    from isaaclab.sim import get_current_stage

                    floor = get_current_stage().GetPrimAtPath(
                        "/World/envs/env_0/PackingTable/container_h20/container_h20_inst/colliders/Cube"
                    )
                    assert floor.IsValid()
                    # Include invisible collision geometry when measuring the bin floor.
                    bounds = (
                        UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], False, True)
                        .ComputeWorldBound(floor)
                        .ComputeAlignedRange()
                    )
                    center = bounds.GetMidpoint()
                    floor_top = bounds.GetMax()[2]
                    pose = wheel.data.root_pose_w.torch.clone()
                    pose[0, :3] = torch.tensor([center[0], center[1], floor_top + 0.10])
                    wheel.write_root_pose_to_sim_index(root_pose=pose)
                    wheel.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6)))

                minimum_z = float("inf")
                with torch.inference_mode():
                    for _ in range(300):
                        env.sim.step(render=False)
                        env.scene.update(env_cfg.sim.dt)
                        minimum_z = min(minimum_z, float(wheel.data.root_pos_w.torch[0, 2].item()))
                        assert torch.isfinite(wheel.data.root_pose_w.torch).all()
                        assert torch.isfinite(wheel.data.root_vel_w.torch).all()
                        object_dropping = env_cfg.terminations.object_dropping.func(
                            env, **env_cfg.terminations.object_dropping.params
                        )
                        assert not bool(object_dropping[0].item()), phase

                assert 300 * env_cfg.sim.dt == pytest.approx(1.5)
                assert minimum_z > env_cfg.terminations.object_dropping.params["minimum_height"]
                if phase == "bin":
                    final_position = wheel.data.root_pos_w.torch[0]
                    assert float(final_position[2]) >= floor_top - 0.005
                    for axis in (0, 1):
                        assert bounds.GetMin()[axis] < float(final_position[axis]) < bounds.GetMax()[axis]
    finally:
        if env is not None:
            env.close()
