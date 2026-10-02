# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate MicroDuck USDs and native BAM bindings on MJWarp."""

import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.cloner import CloneCfg, clone_plan_from_env_0, replicate
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import clone

from isaaclab_assets import MICRODUCK_ALLCOLLISIONS_CFG, MICRODUCK_CFG, MICRODUCK_ROLLERS_CFG

pytestmark = [pytest.mark.integration, pytest.mark.kitless]


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
@pytest.mark.parametrize(
    "asset_cfg, num_joints",
    [(MICRODUCK_CFG, 14), (MICRODUCK_ALLCOLLISIONS_CFG, 14), (MICRODUCK_ROLLERS_CFG, 18)],
    ids=["walk", "allcollisions", "rollers"],
)
def test_microduck_native_bam(asset_cfg, num_joints, device):
    """All variants spawn and step with 14 driven servos; roller wheel joints stay passive."""
    sim_cfg = SimulationCfg(
        dt=0.005,
        device=device,
        use_newton_actuators=True,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=1500, nconmax=200, use_mujoco_contacts=False),
            num_substeps=1,
        ),
    )
    with build_simulation_context(sim_cfg=sim_cfg, add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None
        for index in range(2):
            sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 2.0, 0.0, 0.0))
        cfg = clone(asset_cfg)
        cfg.prim_path = "/World/Env_[^/]*/Robot"
        robot = Articulation(cfg)
        ground_cfg = AssetBaseCfg(prim_path="/World/defaultGroundPlane")
        clone_plan_from_env_0(CloneCfg(clone_template="/World/Env_{}"), [robot.cfg, ground_cfg], 2, 2.0)
        replicate(sim.get_clone_plan())
        sim.reset()
        assert robot.is_initialized
        assert robot.num_joints == num_joints
        assert "servos" in robot.actuators._native_group_names
        servo_ids, _ = robot.find_joints("^(?!passive_).*")
        assert len(servo_ids) == 14
        assert (robot.data.joint_armature.torch[:, servo_ids] > 0.0).all()
        assert (robot.data.joint_friction_coeff.torch[:, servo_ids] > 0.0).all()
        target = robot.data.default_joint_pos.torch.clone()
        target[:, servo_ids] += 0.02
        robot.actuators.target_command.set_position_index(value=target)
        robot.write_data_to_sim()
        for _ in range(20):
            sim.step(render=False)
            robot.update(sim_cfg.dt)
        assert torch.isfinite(robot.data.joint_pos.torch).all()
        assert torch.isfinite(robot.data.joint_vel.torch).all()
        effort = robot.actuators.applied_effort.torch
        assert effort[:, servo_ids].abs().max() > 0.0
        wheel_ids, _ = robot.find_joints("passive_.*") if num_joints > 14 else ([], [])
        if wheel_ids:
            torch.testing.assert_close(effort[:, wheel_ids], torch.zeros_like(effort[:, wheel_ids]))
