# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn and step the MicroDuck variants with native BAM servos on MJWarp."""

import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.cloner import CloneCfg, clone_plan_from_env_0, replicate
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import clone

from isaaclab_assets import (
    MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG,
    MICRODUCK_ALLCOLLISIONS_CFG,
    MICRODUCK_BACKLASH_CFG,
    MICRODUCK_CFG,
    MICRODUCK_ROLLERS_BACKLASH_CFG,
    MICRODUCK_ROLLERS_CFG,
)

pytestmark = [pytest.mark.integration, pytest.mark.kitless]

DT = 0.005
"""Physics timestep [s]."""

NUM_ENVS = 2
"""Environments per scene, so per-environment indexing is observable."""


def _sim_cfg(device: str) -> SimulationCfg:
    """Newton MJWarp with native actuators, sized for the MicroDuck contact count."""
    return SimulationCfg(
        dt=DT,
        device=device,
        use_newton_actuators=True,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=1500, nconmax=200, use_mujoco_contacts=False), num_substeps=1
        ),
    )


def _spawn(sim: SimulationContext, asset_cfg: ArticulationCfg) -> Articulation:
    """Spawn :data:`NUM_ENVS` copies of *asset_cfg* on a ground plane and reset the simulation."""
    for index in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 2.0, 0.0, 0.0))
    cfg = clone(asset_cfg)
    cfg.prim_path = "/World/Env_[^/]*/Robot"
    robot = Articulation(cfg)
    ground_cfg = AssetBaseCfg(prim_path="/World/defaultGroundPlane")
    clone_plan_from_env_0(CloneCfg(clone_template="/World/Env_{}"), [robot.cfg, ground_cfg], NUM_ENVS, 2.0)
    replicate(sim.get_clone_plan())
    sim.reset()
    return robot


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
@pytest.mark.parametrize(
    "asset_cfg, num_joints",
    [
        (MICRODUCK_CFG, 14),
        (MICRODUCK_ALLCOLLISIONS_CFG, 14),
        (MICRODUCK_ROLLERS_CFG, 18),
        (MICRODUCK_BACKLASH_CFG, 28),
        (MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG, 28),
        (MICRODUCK_ROLLERS_BACKLASH_CFG, 32),
    ],
    ids=["walk", "allcollisions", "rollers", "walk-backlash", "allcollisions-backlash", "rollers-backlash"],
)
def test_microduck_native_bam(asset_cfg, num_joints, device):
    """All variants drive 14 servos and step stably; passive wheel and play joints stay undriven."""
    with build_simulation_context(sim_cfg=_sim_cfg(device), add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None
        robot = _spawn(sim, asset_cfg)
        assert robot.num_joints == num_joints
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
            robot.update(DT)
        assert torch.isfinite(robot.data.joint_pos.torch).all()
        assert torch.isfinite(robot.data.joint_vel.torch).all()
        effort = robot.actuators.applied_effort.torch
        assert effort[:, servo_ids].abs().max() > 0.0
        passive_ids, _ = robot.find_joints("passive_.*") if num_joints > 14 else ([], [])
        if passive_ids:
            torch.testing.assert_close(effort[:, passive_ids], torch.zeros_like(effort[:, passive_ids]))


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_microduck_backlash_feedback_reads_each_servos_own_play_hinge(device):
    """Distinct play angles per joint and environment expose wrong hinge pairing or indexing."""
    cfg = clone(MICRODUCK_BACKLASH_CFG)
    servos = cfg.actuators["servos"]
    # Remove supply sampling and command delay so the first-step effort is analytic.
    servos.vin_range = servos.vin_drop_gain_range = None
    servos.min_delay = servos.max_delay = 0
    with build_simulation_context(sim_cfg=_sim_cfg(device), add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None
        robot = _spawn(sim, cfg)
        servo_ids, _ = robot.find_joints("^(?!passive_).*")
        play_ids = [robot.joint_names.index(f"passive_{robot.joint_names[j]}_backlash") for j in servo_ids]
        q = torch.zeros_like(robot.data.default_joint_pos.torch)
        qd = torch.zeros_like(q)
        play = torch.linspace(-0.01, 0.01, NUM_ENVS * 14, device=device).reshape(NUM_ENVS, 14)
        q[:, play_ids] = play
        qd[:, play_ids] = 0.2
        robot.write_joint_state_to_sim_index(position=q, velocity=qd)
        robot.actuators.target_command.set_position_index(value=torch.zeros_like(q))
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(DT)

        # At rest, the duty cycle only sees the play angle; play-hinge velocity adds no back-EMF.
        motor = servos.motor
        expected = -play * servos.kp_fw * motor.error_gain * servos.vin * motor.kt / motor.resistance
        torch.testing.assert_close(robot.actuators.applied_effort.torch[:, servo_ids], expected, atol=1e-6, rtol=1e-5)
