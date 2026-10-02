# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn and step the MicroDuck variants with native BAM servos on MJWarp."""

import numpy as np
import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.cloner import CloneCfg, clone_plan_from_env_0, replicate
from isaaclab.sim import SimulationCfg, build_simulation_context
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
    """All variants drive 14 servos; play hinges affect feedback and passive joints stay undriven."""
    has_backlash = num_joints >= 28
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
        if has_backlash:
            cfg.actuators["servos"].vin_range = None
            cfg.actuators["servos"].vin_drop_gain_range = None
            cfg.actuators["servos"].min_delay = cfg.actuators["servos"].max_delay = 0
        cfg.prim_path = "/World/Env_[^/]*/Robot"
        robot = Articulation(cfg)
        ground_cfg = AssetBaseCfg(prim_path="/World/defaultGroundPlane")
        clone_plan_from_env_0(CloneCfg(clone_template="/World/Env_{}"), [robot.cfg, ground_cfg], 2, 2.0)
        replicate(sim.get_clone_plan())
        sim.reset()
        assert robot.is_initialized
        assert robot.num_joints == num_joints
        servo_ids, _ = robot.find_joints("^(?!passive_).*")
        assert len(servo_ids) == 14
        assert (robot.data.joint_armature.torch[:, servo_ids] > 0.0).all()
        assert (robot.data.joint_friction_coeff.torch[:, servo_ids] > 0.0).all()
        if has_backlash:
            solver = NewtonManager._solver
            play_prims = {p.GetName(): p for p in stage.Traverse() if p.GetName().endswith("_backlash")}
            properties = ("jnt_range", "jnt_solref", "jnt_solimp", "dof_armature", "dof_damping")
            live = {name: getattr(solver.mjw_model, name).numpy() for name in properties}
            checked = set()
            for mj_joint, newton_joint in enumerate(solver.mjc_jnt_to_newton_jnt.numpy()[0]):
                if newton_joint < 0:
                    continue
                name = NewtonManager.backend.model.joint_label[newton_joint].rsplit("/", 1)[-1]
                if name not in play_prims:
                    continue
                prim = play_prims[name]
                checked.add(name)
                dof = solver.mj_model.jnt_dofadr[mj_joint]
                expected_fields = (
                    np.deg2rad([prim.GetAttribute(f"physics:{bound}Limit").Get() for bound in ("lower", "upper")]),
                    prim.GetAttribute("mjc:solreflimit").Get(),
                    prim.GetAttribute("mjc:solimplimit").Get(),
                    prim.GetAttribute("physxJoint:armature").Get(),
                    prim.GetAttribute("mjc:damping").Get(),
                )
                for field, expected in zip(properties, expected_fields):
                    actual = live[field][:, mj_joint if field.startswith("jnt_") else dof]
                    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), rtol=1e-6)
            assert len(checked) == 14
            # Distinct play angles expose incorrect joint pairing and cross-environment indexing.
            play_ids = [robot.joint_names.index(f"passive_{robot.joint_names[j]}_backlash") for j in servo_ids]
            q = torch.zeros_like(robot.data.default_joint_pos.torch)
            qd = torch.zeros_like(q)
            play = torch.linspace(-0.01, 0.01, 28, device=device).reshape(2, 14)
            q[:, play_ids] = play
            qd[:, play_ids] = 0.2
            robot.write_joint_state_to_sim_index(position=q, velocity=qd)
            robot.actuators.target_command.set_position_index(value=torch.zeros_like(q))
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(sim_cfg.dt)
            motor = servo_cfg.motor
            expected = -play * servo_cfg.kp_fw * motor.error_gain * servo_cfg.vin * motor.kt / motor.resistance
            torch.testing.assert_close(
                robot.actuators.applied_effort.torch[:, servo_ids], expected, atol=1e-6, rtol=1e-5
            )
        target = robot.data.default_joint_pos.torch.clone()
        target[:, servo_ids] += 0.02
        robot.actuators.target_command.set_position_index(value=target)
        robot.write_data_to_sim()
        for _ in range(20):
            sim.step(render=False)
            robot.update(sim_cfg.dt)
        assert NewtonManager._graph is not None
        assert torch.isfinite(robot.data.joint_pos.torch).all()
        assert torch.isfinite(robot.data.joint_vel.torch).all()
        effort = robot.actuators.applied_effort.torch
        assert effort[:, servo_ids].abs().max() > 0.0
        passive_ids, _ = robot.find_joints("passive_.*") if num_joints > 14 else ([], [])
        if passive_ids:
            torch.testing.assert_close(effort[:, passive_ids], torch.zeros_like(effort[:, passive_ids]))
