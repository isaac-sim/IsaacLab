# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that ``mjc:*`` USD attributes reach the Newton model through the manager import path."""

import pytest
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager

from pxr import Sdf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, build_simulation_context


def _author_revolute_with_frictionloss(stage, frictionloss: float) -> None:
    """Author a two-body articulation whose revolute joint carries ``mjc:frictionloss``."""
    world = UsdGeom.Xform.Define(stage, "/World/robot")
    UsdPhysics.ArticulationRootAPI.Apply(world.GetPrim())
    for name in ("parent", "child"):
        body = UsdGeom.Cube.Define(stage, f"/World/robot/{name}")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(1.0)
    fixed = UsdPhysics.FixedJoint.Define(stage, "/World/robot/fix")
    fixed.CreateBody1Rel().SetTargets(["/World/robot/parent"])
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/robot/rev")
    joint.CreateBody0Rel().SetTargets(["/World/robot/parent"])
    joint.CreateBody1Rel().SetTargets(["/World/robot/child"])
    joint.CreateAxisAttr("Z")
    joint.GetPrim().CreateAttribute("mjc:frictionloss", Sdf.ValueTypeNames.Float, custom=True).Set(frictionloss)


def test_usd_mjc_frictionloss_reaches_newton_model():
    """A USD joint authored with ``mjc:frictionloss`` yields a matching ``joint_friction``."""
    sim_cfg = SimulationCfg(
        dt=0.005,
        device="cuda:0",
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1, use_cuda_graph=False),
    )
    with build_simulation_context(sim_cfg=sim_cfg):
        stage = sim_utils.get_current_stage()
        _author_revolute_with_frictionloss(stage, frictionloss=0.11)

        NewtonManager.instantiate_builder_from_stage()
        model = NewtonManager._builder.finalize()

        friction = model.joint_friction.numpy()
        assert friction.max() == pytest.approx(0.11), (
            f"mjc:frictionloss was dropped by the USD import (joint_friction={friction})"
        )
