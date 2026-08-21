# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local articulation authoring helpers for kitless Newton integration tests."""

from collections.abc import Sequence

from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg, build_simulation_context


def build_newton_context(*, gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)):
    """Create a fresh CPU Newton simulation context."""
    return build_simulation_context(
        sim_cfg=SimulationCfg(
            device="cpu",
            dt=1.0 / 120.0,
            gravity=gravity,
            physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), use_cuda_graph=False),
        )
    )


def author_fixed_spatial_chain(*, actuators: dict | None = None) -> Articulation:
    """Author the smallest fixed chain with a full-rank spatial Jacobian."""
    link_cfg = sim_utils.CuboidCfg(
        size=(0.08, 0.08, 0.08),
        rigid_props=sim_utils.RigidBodyBaseCfg(disable_gravity=False),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
        collision_props=sim_utils.CollisionBaseCfg(collision_enabled=False),
    )
    stage = sim_utils.get_current_stage()
    robot_path = "/World/Robot"
    root_path = f"{robot_path}/Root"
    sim_utils.create_prim(robot_path, "Xform")
    link_cfg.func(root_path, link_cfg, translation=(0.0, 0.0, 1.0))
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(root_path))
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, f"{robot_path}/RootJoint")
    fixed_joint.CreateBody1Rel().SetTargets([root_path])

    axes: Sequence[str] = ("X", "Y", "Z", "X", "Y", "Z")
    parent_path = root_path
    for joint_index, axis in enumerate(axes):
        child_path = f"{robot_path}/Link_{joint_index}"
        link_cfg.func(child_path, link_cfg, translation=(0.0, 0.0, 1.0))
        joint_path = f"{robot_path}/Joint_{joint_index}"
        if joint_index < 3:
            joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
            joint.CreateLowerLimitAttr().Set(-0.2)
            joint.CreateUpperLimitAttr().Set(0.2)
        else:
            joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
            joint.CreateLowerLimitAttr().Set(-45.0)
            joint.CreateUpperLimitAttr().Set(45.0)
        joint.CreateBody0Rel().SetTargets([parent_path])
        joint.CreateBody1Rel().SetTargets([child_path])
        joint.CreateAxisAttr().Set(axis)
        parent_path = child_path

    return Articulation(
        ArticulationCfg(
            prim_path=robot_path,
            articulation_root_prim_path="/Root",
            actuators=(
                actuators
                if actuators is not None
                else {
                    "joints": ImplicitActuatorCfg(
                        joint_names_expr=["Joint_.*"],
                        stiffness=80.0,
                        damping=8.0,
                    )
                }
            ),
        )
    )
