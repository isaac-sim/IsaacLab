# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless real-solver equivalence test for Newton-native actuators."""

import pytest
import torch
from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager

from pxr import Gf, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg, build_simulation_context

pytestmark = pytest.mark.integration


def _author_two_link_articulations() -> Articulation:
    """Author two local one-DOF articulations for actuator equivalence."""
    link_cfg = sim_utils.CuboidCfg(
        size=(0.4, 0.1, 0.1),
        rigid_props=sim_utils.RigidBodyBaseCfg(disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionBaseCfg(),
    )
    stage = sim_utils.get_current_stage()
    for env_index in range(2):
        env_path = f"/World/Env_{env_index}"
        robot_path = f"{env_path}/Robot"
        root_path = f"{robot_path}/Root"
        child_path = f"{robot_path}/Child"
        sim_utils.create_prim(env_path, "Xform", translation=(2.0 * env_index, 0.0, 0.0))
        sim_utils.create_prim(robot_path, "Xform")
        link_cfg.func(root_path, link_cfg, translation=(0.0, 0.0, 1.0))
        link_cfg.func(child_path, link_cfg, translation=(0.5, 0.0, 1.0))
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(root_path))
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{robot_path}/Joint")
        joint.CreateBody0Rel().SetTargets([root_path])
        joint.CreateBody1Rel().SetTargets([child_path])
        joint.CreateAxisAttr().Set("Z")
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.25, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.25, 0.0, 0.0))
        joint.CreateLowerLimitAttr().Set(-90.0)
        joint.CreateUpperLimitAttr().Set(90.0)

    articulation_cfg = ArticulationCfg(
        prim_path="/World/Env_[^/]*/Robot",
        articulation_root_prim_path="/Root",
        actuators={
            "joint": IdealPDActuatorCfg(
                joint_names_expr=["Joint"],
                stiffness=20.0,
                damping=2.0,
                actuator_effort_limit=50.0,
            )
        },
    )
    articulation_cfg._post_spawn(stage)
    return Articulation(articulation_cfg)


def _run_actuator_path(*, use_newton_actuators: bool) -> dict[str, object]:
    """Run the local articulation through one actuator execution path."""
    sim_cfg = SimulationCfg(
        device="cpu",
        dt=1.0 / 120.0,
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), use_cuda_graph=False),
        use_newton_actuators=use_newton_actuators,
    )
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        articulation = _author_two_link_articulations()
        sim.reset()
        adapter = SimulationManager._adapter
        execution_state = {
            "has_newton_actuators": articulation._has_newton_actuators,
            "native_path_active": articulation._actuator_control.native_actuator_path_active,
            "manager_path_active": SimulationManager._use_newton_actuators_active,
            "adapter_bound": adapter is not None and articulation.newton_actuator_adapter is adapter,
            "model_actuator_count": len(SimulationManager.get_model().actuators),
            "adapter_actuator_count": 0 if adapter is None else len(adapter.actuators),
            "adapter_state_count": 0 if adapter is None else len(adapter._states_a),
        }
        initial_position = articulation.data.joint_pos.torch.clone()
        target_position = initial_position + torch.tensor([[0.10], [-0.15]])
        articulation.actuators.target_command.set_position_index(value=target_position)
        articulation.actuators.target_command.set_velocity_index(value=torch.zeros_like(target_position))

        joint_position = []
        joint_velocity = []
        computed_effort = []
        applied_effort = []
        for _ in range(5):
            articulation.write_data_to_sim()
            sim.step()
            articulation.update(sim.cfg.dt)
            joint_position.append(articulation.data.joint_pos.torch.clone())
            joint_velocity.append(articulation.data.joint_vel.torch.clone())
            computed_effort.append(articulation.actuators.computed_effort.torch.clone())
            applied_effort.append(articulation.actuators.applied_effort.torch.clone())

        return {
            "execution_state": execution_state,
            "initial_position": initial_position,
            "target_position": target_position,
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "computed_effort": computed_effort,
            "applied_effort": applied_effort,
        }


def test_newton_actuator_real_equivalence() -> None:
    """Match one real IdealPD rollout between Isaac Lab and Newton-native execution."""
    lab_result = _run_actuator_path(use_newton_actuators=False)
    newton_result = _run_actuator_path(use_newton_actuators=True)

    assert lab_result["execution_state"] == {
        "has_newton_actuators": False,
        "native_path_active": False,
        "manager_path_active": False,
        "adapter_bound": False,
        "model_actuator_count": 0,
        "adapter_actuator_count": 0,
        "adapter_state_count": 0,
    }
    assert newton_result["execution_state"] == {
        "has_newton_actuators": True,
        "native_path_active": True,
        "manager_path_active": True,
        "adapter_bound": True,
        "model_actuator_count": 1,
        "adapter_actuator_count": 1,
        "adapter_state_count": 1,
    }
    torch.testing.assert_close(newton_result["target_position"], lab_result["target_position"])
    for key in ("joint_position", "joint_velocity", "computed_effort", "applied_effort"):
        for newton_value, lab_value in zip(newton_result[key], lab_result[key], strict=True):
            torch.testing.assert_close(newton_value, lab_value, atol=1e-5, rtol=1e-5)

    for result in (lab_result, newton_result):
        assert max(value.abs().max() for value in result["computed_effort"]) > 1e-3
        assert max(value.abs().max() for value in result["applied_effort"]) > 1e-3
        initial_error = (result["target_position"] - result["initial_position"]).abs()
        final_error = (result["target_position"] - result["joint_position"][-1]).abs()
        joint_motion = (result["joint_position"][-1] - result["initial_position"]).abs()
        assert torch.all(joint_motion > 1e-5)
        assert torch.all(final_error < initial_error)
