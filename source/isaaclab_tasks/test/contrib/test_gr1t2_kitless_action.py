# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-free tests for the GR1T2 kitless OVPhysX action boundary."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from isaaclab_newton.physics import NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab_tasks.contrib.pick_place import pickplace_gr1t2_env_cfg as gr1t2_env_cfg
from isaaclab_tasks.contrib.pick_place.gr1t2_kitless_action import (
    GR1T2KitlessPinkInverseKinematicsAction,
    _HandTargetSlewState,
    bound_ik_joint_targets,
    constrain_hand_joint_targets,
)
from isaaclab_tasks.contrib.pick_place.gr1t2_kitless_action_cfg import (
    GR1T2KitlessPinkInverseKinematicsActionCfg,
)
from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import (
    PickPlaceGR1T2EnvCfg,
    condition_kitless_gr1t2_stage,
    condition_kitless_ovphysx_robot_spawn_properties,
)
from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_waist_enabled_env_cfg import PickPlaceGR1T2WaistEnabledEnvCfg
from isaaclab_tasks.contrib.steering_wheel import SteeringWheelUsdFileCfg
from isaaclab_tasks.utils import parse_env_cfg


@pytest.mark.parametrize("config_type", [PickPlaceGR1T2EnvCfg, PickPlaceGR1T2WaistEnabledEnvCfg])
def test_newton_contract_is_backend_scoped_and_complete(config_type):
    default_cfg = config_type()
    task = (
        "IsaacContrib-PickPlace-GR1T2-WaistEnabled-Abs"
        if config_type is PickPlaceGR1T2WaistEnabledEnvCfg
        else "IsaacContrib-PickPlace-GR1T2-Abs"
    )
    newton_cfg = parse_env_cfg(task, device="cpu", num_envs=1, overrides=["physics=newton_mjwarp"])
    newton_cfg.stabilize_for_kitless_newton_mjwarp()

    expected = {
        "head": (80.0, 8.0, 0.03, 80.0),
        "trunk": (500.0, 50.0, 0.10, 250.0),
        "legs": (300.0, 30.0, 0.125, 250.0),
        "right-arm": (1200.0, 60.0, 0.30, 800.0),
        "left-arm": (1200.0, 60.0, 0.30, 800.0),
        "right-hand": (5.0, 0.5, 0.02, 20.0),
        "left-hand": (5.0, 0.5, 0.02, 20.0),
    }
    assert set(newton_cfg.scene.robot.actuators) == set(expected)
    for name, values in expected.items():
        actuator = newton_cfg.scene.robot.actuators[name]
        assert (actuator.stiffness, actuator.damping, actuator.armature, actuator.joint_effort_limit) == pytest.approx(
            values
        )
    assert newton_cfg.scene.robot.actuators["left-hand"].joint_names_expr == ["L_.*"]
    assert newton_cfg.scene.robot.actuators["right-hand"].joint_names_expr == ["R_.*"]
    assert newton_cfg.sim.physics.solver_cfg.nconmax == 4096
    assert newton_cfg.sim.physics.solver_cfg.njmax == 1536
    assert newton_cfg.isaac_teleop.require_initial_bimanual_hand_tracking is True
    assert newton_cfg.actions.upper_body_ik.kitless_target_conditioning_enabled is True
    assert default_cfg.actions.upper_body_ik.kitless_target_conditioning_enabled is False
    assert isinstance(newton_cfg.scene.robot.spawn, gr1t2_env_cfg.GR1T2KitlessUsdFileCfg)
    assert not isinstance(default_cfg.scene.robot.spawn, gr1t2_env_cfg.GR1T2KitlessUsdFileCfg)
    assert default_cfg.isaac_teleop.require_initial_bimanual_hand_tracking is False

    assert set(default_cfg.scene.robot.actuators) == {
        "trunk",
        "right-arm",
        "left-arm",
        "right-hand",
        "left-hand",
    }
    assert "legs" not in default_cfg.scene.robot.actuators


def test_newton_teleop_self_collision_matches_ovphysx_without_changing_asset_default():
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    default_cfg = PickPlaceGR1T2EnvCfg()
    for backend, fragment_type, attribute in (
        ("newton_mjwarp", NewtonArticulationCfg, "self_collision_enabled"),
        ("ovphysx", PhysxArticulationCfg, "enabled_self_collisions"),
    ):
        cfg = PickPlaceGR1T2EnvCfg()
        cfg.sim.physics = OvPhysxCfg() if backend == "ovphysx" else NewtonCfg()
        getattr(cfg, f"stabilize_for_kitless_{backend}")()
        fragment = next(p for p in cfg.scene.robot.spawn.articulation_props if isinstance(p, fragment_type))
        original = next(p for p in default_cfg.scene.robot.spawn.articulation_props if isinstance(p, fragment_type))
        assert getattr(fragment, attribute) is False
        assert getattr(original, attribute) is True


@pytest.mark.parametrize("backend", ["ovphysx", "newton_mjwarp"])
@pytest.mark.parametrize("copy_prepared", [False, True, "deepcopy"])
@pytest.mark.parametrize("replace_spawn", [False, True])
def test_hand_physical_limits_are_widened_only_for_ovphysx(monkeypatch, backend, copy_prepared, replace_spawn):
    stage = Usd.Stage.CreateInMemory()

    def spawn_robot(path, _cfg, **_kwargs):
        root = stage.DefinePrim(path, "Xform")
        joint = UsdPhysics.RevoluteJoint.Define(stage, path + "/R_thumb_distal_joint")
        joint.CreateLowerLimitAttr().Set(-15.0)
        joint.CreateUpperLimitAttr().Set(80.0)
        return root

    cfg = PickPlaceGR1T2EnvCfg()
    cfg.sim.physics = OvPhysxCfg() if backend == "ovphysx" else NewtonCfg()
    cfg.scene.robot.spawn.func = spawn_robot
    getattr(cfg, f"stabilize_for_kitless_{backend}")()
    if replace_spawn:
        cfg.scene.robot.spawn.func = spawn_robot
        getattr(cfg, f"stabilize_for_kitless_{backend}")()
    original_cfg = cfg
    if copy_prepared:
        cfg = deepcopy(cfg) if copy_prepared == "deepcopy" else cfg.copy()
        assert cfg.actions.upper_body_ik is not original_cfg.actions.upper_body_ik
        getattr(cfg, f"stabilize_for_kitless_{backend}")()
    from isaaclab.assets import Asset, AssetBaseCfg
    from isaaclab.sim import use_stage

    cfg.scene.robot.spawn.semantic_tags = [("class", "gr1t2")]
    # AssetBase delegates to this exact constructor, which copies the asset config before spawning.
    with use_stage(stage):
        asset = Asset(AssetBaseCfg(prim_path="/World/Robot", spawn=cfg.scene.robot.spawn))
    assert asset.cfg.spawn is not cfg.scene.robot.spawn
    assert asset.cfg.spawn.action_cfg is not cfg.actions.upper_body_ik
    asset.cfg.spawn.semantic_tags.append(("instance", "copy"))
    assert cfg.scene.robot.spawn.semantic_tags == [("class", "gr1t2")]
    assert cfg.scene.robot.spawn.copy().action_cfg is not cfg.actions.upper_body_ik
    from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction

    def initialize_solver(self, action_cfg, env):
        self.cfg = action_cfg
        self._env = env
        self._asset = asset
        self._hand_joint_names = ["R_thumb_distal_joint"]
        self._raw_actions = torch.zeros((1, 8))

    monkeypatch.setattr(PinkInverseKinematicsAction, "__init__", initialize_solver)
    action = GR1T2KitlessPinkInverseKinematicsAction(
        cfg.actions.upper_body_ik, SimpleNamespace(device="cpu", num_envs=1)
    )
    assert action._hand_lower.shape == action._hand_upper.shape == (1,)
    assert cfg.actions.upper_body_ik.hand_joint_limits is not asset.cfg.spawn.action_cfg.hand_joint_limits
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/World/Robot/R_thumb_distal_joint"))
    expected = (-720.0, 720.0) if backend == "ovphysx" else (-15.0, 80.0)
    assert (joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()) == pytest.approx(expected)
    assert cfg.actions.upper_body_ik.hand_joint_limits["R_thumb_distal_joint"] == pytest.approx(
        (torch.deg2rad(torch.tensor(-15.0)).item(), torch.deg2rad(torch.tensor(80.0)).item())
    )
    if copy_prepared:
        assert not original_cfg.actions.upper_body_ik.hand_joint_limits


@pytest.mark.parametrize("config_type", [PickPlaceGR1T2EnvCfg, PickPlaceGR1T2WaistEnabledEnvCfg])
def test_newton_wheel_wrapper_hulls_only_non_graspable_details(monkeypatch, config_type):
    stage = Usd.Stage.CreateInMemory()

    def spawn_wheel(prim_path, _cfg):
        root = stage.DefinePrim(prim_path, "Xform")
        for name in (
            "sm_steeringwheel_a01_wheel_rim_01",
            "sm_steeringwheel_a01_wheel_spokes_01",
            "sm_steeringwheel_a01_hub_detail_01",
            "sm_steeringwheel_a01_quick_release_detail_01",
        ):
            mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/{name}").GetPrim()
            UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)
        return root

    cfg = config_type()
    controlled_joints = list(cfg.actions.upper_body_ik.pink_controlled_joint_names)
    if config_type is PickPlaceGR1T2WaistEnabledEnvCfg:
        assert {"waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"} <= set(controlled_joints)
    cfg.sim.physics = NewtonCfg()
    spawn = cfg.scene.object.spawn
    spawn.func = spawn_wheel
    monkeypatch.setattr(gr1t2_env_cfg.sim_utils, "find_matching_prim_paths", lambda _path: ["/World/Wheel"])

    cfg.stabilize_for_kitless_newton_mjwarp()
    spawn = cfg.scene.object.spawn
    wrapped_spawn = spawn.func
    wrapped_spawn("/World/Wheel", spawn)

    for name in ("sm_steeringwheel_a01_wheel_rim_01", "sm_steeringwheel_a01_wheel_spokes_01"):
        assert stage.GetPrimAtPath(f"/World/Wheel/{name}").GetAttribute("physics:approximation").Get() == (
            UsdPhysics.Tokens.convexDecomposition
        )
    for name in ("sm_steeringwheel_a01_hub_detail_01", "sm_steeringwheel_a01_quick_release_detail_01"):
        assert stage.GetPrimAtPath(f"/World/Wheel/{name}").GetAttribute("physics:approximation").Get() == (
            UsdPhysics.Tokens.convexHull
        )

    gr1t2_env_cfg.condition_newton_steering_wheel_detail_collisions(cfg.scene.object)
    assert spawn.func is wrapped_spawn
    assert cfg.actions.upper_body_ik.pink_controlled_joint_names == controlled_joints
    assert not isinstance(config_type().scene.object.spawn, SteeringWheelUsdFileCfg)


@pytest.mark.parametrize("config_type", [PickPlaceGR1T2EnvCfg, PickPlaceGR1T2WaistEnabledEnvCfg])
@pytest.mark.parametrize("backend", ["newton_mjwarp", "ovphysx"])
def test_kitless_kinematics_model_loads_without_the_kit_exporter(config_type, backend):
    from isaaclab.controllers.pink_ik.pink_kinematics_configuration import PinkKinematicsConfiguration

    cfg = config_type()
    assert cfg.actions.upper_body_ik.controller.urdf_path is None
    controlled_joints = list(cfg.actions.upper_body_ik.pink_controlled_joint_names)
    spawn_usd = cfg.scene.robot.spawn.usd_path
    cfg.sim.physics = OvPhysxCfg() if backend == "ovphysx" else NewtonCfg()
    getattr(cfg, f"stabilize_for_kitless_{backend}")()
    action = cfg.actions.upper_body_ik
    assert action.pink_controlled_joint_names == controlled_joints
    assert cfg.scene.robot.spawn.usd_path == spawn_usd
    assert action.controller.urdf_path is not None, "Kitless IK must not invoke Isaac Sim's USD-to-URDF exporter"
    kinematics = PinkKinematicsConfiguration(
        controlled_joint_names=action.pink_controlled_joint_names,
        urdf_path=action.controller.urdf_path,
    )
    assert set(kinematics.controlled_joint_names_pinocchio_order) == set(action.pink_controlled_joint_names)
    for frame in ("left_hand_pitch_link", "right_hand_pitch_link"):
        assert np.isfinite(kinematics.get_transform_frame_to_world(frame).homogeneous).all()
        jacobian = kinematics.get_frame_jacobian(frame)
        assert jacobian.shape == (6, len(action.pink_controlled_joint_names))
        assert np.isfinite(jacobian).all()
        if config_type is PickPlaceGR1T2WaistEnabledEnvCfg:
            waist_yaw = kinematics.controlled_joint_names_pinocchio_order.index("waist_yaw_joint")
            # A real movable waist must rotate either hand about its vertical axis.
            world_axis = kinematics.get_transform_frame_to_world(frame).rotation @ jacobian[3:, waist_yaw]
            np.testing.assert_allclose(world_axis, [0.0, 0.0, 1.0], atol=1e-6)


def test_kitless_conditioning_is_opt_in_and_metadata_is_not_shared():
    first = GR1T2KitlessPinkInverseKinematicsActionCfg()
    second = GR1T2KitlessPinkInverseKinematicsActionCfg()

    first.hand_joint_limits["joint"] = (-1.0, 1.0)

    assert not first.kitless_target_conditioning_enabled
    assert second.hand_joint_limits == {}


def test_ik_guard_is_per_environment_and_holds_invalid_rows():
    current = torch.tensor([[0.0, 0.1], [0.25, -0.25], [-0.1, 0.1]])
    targets = torch.tensor([[1.0, -1.0], [float("nan"), 0.0], [-0.15, 0.15]])

    actual = bound_ik_joint_targets(targets, current, max_delta_rad=0.2, max_magnitude_rad=0.3)

    expected = torch.tensor([[0.2, -0.1], [0.25, -0.25], [-0.15, 0.15]])
    torch.testing.assert_close(actual, expected)


def test_hand_constraints_apply_mimic_equation_and_bounds_without_mutating_input():
    targets = torch.tensor([[0.4, 9.0, -0.9], [-0.3, 9.0, 0.8]])
    original = targets.clone()
    # q_mimic = 2 * q_reference - 0.1; the first row then reaches its upper bound.
    mimic_pairs = ((1, 0, 2.0, -0.1),)
    lower = torch.tensor([-0.5, -0.6, -0.7])
    upper = torch.tensor([0.5, 0.6, 0.7])

    actual = constrain_hand_joint_targets(targets, mimic_pairs, lower, upper)

    torch.testing.assert_close(actual, torch.tensor([[0.4, 0.6, -0.7], [-0.3, -0.6, 0.7]]))
    torch.testing.assert_close(targets, original)


def test_hand_target_slew_clamps_discontinuity_against_previous_command():
    state = _HandTargetSlewState(torch.zeros((1, 2)), max_step_rad=0.079)

    torch.testing.assert_close(state.condition(torch.tensor([[0.2, -0.2]])), torch.tensor([[0.2, -0.2]]))
    torch.testing.assert_close(state.condition(torch.tensor([[0.6, -0.5]])), torch.tensor([[0.279, -0.279]]))


def test_hand_target_slew_preserves_smooth_input_without_command_chasing():
    state = _HandTargetSlewState(torch.zeros((1, 2)), max_step_rad=0.08)
    state.condition(torch.tensor([[0.2, -0.2]]))

    # The conditioner has no measured-state input: a smooth command passes exactly.
    commanded = torch.tensor([[0.25, -0.24]])
    torch.testing.assert_close(state.condition(commanded), commanded)


def test_hand_target_slew_reset_clears_previous_command_history():
    state = _HandTargetSlewState(torch.zeros((2, 1)), max_step_rad=0.08)
    state.condition(torch.tensor([[0.2], [-0.2]]))
    state.condition(torch.tensor([[0.6], [-0.6]]))

    state.reset([0])

    torch.testing.assert_close(state.condition(torch.tensor([[0.7], [-0.7]])), torch.tensor([[0.7], [-0.36]]))


def test_robot_conditioning_authors_collision_offsets_and_deinstances_scopes():
    stage = Usd.Stage.CreateInMemory()

    def spawn_robot(prim_path, _cfg):
        robot = stage.DefinePrim(prim_path, "Xform")
        collisions = stage.DefinePrim(f"{prim_path}/collisions", "Scope")
        collisions.SetInstanceable(True)
        collision = UsdGeom.Cube.Define(stage, f"{prim_path}/collision_shape").GetPrim()
        UsdPhysics.CollisionAPI.Apply(collision)
        return robot

    from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

    spawn = SimpleNamespace(
        func=spawn_robot,
        rigid_props=[PhysxRigidBodyCfg(max_depenetration_velocity=1.0)],
    )
    robot_cfg = SimpleNamespace(spawn=spawn)
    action_cfg = GR1T2KitlessPinkInverseKinematicsActionCfg()

    condition_kitless_ovphysx_robot_spawn_properties(robot_cfg)
    condition_kitless_gr1t2_stage(robot_cfg, action_cfg)
    spawn = robot_cfg.spawn
    spawn.func("/World/Robot", spawn)

    collisions = stage.GetPrimAtPath("/World/Robot/collisions")
    collision = stage.GetPrimAtPath("/World/Robot/collision_shape")
    schemas = collision.GetMetadata("apiSchemas")
    applied = [str(schema) for schema in schemas.GetAddedOrExplicitItems()]
    assert not collisions.IsInstanceable()
    assert "PhysxCollisionAPI" in applied
    assert collision.GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(0.001)
    assert collision.GetAttribute("physxCollision:restOffset").Get() == pytest.approx(0.0)
    assert spawn.rigid_props[0].max_depenetration_velocity == pytest.approx(0.1)


def test_wheel_mesh_collision_geometry_is_not_replaced():
    stage = Usd.Stage.CreateInMemory()
    original = {}

    def spawn_wheel(path, cfg):
        root = stage.DefinePrim(path, "Xform")
        body = stage.DefinePrim(path + "/Body", "Xform")
        UsdPhysics.RigidBodyAPI.Apply(body)
        for name in ("Rim", "Spokes", "Hub", "QuickRelease"):
            mesh = UsdGeom.Mesh.Define(stage, path + "/Body/" + name)
            mesh.CreatePointsAttr().Set([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
            mesh.CreateFaceVertexCountsAttr().Set([3])
            mesh.CreateFaceVertexIndicesAttr().Set([0, 1, 2])
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("convexDecomposition")
        original["layer"] = stage.GetRootLayer().ExportToString()
        return root

    cfg = PickPlaceGR1T2EnvCfg()
    cfg.scene.object.spawn.func = spawn_wheel
    cfg.stabilize_for_kitless_ovphysx()
    root = cfg.scene.object.spawn.func("/World/Wheel", cfg.scene.object.spawn)
    assert stage.GetRootLayer().ExportToString() == original["layer"]
    colliders = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == 4
    assert {p.GetName() for p in colliders} == {"Rim", "Spokes", "Hub", "QuickRelease"}
    assert all(p.IsA(UsdGeom.Mesh) for p in colliders)
    assert all(UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() for p in colliders)
    assert all(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() == "convexDecomposition" for p in colliders)
    assert not stage.GetPrimAtPath("/World/Wheel/Body/KitlessPrimitiveCollision")
