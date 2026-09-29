# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real PhysX articulation coverage.

Most checks run against one module-scoped composite scene built from the local branching fixture. Each
articulation island holds two environments so that partial writes can target environment 1 and prove that
environment 0 is preserved in the real PhysX state. Tests that need their own simulation context (failure
modes and asset-specific seams) are defined first: pytest runs them before the composite scene is created,
and a new simulation context would replace the composite stage.
"""

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from isaaclab.test.utils import DeviceScope, launch_test_simulation, test_devices
from isaaclab.test.utils.articulation_ordering import (
    BRANCHING_MJWARP_BODY_NAMES,
    BRANCHING_MJWARP_JOINT_NAMES,
    BRANCHING_PHYSX_BODY_NAMES,
    BRANCHING_PHYSX_JOINT_NAMES,
)
from isaaclab.utils import clone, replace

launch_test_simulation()

import math
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import warp as wp
from isaaclab_physx.assets import Articulation

from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, get_articulation_name_ordering
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sim import SimulationContext, build_simulation_context
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_quat,
    quat_apply,
    quat_inv,
    subtract_frame_transforms,
)

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG  # isort:skip
from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_PHYSX_CFG  # isort:skip

_FIXTURE = Path(__file__).parent / "data" / "articulation_ordering_branching.usda"
_NUM_ENVS = 2
_ENV_SPACING = 3.0

# The branching fixture authors no geometry. Place the links along the x axis so that joints, Jacobians,
# and center-of-mass offsets act on non-degenerate lever arms. Each joint is anchored at its child link origin,
# offset from the parent link origin by the listed vector.
_JOINT_ANCHORS = {
    "left_shoulder": (0.2, 0.0, 0.0),
    "left_elbow": (0.5, 0.0, 0.0),
    "right_shoulder": (-0.2, 0.0, 0.0),
    "right_elbow": (-0.5, 0.0, 0.0),
}
_LINK_POSITIONS = {
    "left_upper": (0.2, 0.0, 0.0),
    "left_tip": (0.7, 0.0, 0.0),
    "right_upper": (-0.2, 0.0, 0.0),
    "right_tip": (-0.7, 0.0, 0.0),
}
# Drive and limit values authored in USD, keyed by joint name. USD angular drives use degree units, so the
# authored values are converted such that PhysX reports the SI values listed here.
_USD_STIFFNESS = {"left_shoulder": 4.0, "left_elbow": 5.0, "right_shoulder": 6.0, "right_elbow": 7.0}
_USD_DAMPING = {"left_shoulder": 0.4, "left_elbow": 0.5, "right_shoulder": 0.6, "right_elbow": 0.7}
_USD_MAX_FORCE = {"left_shoulder": 40.0, "left_elbow": 41.0, "right_shoulder": 42.0, "right_elbow": 43.0}
_USD_MAX_VELOCITY = {"left_shoulder": 2.0, "left_elbow": 3.0, "right_shoulder": 4.0, "right_elbow": 5.0}
# Configured gains and limits shared by both floating islands, so that their responses are comparable.
_FLOATING_ACTUATORS = {
    "joints": ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        stiffness={".*_shoulder": 6.0, ".*_elbow": 4.0},
        damping={".*_shoulder": 0.6, ".*_elbow": 0.4},
        joint_velocity_limit=7.0,
        joint_effort_limit=30.0,
    )
}


def _branching_cfg(prim_path: str = "/World/Robot", **kwargs) -> ArticulationCfg:
    """Create a configuration of the local branching fixture with implicit drives on every joint."""
    kwargs.setdefault("actuators", {"joints": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=5.0, damping=0.5)})
    return ArticulationCfg(prim_path=prim_path, spawn=sim_utils.UsdFileCfg(usd_path=str(_FIXTURE)), **kwargs)


def _author_branching_robot(
    robot_path: str,
    *,
    fixed_base: bool = False,
    root_on_base: bool = False,
    reversed_left_elbow: bool = False,
    spatial_tendon: bool = False,
    joint_limits_deg: tuple[float, float] = (-170.0, 170.0),
) -> None:
    """Author geometry, drives, and optional structure on one spawned branching robot.

    Args:
        robot_path: Prim path of the spawned robot.
        fixed_base: Whether to fix the base link to the world with a fixed joint.
        root_on_base: Whether to move the articulation root API from the robot prim onto the base link.
        reversed_left_elbow: Whether to swap the bodies of the left elbow joint.
        spatial_tendon: Whether to attach a spatial tendon from the base to the left tip.
        joint_limits_deg: Lower and upper joint limits [deg] applied to every joint. PhysX ignores limit writes
            on joints authored without limits.
    """
    stage = sim_utils.get_current_stage()
    for body_name, position in _LINK_POSITIONS.items():
        UsdGeom.Xformable(stage.GetPrimAtPath(f"{robot_path}/{body_name}")).AddTranslateOp().Set(Gf.Vec3d(*position))
    for joint_name, anchor in _JOINT_ANCHORS.items():
        joint = UsdPhysics.RevoluteJoint.Get(stage, f"{robot_path}/{joint_name}")
        if reversed_left_elbow and joint_name == "left_elbow":
            body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            joint.GetBody0Rel().SetTargets(body1)
            joint.GetBody1Rel().SetTargets(body0)
            joint.CreateLocalPos0Attr(Gf.Vec3f(0.0))
            joint.CreateLocalPos1Attr(Gf.Vec3f(*anchor))
        else:
            joint.CreateLocalPos0Attr(Gf.Vec3f(*anchor))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        joint.CreateLowerLimitAttr(joint_limits_deg[0])
        joint.CreateUpperLimitAttr(joint_limits_deg[1])
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateStiffnessAttr(math.radians(_USD_STIFFNESS[joint_name]))
        drive.CreateDampingAttr(math.radians(_USD_DAMPING[joint_name]))
        drive.CreateMaxForceAttr(_USD_MAX_FORCE[joint_name])
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
        physx_joint.CreateMaxJointVelocityAttr(math.degrees(_USD_MAX_VELOCITY[joint_name]))
    # The base carries one collision shape so that its inertia follows from real geometry.
    collision = UsdGeom.Cube.Define(stage, f"{robot_path}/base/collision")
    collision.CreateSizeAttr(0.1)
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    if fixed_base:
        fixed_joint = UsdPhysics.FixedJoint.Define(stage, f"{robot_path}/fixed_root")
        fixed_joint.GetBody1Rel().SetTargets([f"{robot_path}/base"])
    if root_on_base:
        stage.GetPrimAtPath(robot_path).RemoveAPI(UsdPhysics.ArticulationRootAPI)
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(f"{robot_path}/base"))
    if spatial_tendon:
        root_prim = stage.GetPrimAtPath(f"{robot_path}/base")
        root_attachment = PhysxSchema.PhysxTendonAttachmentAPI(root_prim, "root")
        root_attachment.CreateLocalPosAttr(Gf.Vec3f(0.0))
        root = PhysxSchema.PhysxTendonAttachmentRootAPI.Apply(root_prim, "root")
        root.CreateStiffnessAttr(5.0)
        root.CreateDampingAttr(0.5)
        root.CreateLimitStiffnessAttr(1.0)
        root.CreateOffsetAttr(0.0)
        leaf_prim = stage.GetPrimAtPath(f"{robot_path}/left_tip")
        leaf_attachment = PhysxSchema.PhysxTendonAttachmentAPI(leaf_prim, "leaf")
        leaf_attachment.CreateLocalPosAttr(Gf.Vec3f(0.0))
        leaf_attachment.CreateParentAttachmentAttr("root")
        leaf_attachment.CreateParentLinkRel().SetTargets([root_prim.GetPath()])
        leaf = PhysxSchema.PhysxTendonAttachmentLeafAPI.Apply(leaf_prim, "leaf")
        leaf.CreateRestLengthAttr(0.5)
        leaf.CreateLowerLimitAttr(0.0)
        leaf.CreateUpperLimitAttr(2.0)


def _in_user_order(values: dict[str, float], names: list[str], device: str) -> torch.Tensor:
    """Return per-name values as a ``(num_envs, len(names))`` tensor in the given name order."""
    return torch.tensor([values[name] for name in names], device=device).repeat(_NUM_ENVS, 1)


def _yaw_quat(angle: float) -> tuple[float, float, float, float]:
    """Return a unit quaternion ``(x, y, z, w)`` for a rotation of ``angle`` [rad] about the world z axis."""
    return (0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle))


@pytest.fixture
def sim(request) -> Iterator[SimulationContext]:
    """Create a function-scoped simulation context for tests that own their scene."""
    device = request.getfixturevalue("device")
    gravity_enabled = request.getfixturevalue("gravity_enabled") if "gravity_enabled" in request.fixturenames else True
    with build_simulation_context(device=device, auto_add_lighting=True, gravity_enabled=gravity_enabled) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


##
# Tests that own their simulation context. Keep them above the composite scene.
##


def generate_articulation(
    articulation_cfg: ArticulationCfg, num_articulations: int, device: str
) -> tuple[Articulation, torch.Tensor]:
    """Spawn ``num_articulations`` copies of an articulation 2.5 m apart along x.

    Args:
        articulation_cfg: Articulation configuration.
        num_articulations: Number of articulations to generate.
        device: Device to use for the tensors.

    Returns:
        The articulation and environment translations.
    """
    translations = torch.zeros(num_articulations, 3, device=device)
    translations[:, 0] = torch.arange(num_articulations) * 2.5
    for i in range(num_articulations):
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=translations[i][:3])
    articulation = Articulation(replace(articulation_cfg, prim_path="/World/Env_[^/]*/Robot"))
    return articulation, translations


@pytest.mark.parametrize(
    ("init_state", "message"),
    [
        ({"joint_pos": {"left_shoulder": 1.0}}, "default positions out of the limits"),
        ({"joint_vel": {"left_elbow": 10.0}}, "default velocities out of the limits"),
    ],
)
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_out_of_range_default_joint_state_raises(sim, device, init_state, message) -> None:
    """Initialization rejects configured default joint states outside the solver limits."""
    articulation = Articulation(_branching_cfg(init_state=ArticulationCfg.InitialStateCfg(**init_state)))
    # Limits of +-30 deg and the USD velocity limits keep the configured defaults out of range.
    _author_branching_robot("/World/Robot", fixed_base=True, joint_limits_deg=(-30.0, 30.0))

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    with pytest.raises(ValueError, match=message):
        sim.reset()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_setting_invalid_articulation_root_prim_path(sim, device) -> None:
    """A configured articulation root path that does not exist fails initialization."""
    articulation = Articulation(_branching_cfg(articulation_root_prim_path="/non_existing_prim_path"))
    _author_branching_robot("/World/Robot")

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    with pytest.raises(RuntimeError, match="Failed to create articulation at"):
        sim.reset()


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
def test_fixed_tendon_position_target_writes_offset(sim, num_articulations, device) -> None:
    """A tendon length target lands in the simulation as ``rest_length - target`` on the selected cells only.

    The index form commands every tendon of environment 0; the mask form commands tendon 0 of
    environment 1. Every other cell must keep its initial offset. The Shadow Hand is the shipped asset whose
    native PhysX fixed tendons these writers command.
    """
    articulation_cfg = SHADOW_HAND_PHYSX_CFG
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)

    sim.reset()
    assert articulation.is_initialized
    assert articulation.is_fixed_base
    assert articulation.data.joint_pos.torch.shape == (num_articulations, 24)
    assert articulation.data.body_mass.torch.shape == (num_articulations, articulation.num_bodies)
    assert articulation.data.body_inertia.torch.shape == (num_articulations, articulation.num_bodies, 9)
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert actuator.is_implicit_model == is_implicit_model_cfg
    num_tendons = articulation.num_fixed_tendons
    assert num_tendons > 0
    rest_length = articulation.data.fixed_tendon_rest_length.torch.clone()
    initial_offset = articulation.data.fixed_tendon_offset.torch.clone()

    index_target = torch.full((1, num_tendons), 0.3, dtype=torch.float32, device=device)
    articulation.set_fixed_tendon_position_target_index(target=index_target, env_ids=[0])
    # Distinct per-cell values: a uniform target cannot catch the mask form reading the wrong
    # cell, because every wrong read returns the same number.
    mask_target = (
        0.7
        + 0.1 * torch.arange(num_articulations, dtype=torch.float32, device=device).unsqueeze(1)
        + 0.01 * torch.arange(num_tendons, dtype=torch.float32, device=device).unsqueeze(0)
    )
    env_mask = wp.array([False, True], dtype=wp.bool, device=device)
    tendon_mask = wp.array([i == 0 for i in range(num_tendons)], dtype=wp.bool, device=device)
    articulation.set_fixed_tendon_position_target_mask(
        target=mask_target, fixed_tendon_mask=tendon_mask, env_mask=env_mask
    )

    articulation.write_data_to_sim()
    sim.step()
    articulation.update(sim.cfg.dt)

    expected = initial_offset.clone()
    expected[0] = rest_length[0] - 0.3
    expected[1, 0] = rest_length[1, 0] - mask_target[1, 0]
    torch.testing.assert_close(articulation.data.fixed_tendon_offset.torch, expected)


@pytest.mark.parametrize("num_articulations", [1])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.isaacsim_ci
def test_get_gravity_compensation_forces_static_equilibrium(sim, num_articulations, device) -> None:
    """PhysX accuracy: ``τ_gc`` must hold the manipulator in static equilibrium.

    The contract is the EOM identity ``M(q) q̈ + C(q,q̇) q̇ + g(q) = τ_input``.
    Setting ``τ_input = g(q)`` at ``q̇ = 0`` gives ``q̈ = 0`` — the arm should
    not move. This pins
    :attr:`~isaaclab.assets.BaseArticulationData.gravity_compensation_forces`
    in isolation: sign errors, frame errors, and DoF-ordering errors all
    surface as joint drift, while a controller-level test would have those
    bugs averaged out by PD damping.

    Newton-side variant of the same name lives in
    ``isaaclab_newton/test/assets/test_articulation.py`` (backend parity).
    """
    # Replace default Franka actuators with a passthrough implicit actuator
    # (stiffness = 0, damping = 0). With both gains zero the effort target
    # we set IS the joint torque applied — no PD spring-damper masks the
    # gravity-comp signal. Default Franka cfg has stiffness=80 / damping=4
    # which would absorb gravity through PD bias and hide accessor bugs.
    cfg = replace(
        FRANKA_PANDA_CFG, actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)}
    )
    # FRANKA_PANDA_CFG has rigid_props.disable_gravity=False already, but be
    # defensive — gravity must be ON for τ_gc to have anything to cancel.
    cfg = replace(cfg, spawn=replace(cfg.spawn, rigid_props=replace(cfg.spawn.rigid_props, disable_gravity=False)))

    articulation, _ = generate_articulation(cfg, num_articulations, device=device)
    sim.reset()
    assert articulation.is_initialized

    # Force a clean static state: default joint positions, zero velocities.
    # ``sim.reset`` may leave residual ``q_dot`` from solver settling under
    # gravity, so we pin it explicitly here.
    default_q = articulation.data.default_joint_pos.torch.clone()
    default_qd = torch.zeros_like(default_q)
    articulation.write_joint_state_to_sim(default_q, default_qd)
    articulation.update(sim.cfg.dt)

    # Default joint pose from FRANKA_PANDA_CFG bends the elbow
    # (joint2=-0.569, joint4=-2.81, joint6=3.04) so several links carry a
    # gravity load — τ_gc is non-trivial in this configuration. A natural-
    # hang pose (all zeros) would produce near-zero τ_gc and make this
    # test uninformative.
    init_q = articulation.data.joint_pos.torch.clone()

    # Step 100 times applying only τ_gc as joint efforts.
    for _ in range(100):
        # ``gravity_compensation_forces`` shape is ``(N, num_joints + num_base_dofs)``
        # — leading ``num_base_dofs`` floating-base entries (0 on fixed-base) followed
        # by the actuated-joint entries. Slice past the floating-base entries so the
        # remaining tensor aligns with ``set_joint_effort_target`` (actuated only).
        tau_gc = articulation.data.gravity_compensation_forces.torch[:, articulation.num_base_dofs :]
        articulation.set_joint_effort_target(tau_gc)
        articulation.write_data_to_sim()
        sim.step()
        articulation.update(sim.cfg.dt)

    final_q = articulation.data.joint_pos.torch
    drift = (final_q - init_q).abs().max()
    # Tight bound: 5e-3 rad ≈ 0.3°. Numerical integration over 100 steps will
    # accumulate some floor (sub-millirad on Franka), but a sign or frame bug
    # in τ_gc produces drift of at least a degree per step on bent-elbow
    # poses. This bound separates "correct" from "broken" cleanly.
    assert drift < 5e-3, (
        f"max joint drift {drift:.5f} rad after 100 gravity-comp-only steps —"
        " τ_gc did not hold static equilibrium. Check sign, DoF ordering, and"
        " whether gravity_compensation_forces returns g(q) (positive) or"
        " its negation."
    )


# ---------------------------------------------------------------------------
# Franka task-space tracking helpers for the IK test.
# Mirrors the helpers in ``isaaclab_newton/test/assets/test_articulation.py``.
# ---------------------------------------------------------------------------


def _setup_franka_at_home_pose(sim):
    """Build a Franka articulation at its configured home pose.

    See the Newton-side mirror for full docs. Standalone tests skip the
    env reset path that normally pushes ``default_joint_pos`` to sim,
    so we teleport explicitly to avoid the URDF-neutral
    near-singular pose where the Franka wrist axes nearly align.

    Args:
        sim: The simulation context to use.

    Returns:
        Tuple of ``(robot, ee_frame_idx, ee_jacobi_idx, arm_joint_ids)``.
    """
    cfg = replace(clone(FRANKA_PANDA_HIGH_PD_CFG), prim_path="/World/Env_[^/]*/Robot")
    sim_utils.create_prim("/World/Env_0", "Xform", translation=(0.0, 0.0, 0.0))
    robot = Articulation(cfg)
    sim.reset()
    assert robot.is_initialized

    ee_frame_idx = robot.find_bodies("panda_hand")[0][0]
    ee_jacobi_idx = ee_frame_idx - 1
    arm_joint_ids = robot.find_joints(["panda_joint.*"])[0]

    robot.write_joint_state_to_sim(
        position=robot.data.default_joint_pos.torch[:, :].clone(),
        velocity=robot.data.default_joint_vel.torch[:, :].clone(),
    )
    return robot, ee_frame_idx, ee_jacobi_idx, arm_joint_ids


def _compute_ee_pose_root(robot, ee_frame_idx):
    """Return ``(ee_pos_b, ee_quat_b, root_pose_w)`` in the root frame."""
    ee_pose_w = robot.data.body_pose_w.torch[:, ee_frame_idx]
    root_pose_w = robot.data.root_pose_w.torch
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    return ee_pos_b, ee_quat_b, root_pose_w


def _compute_jacobian_root_frame(robot, ee_jacobi_idx, arm_joint_ids):
    """Return the EE Jacobian sliced to ``arm_joint_ids`` and rotated to the root frame."""
    jacobian = robot.data.body_link_jacobian_w.torch[:, ee_jacobi_idx, :, :][:, :, arm_joint_ids]
    base_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_pose_w.torch[:, 3:7]))
    jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
    return jacobian


def _build_relative_pose_target(robot, ee_frame_idx, delta_xyz, device):
    """Build a target pose = (current EE pose) + ``delta_xyz``, preserving orientation."""
    initial_ee_pos_b, initial_ee_quat_b, _ = _compute_ee_pose_root(robot, ee_frame_idx)
    target_pos_b = initial_ee_pos_b + torch.tensor([list(delta_xyz)], device=device, dtype=initial_ee_pos_b.dtype)
    return torch.cat([target_pos_b, initial_ee_quat_b], dim=-1)


def _summarize_history(history, tail: int = 200):
    """Return ``(min, mean)`` over the last ``tail`` samples."""
    tail_slice = history[-tail:]
    return min(tail_slice), sum(tail_slice) / len(tail_slice)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_franka_ik_tracking_accuracy(sim, device, gravity_enabled) -> None:
    """PhysX-side IK convergence sentinel — backend parity with the Newton test.

    Mirrors :func:`isaaclab_newton.test.assets.test_articulation.test_franka_ik_tracking_accuracy`
    so both backends are pinned by the same IK trajectory. With the
    robot teleported to its configured init_state home pose and scene
    gravity off, PhysX's IK converges to ~mm precision on this 5 cm
    Cartesian step. A bridge regression (wrong J shape, wrong DoF
    ordering) would push the steady-state error well past the
    threshold.
    """
    robot, ee_frame_idx, ee_jacobi_idx, arm_joint_ids = _setup_franka_at_home_pose(sim)

    sim.step()
    robot.update(sim.cfg.dt)
    target_pose_b = _build_relative_pose_target(robot, ee_frame_idx, (0.05, 0.0, 0.0), device)

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1,
        device=device,
    )
    ik.set_command(target_pose_b)

    pos_history: list[float] = []
    rot_history: list[float] = []
    for _ in range(800):
        jacobian = _compute_jacobian_root_frame(robot, ee_jacobi_idx, arm_joint_ids)
        ee_pos_b, ee_quat_b, _ = _compute_ee_pose_root(robot, ee_frame_idx)
        joint_pos = robot.data.joint_pos.torch[:, arm_joint_ids]

        joint_pos_des = ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        robot.set_joint_position_target(joint_pos_des, joint_ids=arm_joint_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

        pos_error, rot_error = compute_pose_error(ee_pos_b, ee_quat_b, target_pose_b[:, 0:3], target_pose_b[:, 3:7])
        pos_history.append(pos_error.norm(dim=-1).max().item())
        rot_history.append(rot_error.norm(dim=-1).max().item())

    pos_min, pos_mean = _summarize_history(pos_history)
    rot_min, rot_mean = _summarize_history(rot_history)

    print(f"IK_METRIC pos_min={pos_min:.5f} pos_mean={pos_mean:.5f} rot_min={rot_min:.5f} rot_mean={rot_mean:.5f}")

    # Assert on tail mean (not min) so an oscillating envelope can't
    # squeeze through. Threshold matched to the Newton-side test
    # (5 mm / 0.05 rad).
    assert pos_mean < 5e-3, f"IK pos_mean {pos_mean:.5f} > 5 mm — bridge regression?"
    assert rot_mean < 5e-2, f"IK rot_mean {rot_mean:.5f} > 0.05 rad — bridge regression?"


##
# Composite scene shared by the remaining tests.
##


@dataclass
class _ArticulationScene:
    """Articulation islands that share one real PhysX lifecycle."""

    sim: SimulationContext
    device: str
    ordered: Articulation
    """Fixed base, MJWarp joint and body ordering, gains and limits loaded from USD."""
    floating: Articulation
    """Floating base discovered on the base link below the spawned prim, gains and limits from the config."""
    reordered: Articulation
    """Floating base with an explicit root path and a body ordering that moves the root link."""
    tendon: Articulation
    """Fixed base with a reversed left elbow and a spatial tendon."""
    origins: dict[str, torch.Tensor]
    """World position of every environment of each island, keyed by island name."""
    refcounts: dict[str, int]
    """Reference count of each articulation right after construction."""
    cold_inertial_view_values: tuple[torch.Tensor, torch.Tensor]
    """Masses and inertias written through the floating island's view before any read."""
    cold_inertial_reads: tuple[torch.Tensor, torch.Tensor]
    """Masses and inertias of the floating island on their first read after that write."""

    @property
    def islands(self) -> dict[str, Articulation]:
        """All articulations keyed by island name."""
        return {"ordered": self.ordered, "floating": self.floating, "reordered": self.reordered, "tendon": self.tendon}

    def step(self, num_steps: int = 1) -> None:
        """Write, step, and update every articulation."""
        for _ in range(num_steps):
            for articulation in self.islands.values():
                articulation.write_data_to_sim()
            self.sim.step()
            for articulation in self.islands.values():
                articulation.update(self.sim.cfg.dt)


def _spawn_island(name: str, y_offset: float, cfg: ArticulationCfg, **authoring) -> tuple[Articulation, torch.Tensor]:
    """Spawn one two-environment island and return the articulation and its environment origins."""
    island_path = f"/World/{name}"
    sim_utils.create_prim(island_path, "Xform", translation=(0.0, y_offset, 0.0))
    origins = []
    for env_index in range(_NUM_ENVS):
        sim_utils.create_prim(f"{island_path}/Env_{env_index}", "Xform", translation=(_ENV_SPACING * env_index, 0, 0))
        origins.append((_ENV_SPACING * env_index, y_offset, 0.0))
    articulation = Articulation(replace(cfg, prim_path=f"{island_path}/Env_[^/]*/Robot"))
    for env_index in range(_NUM_ENVS):
        _author_branching_robot(f"{island_path}/Env_{env_index}/Robot", **authoring)
    return articulation, torch.tensor(origins)


@pytest.fixture(scope="module", params=test_devices())
def articulation_scene(request) -> Iterator[_ArticulationScene]:
    """Initialize every composite-scene articulation once for this module."""
    device = request.param
    with build_simulation_context(device=device, gravity_enabled=False) as sim:
        sim._app_control_on_stop_handle = None
        islands, origins, refcounts = {}, {}, {}
        island_specs = {
            "ordered": (
                _branching_cfg(
                    actuators={"joints": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
                    joint_ordering="mjwarp",
                    body_ordering="mjwarp",
                ),
                {"fixed_base": True},
            ),
            "floating": (_branching_cfg(actuators=_FLOATING_ACTUATORS), {"root_on_base": True}),
            "reordered": (
                _branching_cfg(
                    actuators=_FLOATING_ACTUATORS,
                    articulation_root_prim_path="/base",
                    body_ordering=("left_tip", "base", "right_upper", "right_tip", "left_upper"),
                ),
                {"root_on_base": True},
            ),
            "tendon": (_branching_cfg(), {"fixed_base": True, "reversed_left_elbow": True, "spatial_tendon": True}),
        }
        for index, (name, (cfg, authoring)) in enumerate(island_specs.items()):
            islands[name], origins[name] = _spawn_island(name.capitalize(), 3.0 * index, cfg, **authoring)
            refcounts[name] = sys.getrefcount(islands[name])
        sim.reset()
        # Write inertial properties through the tensor view of the floating island, whose identity body order keeps
        # the construction-time buffer timestamps, and read them once before any test steps the scene or reads these
        # buffers. Restore the view afterwards.
        floating_view = islands["floating"].root_view
        cpu_env_ids = wp.array(list(range(_NUM_ENVS)), dtype=wp.int32, device="cpu")
        view_masses = wp.to_torch(floating_view.get_masses()).clone()
        view_inertias = wp.to_torch(floating_view.get_inertias()).clone()
        cold_masses = view_masses + 0.293
        cold_inertias = view_inertias.clone()
        cold_inertias[..., [0, 4, 8]] += 0.023
        floating_view.set_masses(wp.from_torch(cold_masses, dtype=wp.float32), indices=cpu_env_ids)
        floating_view.set_inertias(wp.from_torch(cold_inertias, dtype=wp.float32), indices=cpu_env_ids)
        cold_reads = (
            islands["floating"].data.body_mass.torch.clone(),
            islands["floating"].data.body_inertia.torch.clone(),
        )
        floating_view.set_masses(wp.from_torch(view_masses, dtype=wp.float32), indices=cpu_env_ids)
        floating_view.set_inertias(wp.from_torch(view_inertias, dtype=wp.float32), indices=cpu_env_ids)
        yield _ArticulationScene(
            sim=sim,
            device=device,
            origins={name: value.to(device) for name, value in origins.items()},
            refcounts=refcounts,
            cold_inertial_view_values=(cold_masses, cold_inertias),
            cold_inertial_reads=cold_reads,
            **islands,
        )


def test_articulation_initialization_and_ordering(articulation_scene: _ArticulationScene) -> None:
    """Resolve the configured structure, orderings, gains, and limits of every real island."""
    scene = articulation_scene
    device = scene.device
    for name, articulation in scene.islands.items():
        assert articulation.is_initialized, name
        assert articulation.num_instances == _NUM_ENVS, name
        # Check that the framework doesn't hold excessive strong references.
        assert scene.refcounts[name] < 10, name
        for actuator_name, actuator in articulation.actuators.items():
            is_implicit_model_cfg = isinstance(articulation.cfg.actuators[actuator_name], ImplicitActuatorCfg)
            assert actuator.is_implicit_model == is_implicit_model_cfg
    assert scene.ordered.is_fixed_base and scene.tendon.is_fixed_base
    assert not scene.floating.is_fixed_base and not scene.reordered.is_fixed_base
    # The floating root is discovered on the base link below the spawned robot prim; the reordered root is
    # resolved through the configured root path.
    assert scene.floating.root_view.prim_paths[0] == "/World/Floating/Env_0/Robot/base"
    assert scene.reordered.root_view.prim_paths[0] == "/World/Reordered/Env_0/Robot/base"

    ordered, floating = scene.ordered, scene.floating
    assert tuple(ordered.backend_joint_names) == BRANCHING_PHYSX_JOINT_NAMES
    assert tuple(ordered.backend_body_names) == BRANCHING_PHYSX_BODY_NAMES
    assert get_articulation_name_ordering(ordered, "mjwarp", "joint") == BRANCHING_MJWARP_JOINT_NAMES
    assert get_articulation_name_ordering(ordered, "mjwarp", "body") == BRANCHING_MJWARP_BODY_NAMES
    assert tuple(ordered.joint_names) == BRANCHING_MJWARP_JOINT_NAMES
    assert tuple(ordered.body_names) == BRANCHING_MJWARP_BODY_NAMES
    assert ordered.joint_ordering is not None
    assert ordered.body_ordering is not None
    # Without a custom ordering the public body order is the PhysX link order.
    assert floating.body_ordering is None
    assert [path.split("/")[-1] for path in floating.root_view.link_paths[0]] == floating.body_names
    for articulation in (ordered, floating):
        assert articulation.data.root_pos_w.torch.shape == (_NUM_ENVS, 3)
        assert articulation.data.root_quat_w.torch.shape == (_NUM_ENVS, 4)
        assert articulation.data.joint_pos.torch.shape == (_NUM_ENVS, 4)
        assert articulation.data.body_mass.torch.shape == (_NUM_ENVS, 5)
        assert articulation.data.body_inertia.torch.shape == (_NUM_ENVS, 5, 9)

    # Unset actuator gains and limits are loaded from the USD drives, in public joint order.
    joint_backend_to_user = list(ordered.joint_ordering.backend_to_user_indices)
    for public, usd_values, raw in (
        (ordered.actuators["joints"].stiffness, _USD_STIFFNESS, ordered.root_view.get_dof_stiffnesses()),
        (ordered.actuators["joints"].damping, _USD_DAMPING, ordered.root_view.get_dof_dampings()),
        (ordered.data.joint_effort_limits.torch, _USD_MAX_FORCE, ordered.root_view.get_dof_max_forces()),
        (ordered.data.joint_vel_limits.torch, _USD_MAX_VELOCITY, ordered.root_view.get_dof_max_velocities()),
    ):
        expected = _in_user_order(usd_values, ordered.joint_names, device)
        torch.testing.assert_close(public, expected)
        torch.testing.assert_close(wp.to_torch(raw).to(device), expected[:, joint_backend_to_user])
    torch.testing.assert_close(ordered.data.joint_stiffness.torch, ordered.actuators["joints"].stiffness)
    torch.testing.assert_close(ordered.data.joint_damping.torch, ordered.actuators["joints"].damping)
    # Configured gains and limits override USD and reach the solver.
    for public, raw, values in (
        (floating.actuators["joints"].stiffness, floating.root_view.get_dof_stiffnesses(), (6.0, 4.0)),
        (floating.actuators["joints"].damping, floating.root_view.get_dof_dampings(), (0.6, 0.4)),
    ):
        per_joint = {name: values[0] if name.endswith("shoulder") else values[1] for name in floating.joint_names}
        expected = _in_user_order(per_joint, floating.joint_names, device)
        torch.testing.assert_close(public, expected)
        torch.testing.assert_close(wp.to_torch(raw).to(device), expected)
    for public, raw, limit in (
        (floating.data.joint_vel_limits.torch, floating.root_view.get_dof_max_velocities(), 7.0),
        (floating.data.joint_effort_limits.torch, floating.root_view.get_dof_max_forces(), 30.0),
    ):
        torch.testing.assert_close(public, torch.full((_NUM_ENVS, 4), limit, device=device))
        torch.testing.assert_close(wp.to_torch(raw).to(device), public)


def test_articulation_joint_state_writes_follow_ordering(articulation_scene: _ArticulationScene) -> None:
    """Joint state writes reach PhysX in backend order and reads return public order."""
    scene = articulation_scene
    device = scene.device
    articulation = scene.ordered
    joint_user_to_backend = list(articulation.joint_ordering.user_to_backend_indices)
    joint_backend_to_user = list(articulation.joint_ordering.backend_to_user_indices)
    body_user_to_backend = list(articulation.body_ordering.user_to_backend_indices)

    # Full-data writes land in backend order.
    joint_pos = torch.linspace(-0.3, 0.3, 4, device=device).repeat(_NUM_ENVS, 1)
    joint_vel = torch.linspace(0.05, 0.13, 4, device=device).repeat(_NUM_ENVS, 1)
    initial_stiffness = articulation.data.joint_stiffness.torch.clone()
    joint_stiffness = 10.0 + torch.arange(4, device=device, dtype=torch.float32).repeat(_NUM_ENVS, 1)
    original_body_link_pose_w = articulation.data.body_link_pose_w.torch.clone()
    articulation.write_joint_stiffness_to_sim_index(stiffness=joint_stiffness, full_data=True)
    articulation.write_joint_position_to_sim_index(position=joint_pos, full_data=True)
    articulation.write_joint_velocity_to_sim_index(velocity=joint_vel, full_data=True)
    for raw, written in (
        (articulation.root_view.get_dof_positions(), joint_pos),
        (articulation.root_view.get_dof_velocities(), joint_vel),
        (articulation.root_view.get_dof_stiffnesses(), joint_stiffness),
    ):
        torch.testing.assert_close(wp.to_torch(raw).to(device), written[:, joint_backend_to_user])

    # A joint write refreshes the body state without stepping.
    body_link_pose_w = articulation.data.body_link_pose_w.torch
    assert torch.all((original_body_link_pose_w[:, 1:] != body_link_pose_w[:, 1:]).any(dim=-1))
    body_com_pose_b = articulation.data.body_com_pose_b.torch
    expected_com_pos, expected_com_quat = combine_frame_transforms(
        body_link_pose_w[..., :3].reshape(-1, 3),
        body_link_pose_w[..., 3:].reshape(-1, 4),
        body_com_pose_b[..., :3].reshape(-1, 3),
        body_com_pose_b[..., 3:].reshape(-1, 4),
    )
    torch.testing.assert_close(expected_com_pos.view(_NUM_ENVS, -1, 3), articulation.data.body_com_pos_w.torch)
    torch.testing.assert_close(expected_com_quat.view(_NUM_ENVS, -1, 4), articulation.data.body_com_quat_w.torch)
    body_com_vel_w = articulation.data.body_com_vel_w.torch
    torch.testing.assert_close(body_com_vel_w[..., 3:], articulation.data.body_link_vel_w.torch[..., 3:])
    torch.testing.assert_close(body_com_vel_w[..., :3], articulation.data.body_com_lin_vel_w.torch)
    torch.testing.assert_close(body_com_vel_w[..., 3:], articulation.data.body_com_ang_vel_w.torch)

    # After a step, public reads are the backend state in public order.
    scene.step()
    for public, raw, user_to_backend in (
        (articulation.data.joint_pos.torch, articulation.root_view.get_dof_positions(), joint_user_to_backend),
        (articulation.data.joint_vel.torch, articulation.root_view.get_dof_velocities(), joint_user_to_backend),
        (articulation.data.joint_stiffness.torch, articulation.root_view.get_dof_stiffnesses(), joint_user_to_backend),
        (articulation.data.body_link_pose_w.torch, articulation.root_view.get_link_transforms(), body_user_to_backend),
        (articulation.data.body_com_pose_b.torch, articulation.root_view.get_coms(), body_user_to_backend),
    ):
        torch.testing.assert_close(public, wp.to_torch(raw).to(device)[:, user_to_backend])
    # The split accessors must be sliced from the reordered public poses, not from a stale backend-order cache.
    data = articulation.data
    torch.testing.assert_close(data.body_com_pos_b.torch, data.body_com_pose_b.torch[..., :3])
    torch.testing.assert_close(data.body_com_quat_b.torch, data.body_com_pose_b.torch[..., 3:])
    torch.testing.assert_close(data.body_pos_w.torch, data.body_link_pose_w.torch[..., :3])
    torch.testing.assert_close(data.body_quat_w.torch, data.body_link_pose_w.torch[..., 3:])

    # Partial writes with int64 selectors in non-sorted order reach only the selected cells.
    env_ids = torch.tensor([1], dtype=torch.int64, device=device)
    joint_ids = torch.tensor([3, 0], dtype=torch.int64, device=device)
    position = torch.tensor([[0.21, -0.13]], device=device)
    velocity = torch.tensor([[0.41, -0.23]], device=device)
    expected_position = articulation.data.joint_pos.torch.clone()
    expected_velocity = articulation.data.joint_vel.torch.clone()
    expected_position[env_ids[:, None], joint_ids] = position
    expected_velocity[env_ids[:, None], joint_ids] = velocity
    articulation.write_joint_state_to_sim_index(
        position=position, velocity=velocity, env_ids=env_ids, joint_ids=joint_ids
    )
    torch.testing.assert_close(articulation.data.joint_pos.torch, expected_position)
    torch.testing.assert_close(articulation.data.joint_vel.torch, expected_velocity)
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_dof_positions()).to(device),
        expected_position[:, joint_backend_to_user],
    )
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_dof_velocities()).to(device),
        expected_velocity[:, joint_backend_to_user],
    )
    # Restore the USD gains for the following tests.
    articulation.write_joint_stiffness_to_sim_index(stiffness=initial_stiffness, full_data=True)


def test_articulation_joint_and_body_properties_round_trip(articulation_scene: _ArticulationScene) -> None:
    """Selected joint and body properties reach the selected PhysX entries and read back in public order."""
    scene = articulation_scene
    device = scene.device

    # Direct tensor-view mass and inertia writes become visible on the lazy read: on the first read of a cold
    # buffer, taken by the fixture before any test step, and after the next update once the buffer is primed.
    for cold_read, view_values in zip(scene.cold_inertial_reads, scene.cold_inertial_view_values):
        torch.testing.assert_close(cold_read, view_values.to(device))
    cpu_env_ids = wp.array(list(range(_NUM_ENVS)), dtype=wp.int32, device="cpu")

    def write_backend_mass_inertia(
        articulation: Articulation, delta_mass: float, delta_inertia: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write distinct backend-order masses/inertias straight through the tensor view."""
        backend_masses = wp.to_torch(articulation.root_view.get_masses()).clone() + delta_mass
        backend_inertias = wp.to_torch(articulation.root_view.get_inertias()).clone()
        backend_inertias[..., [0, 4, 8]] += delta_inertia
        articulation.root_view.set_masses(wp.from_torch(backend_masses, dtype=wp.float32), indices=cpu_env_ids)
        articulation.root_view.set_inertias(wp.from_torch(backend_inertias, dtype=wp.float32), indices=cpu_env_ids)
        return backend_masses.to(device), backend_inertias.to(device)

    articulation = scene.ordered
    body_user_to_backend = list(articulation.body_ordering.user_to_backend_indices)
    body_backend_to_user = list(articulation.body_ordering.backend_to_user_indices)
    joint_backend_to_user = list(articulation.joint_ordering.backend_to_user_indices)
    # Reading the buffers primes them; the later tests drive the island with these initial properties.
    initial_properties = {
        "masses": articulation.data.body_mass.torch.clone(),
        "coms": articulation.data.body_com_pose_b.torch.clone(),
        "inertias": articulation.data.body_inertia.torch.clone(),
    }
    backend_masses, backend_inertias = write_backend_mass_inertia(articulation, 0.137, 0.011)
    articulation.update(scene.sim.cfg.dt)
    torch.testing.assert_close(articulation.data.body_mass.torch, backend_masses[:, body_user_to_backend])
    torch.testing.assert_close(articulation.data.body_inertia.torch, backend_inertias[:, body_user_to_backend])

    # Joint friction coefficients reach their TensorAPI slots for the selected cells only.
    env_ids = torch.tensor([1], dtype=torch.int32, device=device)
    joint_ids = torch.tensor([3, 0], dtype=torch.int32, device=device)
    raw_friction_before = wp.to_torch(articulation.root_view.get_dof_friction_properties()).clone()
    initial_frictions = {
        "joint_friction_coeff": articulation.data.joint_friction_coeff.torch.clone(),
        "joint_dynamic_friction_coeff": articulation.data.joint_dynamic_friction_coeff.torch.clone(),
        "joint_viscous_friction_coeff": articulation.data.joint_viscous_friction_coeff.torch.clone(),
    }
    frictions = {
        "joint_friction_coeff": torch.tensor([[0.9, 0.7]], device=device),
        "joint_dynamic_friction_coeff": torch.tensor([[0.4, 0.3]], device=device),
        "joint_viscous_friction_coeff": torch.tensor([[0.11, 0.22]], device=device),
    }
    friction_readers = {
        "joint_friction_coeff": lambda: articulation.data.joint_friction_coeff.torch,
        "joint_dynamic_friction_coeff": lambda: articulation.data.joint_dynamic_friction_coeff.torch,
        "joint_viscous_friction_coeff": lambda: articulation.data.joint_viscous_friction_coeff.torch,
    }
    expected_frictions = {name: read().clone() for name, read in friction_readers.items()}
    articulation.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=frictions["joint_friction_coeff"],
        joint_dynamic_friction_coeff=frictions["joint_dynamic_friction_coeff"],
        joint_viscous_friction_coeff=frictions["joint_viscous_friction_coeff"],
        env_ids=env_ids,
        joint_ids=joint_ids,
    )
    expected_raw_friction = raw_friction_before.clone()
    for component, (name, values) in enumerate(frictions.items()):
        expected_frictions[name][env_ids[:, None], joint_ids] = values
        torch.testing.assert_close(friction_readers[name](), expected_frictions[name])
        expected_raw_friction[..., component] = expected_frictions[name][:, joint_backend_to_user].cpu()
    torch.testing.assert_close(wp.to_torch(articulation.root_view.get_dof_friction_properties()), expected_raw_friction)

    # Joint position limits reach the solver; limits that exclude a default position clamp it.
    raw_limits_before = wp.to_torch(articulation.root_view.get_dof_limits()).clone()
    initial_limits = articulation.data.joint_pos_limits.torch.clone()
    default_joint_pos = articulation.data.default_joint_pos.torch.clone()
    limits = torch.tensor([[[-1.5, 1.25], [-1.0, 0.75]]], device=device)
    articulation.write_joint_position_limit_to_sim_index(limits=limits, env_ids=env_ids, joint_ids=joint_ids)
    expected_limits = initial_limits.clone()
    expected_limits[env_ids[:, None], joint_ids] = limits
    torch.testing.assert_close(articulation.data.joint_pos_limits.torch, expected_limits)
    torch.testing.assert_close(articulation.data.default_joint_pos.torch, default_joint_pos)
    expected_raw_limits = raw_limits_before.clone()
    expected_raw_limits[1] = expected_limits[1, joint_backend_to_user].cpu()
    torch.testing.assert_close(wp.to_torch(articulation.root_view.get_dof_limits()), expected_raw_limits)
    articulation.write_joint_position_limit_to_sim_index(
        limits=torch.tensor([[[0.05, 0.2], [-0.2, -0.1]]], device=device), env_ids=env_ids, joint_ids=joint_ids
    )
    expected_defaults = default_joint_pos.clone()
    expected_defaults[1, joint_ids] = torch.tensor([0.05, -0.1], device=device)
    torch.testing.assert_close(articulation.data.default_joint_pos.torch, expected_defaults)
    articulation.write_joint_position_limit_to_sim_index(limits=initial_limits, full_data=True)

    # Inertial properties of selected bodies reach the backend entries of the reordered bodies.
    body_ids = torch.tensor([4, 1], dtype=torch.int32, device=device)
    masses = torch.tensor([[2.5, 3.5]], device=device)
    coms = articulation.data.body_com_pose_b.torch[env_ids][:, body_ids].clone()
    coms[0, 0, :3] = torch.tensor([0.02, -0.01, 0.03], device=device)
    coms[0, 1, :3] = torch.tensor([-0.03, 0.01, 0.02], device=device)
    inertias = articulation.data.body_inertia.torch[env_ids][:, body_ids].clone()
    inertias[0, 0, 0] *= 1.2
    inertias[0, 1, 4] *= 1.3
    expected_masses = articulation.data.body_mass.torch.clone()
    expected_masses[env_ids[:, None], body_ids] = masses
    articulation.set_masses_index(masses=masses, env_ids=env_ids, body_ids=body_ids)
    torch.testing.assert_close(articulation.data.body_mass.torch, expected_masses)
    expected_coms = articulation.data.body_com_pose_b.torch.clone()
    expected_coms[env_ids[:, None], body_ids] = coms
    articulation.set_coms_index(coms=coms, env_ids=env_ids, body_ids=body_ids)
    torch.testing.assert_close(articulation.data.body_com_pose_b.torch, expected_coms)
    expected_inertias = articulation.data.body_inertia.torch.clone()
    expected_inertias[env_ids[:, None], body_ids] = inertias
    articulation.set_inertias_index(inertias=inertias, env_ids=env_ids, body_ids=body_ids)
    torch.testing.assert_close(articulation.data.body_inertia.torch, expected_inertias)
    for public, raw in (
        (articulation.data.body_mass.torch, articulation.root_view.get_masses()),
        (articulation.data.body_com_pose_b.torch, articulation.root_view.get_coms()),
        (articulation.data.body_inertia.torch, articulation.root_view.get_inertias()),
    ):
        torch.testing.assert_close(wp.to_torch(raw).to(device), public[:, body_backend_to_user])

    # Restore the initial properties for the tests that drive this island.
    articulation.set_masses_index(masses=initial_properties["masses"], full_data=True)
    articulation.set_coms_index(coms=initial_properties["coms"], full_data=True)
    articulation.set_inertias_index(inertias=initial_properties["inertias"], full_data=True)
    articulation.write_joint_friction_coefficient_to_sim_index(**initial_frictions, full_data=True)


def _rotate_z(angle: torch.Tensor, vector: tuple[float, float, float] | torch.Tensor) -> torch.Tensor:
    """Rotate vectors about the world z axis.

    Args:
        angle: Rotation angles [rad], shape [N].
        vector: Vector to rotate, shape [3] or [N, 3].

    Returns:
        Rotated vectors, shape [N, 3].
    """
    vector = torch.as_tensor(vector, dtype=angle.dtype, device=angle.device).expand(*angle.shape, 3)
    cos, sin = torch.cos(angle), torch.sin(angle)
    return torch.stack(
        (cos * vector[..., 0] - sin * vector[..., 1], sin * vector[..., 0] + cos * vector[..., 1], vector[..., 2]), -1
    )


def _cross_z(rate: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Return ``(rate * z) x vector``, the velocity induced by a rotation about the world z axis.

    Args:
        rate: Angular rates about the world z axis [rad/s], shape [N].
        vector: Lever arms in the world frame [m], shape [N, 3].

    Returns:
        Induced linear velocities [m/s], shape [N, 3].
    """
    return torch.stack((-rate * vector[..., 1], rate * vector[..., 0], torch.zeros_like(rate)), -1)


def _expected_fixed_branching_state(
    articulation: Articulation, origins: torch.Tensor, com_b: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Compute link and center-of-mass kinematics of a fixed branching island from its joint state.

    Every joint rotates about +z and is anchored at its child link origin, so the expected state follows from
    the literal fixture geometry, the joint state, and the body-frame center-of-mass offsets.

    Args:
        articulation: Fixed-base branching articulation.
        origins: World position of each environment [m], shape [N, 3].
        com_b: Center-of-mass offsets in the link frames [m], shape [N, B, 3], in public body order.

    Returns:
        World-frame link x axes (``heading``), link and center-of-mass positions [m], angular velocities
        [rad/s], and link and center-of-mass linear velocities [m/s], each of shape [N, B, 3].
    """
    q = dict(zip(articulation.joint_names, articulation.data.joint_pos.torch.unbind(-1)))
    qd = dict(zip(articulation.joint_names, articulation.data.joint_vel.torch.unbind(-1)))
    zero = torch.zeros_like(q["left_shoulder"])
    links = {"base": (zero, origins, zero, torch.zeros_like(origins))}
    for side in ("left", "right"):
        shoulder, elbow = f"{side}_shoulder", f"{side}_elbow"
        upper_pos = origins + torch.tensor(_JOINT_ANCHORS[shoulder], device=origins.device)
        tip_lever = _rotate_z(q[shoulder], _JOINT_ANCHORS[elbow])
        links[f"{side}_upper"] = (q[shoulder], upper_pos, qd[shoulder], torch.zeros_like(origins))
        links[f"{side}_tip"] = (
            q[shoulder] + q[elbow],
            upper_pos + tip_lever,
            qd[shoulder] + qd[elbow],
            _cross_z(qd[shoulder], tip_lever),
        )
    expected = {key: [] for key in ("heading", "link_pos", "ang_vel", "link_lin_vel", "com_pos", "com_lin_vel")}
    for body_index, body_name in enumerate(articulation.body_names):
        yaw, link_pos, yaw_rate, link_lin_vel = links[body_name]
        com_lever = _rotate_z(yaw, com_b[:, body_index])
        expected["heading"].append(_rotate_z(yaw, (1.0, 0.0, 0.0)))
        expected["link_pos"].append(link_pos)
        expected["ang_vel"].append(torch.stack((zero, zero, yaw_rate), -1))
        expected["link_lin_vel"].append(link_lin_vel)
        expected["com_pos"].append(link_pos + com_lever)
        expected["com_lin_vel"].append(link_lin_vel + _cross_z(yaw_rate, com_lever))
    return {key: torch.stack(values, dim=1) for key, values in expected.items()}


def _assert_jacobian_contract(articulation: Articulation, generalized_velocity: torch.Tensor) -> None:
    """Check the link and center-of-mass Jacobians against body velocities and the mass-matrix contract.

    Args:
        articulation: Articulation whose state was just written.
        generalized_velocity: Base and joint velocities, shape [N, num_base_dofs + num_joints].
    """
    data = articulation.data
    num_dofs = articulation.num_joints + articulation.num_base_dofs
    first_body = 1 if articulation.is_fixed_base else 0
    num_jacobian_bodies = articulation.num_bodies - first_body
    # Read the body state first: it refreshes forward kinematics after the joint write.
    lin_vels = (data.body_link_lin_vel_w.torch.clone(), data.body_com_lin_vel_w.torch.clone())
    ang_vel = data.body_link_ang_vel_w.torch.clone()
    for jacobian, lin_vel in zip((data.body_link_jacobian_w.torch, data.body_com_jacobian_w.torch), lin_vels):
        # Shape contract: fixed-base Jacobians omit the root body; floating-base ones prepend the base DoFs.
        assert jacobian.shape == (_NUM_ENVS, num_jacobian_bodies, 6, num_dofs)
        predicted = torch.einsum("nbij,nj->nbi", jacobian, generalized_velocity)
        torch.testing.assert_close(predicted[..., :3], lin_vel[:, first_body:], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(predicted[..., 3:], ang_vel[:, first_body:], atol=1e-4, rtol=1e-4)
    # The joint-space mass matrix is square over all DoFs, symmetric, and positive definite.
    mass_matrix = data.mass_matrix.torch
    assert mass_matrix.shape == (_NUM_ENVS, num_dofs, num_dofs)
    assert (mass_matrix.diagonal(dim1=-2, dim2=-1) > 1e-6).all()
    asymmetry = (mass_matrix - mass_matrix.transpose(-1, -2)).abs().max().item()
    assert asymmetry < 1e-4, f"|M - M^T|_max = {asymmetry:.3e} — mass matrix is not symmetric"
    eye = torch.eye(num_dofs, device=mass_matrix.device, dtype=mass_matrix.dtype).expand_as(mass_matrix)
    torch.linalg.cholesky(mass_matrix + 1e-6 * eye)
    # The generalized kinetic energy equals the kinetic energy of the bodies.
    generalized_energy = 0.5 * torch.einsum("ni,nij,nj->n", generalized_velocity, mass_matrix, generalized_velocity)
    body_velocity = data.body_com_vel_w.torch
    body_inertia = data.body_inertia.torch.reshape(_NUM_ENVS, articulation.num_bodies, 3, 3)
    body_rotation = matrix_from_quat(data.body_com_quat_w.torch)
    world_inertia = body_rotation @ body_inertia @ body_rotation.transpose(-1, -2)
    body_energy = 0.5 * (
        (data.body_mass.torch.unsqueeze(-1) * body_velocity[..., :3].square()).sum((-1, -2))
        + torch.einsum("nbi,nbij,nbj->n", body_velocity[..., 3:], world_inertia, body_velocity[..., 3:])
    )
    torch.testing.assert_close(generalized_energy, body_energy, atol=1e-5, rtol=1e-4)


@pytest.mark.isaacsim_ci
def test_articulation_drive_and_dynamics(articulation_scene: _ArticulationScene) -> None:
    """Drive targets and efforts move the selected joints; dynamics quantities match the live state."""
    scene = articulation_scene
    device = scene.device
    articulation = scene.ordered
    joint_backend_to_user = list(articulation.joint_ordering.backend_to_user_indices)
    left_shoulder = articulation.find_joints("left_shoulder")[0][0]
    right_shoulder = articulation.find_joints("right_shoulder")[0][0]

    # Start at rest on the drive targets, then command one position target and one effort in environment 1.
    zeros = torch.zeros((_NUM_ENVS, articulation.num_joints), device=device)
    articulation.write_joint_state_to_sim_index(position=zeros, velocity=zeros, full_data=True)
    target_command = articulation.actuators.target_command
    target_command.set_position_index(value=zeros)
    target_command.set_velocity_index(value=zeros)
    target_command.set_effort_index(value=zeros)
    target_command.set_position_index(
        value=torch.tensor([[0.4]], device=device), joint_ids=[left_shoulder], env_ids=[1]
    )
    target_command.set_effort_index(
        value=torch.tensor([[10.0]], device=device), joint_ids=[right_shoulder], env_ids=[1]
    )
    articulation.write_data_to_sim()
    expected_targets = zeros.clone()
    expected_targets[1, left_shoulder] = 0.4
    expected_efforts = zeros.clone()
    expected_efforts[1, right_shoulder] = 10.0
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_dof_position_targets()).to(device),
        expected_targets[:, joint_backend_to_user],
    )
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_dof_actuation_forces()).to(device),
        expected_efforts[:, joint_backend_to_user],
    )
    # The effort must reach the solver, not only its staging buffer.
    scene.step()
    assert articulation.data.joint_vel.torch[1, right_shoulder] > 1e-3, "a positive effort must accelerate the joint"
    scene.step(19)
    # The commanded joint moves toward its target; environment 0 stays at rest on its targets.
    assert 0.05 < articulation.data.joint_pos.torch[1, left_shoulder] < 0.4
    torch.testing.assert_close(articulation.data.joint_pos.torch[0], zeros[0], atol=1e-6, rtol=0.0)
    torch.testing.assert_close(articulation.data.joint_vel.torch[0], zeros[0], atol=1e-6, rtol=0.0)
    # The fixed root stays at its default state.
    for fixed_articulation, origins in (
        (articulation, scene.origins["ordered"]),
        (scene.tendon, scene.origins["tendon"]),
    ):
        default_root_pose = fixed_articulation.data.default_root_pose.torch.clone()
        default_root_pose[:, :3] += origins
        torch.testing.assert_close(fixed_articulation.data.root_link_pose_w.torch, default_root_pose)
        torch.testing.assert_close(
            fixed_articulation.data.root_com_vel_w.torch, fixed_articulation.data.default_root_vel.torch
        )

    # With known center-of-mass offsets, the link and center-of-mass kinematics follow the joint state.
    com_b = torch.tensor(
        [[0.0, 0.0, 0.0], [0.03, 0.02, -0.01], [0.05, -0.02, 0.01], [-0.04, 0.01, 0.02], [0.02, 0.03, -0.02]],
        device=device,
    ).repeat(_NUM_ENVS, 1, 1)
    coms = torch.cat((com_b, torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).expand(_NUM_ENVS, 5, 4)), dim=-1)
    articulation.set_coms_index(coms=coms, full_data=True)
    # PhysX applies new mass properties to the kinematic state on the next step.
    scene.step()
    joint_pos = torch.tensor([[0.3, -0.4, 0.2, 0.5], [-0.2, 0.6, -0.5, 0.1]], device=device)
    joint_vel = torch.tensor([[0.8, -0.6, 0.5, 0.7], [-0.4, 0.9, 0.3, -0.5]], device=device)
    articulation.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, full_data=True)
    expected = _expected_fixed_branching_state(articulation, scene.origins["ordered"], com_b)
    data = articulation.data
    torch.testing.assert_close(
        quat_apply(data.body_link_quat_w.torch, torch.tensor([1.0, 0.0, 0.0], device=device).expand(_NUM_ENVS, 5, 3)),
        expected["heading"],
        atol=1e-4,
        rtol=1e-4,
    )
    for public, key in (
        (data.body_link_pos_w.torch, "link_pos"),
        (data.body_link_lin_vel_w.torch, "link_lin_vel"),
        (data.body_link_ang_vel_w.torch, "ang_vel"),
        (data.body_com_pos_w.torch, "com_pos"),
        (data.body_com_lin_vel_w.torch, "com_lin_vel"),
        (data.body_com_ang_vel_w.torch, "ang_vel"),
    ):
        torch.testing.assert_close(public, expected[key], atol=1e-4, rtol=1e-4, msg=key)
    _assert_jacobian_contract(articulation, data.joint_vel.torch)

    # A reversed joint keeps the dynamics in the public joint basis; floating Jacobians prepend the base DoFs and,
    # with center-of-mass offsets on the moving links, shift their linear rows to the link origins.
    scene.floating.set_coms_index(coms=coms, full_data=True)
    scene.step()
    for island in (scene.tendon, scene.floating):
        island.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, full_data=True)
    root_velocity = torch.tensor([[0.3, -0.2, 0.1, 0.2, 0.1, -0.3], [-0.1, 0.2, 0.3, -0.2, 0.3, 0.1]], device=device)
    scene.floating.write_root_link_velocity_to_sim_index(root_velocity=root_velocity)
    _assert_jacobian_contract(scene.tendon, scene.tendon.data.joint_vel.torch)
    floating_velocity = torch.cat(
        (scene.floating.data.root_link_vel_w.torch, scene.floating.data.joint_vel.torch), dim=-1
    )
    _assert_jacobian_contract(scene.floating, floating_velocity)

    # Each Jacobian and mass-matrix read independently reflects a manual joint write without stepping.
    for island in (articulation, scene.floating):
        q_initial = island.data.joint_pos.torch.clone()[:1]
        env_ids = wp.array([0], dtype=wp.int32, device=device)
        for property_name in ("body_link_jacobian_w", "body_com_jacobian_w", "mass_matrix"):
            island.write_joint_position_to_sim_index(position=q_initial, env_ids=env_ids)
            before = getattr(island.data, property_name).torch.clone()
            # A separate write for each getter prevents another getter from refreshing FK on its behalf.
            island.write_joint_position_to_sim_index(position=q_initial + 0.5, env_ids=env_ids)
            after = getattr(island.data, property_name).torch.clone()
            assert not torch.allclose(before, after, atol=1e-3), f"{property_name} stayed stale after a joint write"


def _place_at_rest(articulation: Articulation, root_pose: torch.Tensor) -> None:
    """Teleport a floating island to ``root_pose`` with zero root and joint velocities and no external wrench."""
    zeros = torch.zeros((_NUM_ENVS, articulation.num_joints), device=root_pose.device)
    articulation.write_root_link_pose_to_sim_index(root_pose=root_pose)
    articulation.write_root_velocity_to_sim_index(root_velocity=torch.zeros((_NUM_ENVS, 6), device=root_pose.device))
    articulation.write_joint_state_to_sim_index(position=zeros, velocity=zeros, full_data=True)
    articulation.reset()


def test_floating_articulation_root_writes_and_wrenches(articulation_scene: _ArticulationScene, monkeypatch) -> None:
    """Root writes and external wrenches reach the selected floating roots in the frames they are given in."""
    scene = articulation_scene
    device = scene.device
    floating, reordered = scene.floating, scene.reordered
    rest_pose = torch.cat(
        (scene.origins["floating"], torch.tensor(_yaw_quat(0.5 * math.pi), device=device).expand(_NUM_ENVS, 4)), -1
    )
    _place_at_rest(floating, rest_pose)
    reordered_rest_pose = rest_pose.clone()
    reordered_rest_pose[:, :3] = scene.origins["reordered"]
    _place_at_rest(reordered, reordered_rest_pose)

    # A partial root pose write reaches only the selected root.
    target_pose = rest_pose[1:].clone()
    target_pose[:, :3] += torch.tensor([0.2, -0.1, 0.3], device=device)
    target_pose[:, 3:] = torch.tensor(_yaw_quat(0.7), device=device)
    floating.write_root_link_pose_to_sim_index(root_pose=target_pose, env_ids=[1])
    expected_pose = torch.cat((rest_pose[:1], target_pose))
    torch.testing.assert_close(floating.data.root_link_pose_w.torch, expected_pose)
    torch.testing.assert_close(wp.to_torch(floating.root_view.get_root_transforms()).to(device), expected_pose)

    # Root writers are invariant to a public body order that moves the root link.
    assert reordered.body_ordering is not None
    assert reordered.body_ordering.backend_to_user_indices[0] != 0
    backend_coms = torch.zeros((_NUM_ENVS, 5, 7), device=device)
    body_index = torch.arange(5, device=device, dtype=torch.float32)
    backend_coms[:, :, 0] = 0.05 + 0.01 * body_index
    backend_coms[:, :, 1] = -0.03 - 0.02 * body_index
    backend_coms[:, :, 2] = 0.02 + 0.03 * body_index
    backend_coms[..., 6] = 1.0
    reordered_user_to_backend = list(reordered.body_ordering.user_to_backend_indices)
    floating.set_coms_index(coms=backend_coms, full_data=True)
    reordered.set_coms_index(coms=backend_coms[:, reordered_user_to_backend], full_data=True)
    torch.testing.assert_close(wp.to_torch(floating.root_view.get_coms()).to(device), backend_coms)
    torch.testing.assert_close(wp.to_torch(reordered.root_view.get_coms()).to(device), backend_coms)
    root_com_pose = rest_pose.clone()
    root_com_pose[:, :3] += torch.tensor([0.5, 0.25, 0.75], device=device)
    root_com_pose[:, 3:] = torch.tensor(_yaw_quat(0.6), device=device)
    island_offset = scene.origins["reordered"] - scene.origins["floating"]
    root_link_velocity = torch.tensor([[0.4, -0.3, 0.2, 1.1, -0.7, 0.9]], device=device).repeat(_NUM_ENVS, 1)
    for articulation, offset in ((floating, 0.0), (reordered, island_offset)):
        pose = root_com_pose.clone()
        pose[:, :3] += offset
        articulation.write_root_com_pose_to_sim_index(root_pose=pose)
        articulation.write_root_link_velocity_to_sim_index(root_velocity=root_link_velocity)
        torch.testing.assert_close(articulation.data.root_com_pose_w.torch, pose)
        torch.testing.assert_close(articulation.data.root_link_vel_w.torch, root_link_velocity)
    reordered_transforms = wp.to_torch(reordered.root_view.get_root_transforms()).to(device).clone()
    reordered_transforms[:, :3] -= island_offset
    torch.testing.assert_close(reordered_transforms, wp.to_torch(floating.root_view.get_root_transforms()).to(device))
    torch.testing.assert_close(
        wp.to_torch(reordered.root_view.get_root_velocities()).to(device),
        wp.to_torch(floating.root_view.get_root_velocities()).to(device),
    )
    torch.testing.assert_close(reordered.data.root_com_vel_w.torch, floating.data.root_com_vel_w.torch)
    # The derived root link pose carries the written center-of-mass pose at the root's center-of-mass offset.
    for articulation, offset in ((floating, 0.0), (reordered, island_offset)):
        link_pose = articulation.data.root_link_pose_w.torch
        root_com_b = articulation.data.body_com_pose_b.torch[:, articulation.body_names.index("base")]
        com_pos, com_quat = combine_frame_transforms(
            link_pose[:, :3], link_pose[:, 3:], root_com_b[:, :3], root_com_b[:, 3:]
        )
        expected_com_pose = root_com_pose.clone()
        expected_com_pose[:, :3] += offset
        torch.testing.assert_close(torch.cat((com_pos, com_quat), -1), expected_com_pose)

    # Link-pose and center-of-mass velocity writes keep the other frame consistent with the offset.
    floating.write_root_link_pose_to_sim_index(root_pose=root_com_pose)
    floating.write_root_com_velocity_to_sim_index(root_velocity=root_link_velocity)
    torch.testing.assert_close(floating.data.root_link_pose_w.torch, root_com_pose)
    torch.testing.assert_close(floating.data.root_com_vel_w.torch, root_link_velocity)
    expected_com_pos, expected_com_quat = combine_frame_transforms(
        root_com_pose[:, :3], root_com_pose[:, 3:], backend_coms[:, 0, :3], backend_coms[:, 0, 3:]
    )
    torch.testing.assert_close(
        floating.data.root_com_pose_w.torch, torch.cat((expected_com_pos, expected_com_quat), -1)
    )
    torch.testing.assert_close(floating.data.root_com_vel_w.torch[:, 3:], floating.data.root_link_vel_w.torch[:, 3:])

    # A body-frame force on environment 1 accelerates only that root, along the rotated force direction.
    base = [floating.find_bodies("base")[0][0]]
    _place_at_rest(floating, rest_pose)
    floating.permanent_wrench_composer.set_forces_and_torques_index(
        forces=torch.tensor([[[8.0, 0.0, 0.0]]], device=device), body_ids=base, env_ids=[1]
    )
    scene.step()
    root_velocity = floating.data.root_com_lin_vel_w.torch
    assert root_velocity[1, 1] > 0.01, f"body-frame force did not act along the rotated axis: {root_velocity[1]}"
    torch.testing.assert_close(root_velocity[0], torch.zeros(3, device=device), atol=1e-6, rtol=0.0)

    # A world-frame force, and a force at a world-frame position, act like their body-frame counterparts.
    for local_wrench, global_wrench, response in (
        ({"forces": [[[8.0, 0.0, 0.0]]]}, {"forces": [[[0.0, 8.0, 0.0]]]}, lambda: floating.data.root_com_lin_vel_w),
        (
            {"forces": [[[0.0, 0.0, 8.0]]], "positions": [[[0.0, 1.0, 0.0]]]},
            {"forces": [[[0.0, 0.0, 8.0]]], "positions": [[[-1.0, 0.0, 0.0]]]},
            lambda: floating.data.root_ang_vel_b,
        ),
    ):
        _place_at_rest(floating, rest_pose)
        local_wrench = {key: torch.tensor(value, device=device) for key, value in local_wrench.items()}
        global_wrench = {key: torch.tensor(value, device=device) for key, value in global_wrench.items()}
        if "positions" in global_wrench:
            # World-frame positions add the rotated body-frame lever arm to the base center of mass.
            global_wrench["positions"] = global_wrench["positions"] + floating.data.body_com_pos_w.torch[:1, base]
        floating.permanent_wrench_composer.set_forces_and_torques_index(body_ids=base, env_ids=[1], **local_wrench)
        floating.permanent_wrench_composer.set_forces_and_torques_index(
            body_ids=base, env_ids=[0], is_global=True, **global_wrench
        )
        scene.step()
        response_value = response().torch
        torch.testing.assert_close(response_value[0], response_value[1], atol=1e-4, rtol=1e-3)
    # Applied 1 m along the base y-axis instead of at the center of mass, the upward force also rolls the base
    # about its x-axis.
    _place_at_rest(floating, rest_pose)
    upward_force = torch.tensor([[[0.0, 0.0, 8.0]]], device=device)
    floating.permanent_wrench_composer.set_forces_and_torques_index(forces=upward_force, body_ids=base, env_ids=[0])
    floating.permanent_wrench_composer.set_forces_and_torques_index(
        forces=upward_force, positions=torch.tensor([[[0.0, 1.0, 0.0]]], device=device), body_ids=base, env_ids=[1]
    )
    scene.step()
    roll_rate = floating.data.root_ang_vel_b.torch[:, 0]
    assert roll_rate[1] - roll_rate[0] > 0.1, roll_rate

    # Forces on several bodies push each selected body along its own force direction. Under a public body order
    # that differs from the PhysX link order, every body responds as in the identity-ordered island.
    tip_forces = torch.tensor([[[0.0, 4.0, 0.0], [0.0, -4.0, 0.0]]], device=device)
    for articulation, pose in ((floating, rest_pose), (reordered, reordered_rest_pose)):
        _place_at_rest(articulation, pose)
        tips = articulation.find_bodies(["left_tip", "right_tip"], preserve_order=True)[0]
        articulation.permanent_wrench_composer.set_forces_and_torques_index(
            forces=tip_forces, body_ids=tips, env_ids=[1]
        )
        # Refresh the link transforms after the teleport: on the GPU pipeline, the step applies link-frame wrenches
        # with the link transforms of the last kinematic update.
        articulation.data.body_link_pose_w.torch
    scene.step()
    tips = reordered.find_bodies(["left_tip", "right_tip"], preserve_order=True)[0]
    tip_velocity = reordered.data.body_link_lin_vel_w.torch[:, tips]
    tip_direction = quat_apply(reordered.data.body_link_quat_w.torch[1, tips], tip_forces[0])
    assert torch.all((tip_velocity[1] * tip_direction).sum(-1) > 1e-3), tip_velocity[1]
    torch.testing.assert_close(tip_velocity[0], torch.zeros_like(tip_velocity[0]), atol=1e-6, rtol=0.0)
    floating_order = [reordered.body_names.index(name) for name in floating.body_names]
    torch.testing.assert_close(
        reordered.data.body_link_lin_vel_w.torch[:, floating_order], floating.data.body_link_lin_vel_w.torch
    )

    # A partial reset clears the selected environment; a full reset resets every actuator environment and
    # clears all external forces and torques.
    actuator = next(iter(floating.actuators.values()))
    actuator_reset = actuator.reset
    reset_env_ids = []

    def record_actuator_reset(env_ids=None) -> None:
        reset_env_ids.append(env_ids)
        actuator_reset(env_ids)

    monkeypatch.setattr(actuator, "reset", record_actuator_reset)
    composers = (floating.instantaneous_wrench_composer, floating.permanent_wrench_composer)
    ones = torch.ones((_NUM_ENVS, floating.num_bodies, 3), device=device)
    floating.permanent_wrench_composer.set_forces_and_torques_index(forces=ones, torques=ones)
    floating.instantaneous_wrench_composer.add_forces_and_torques_index(forces=ones, torques=ones)
    floating.reset(env_ids=torch.tensor([0], device=device))
    for composer in composers:
        assert composer.active
        for buffer in (composer.out_force_b.torch, composer.out_torque_b.torch):
            assert torch.count_nonzero(buffer[0]) == 0
            assert torch.count_nonzero(buffer[1:]) == buffer[1:].numel()
    reset_env_ids.clear()
    floating.reset()
    assert reset_env_ids == [None]
    for composer in composers:
        assert not composer.active
        assert torch.count_nonzero(composer.out_force_b.torch) == 0
        assert torch.count_nonzero(composer.out_torque_b.torch) == 0


def test_spatial_tendon_properties_round_trip(articulation_scene: _ArticulationScene) -> None:
    """A locally authored spatial tendon is discovered and its selected properties reach PhysX."""
    device = articulation_scene.device
    articulation = articulation_scene.tendon
    assert articulation.is_fixed_base
    assert articulation.num_spatial_tendons == 1
    root_view = articulation.root_view
    getters = {
        "stiffness": root_view.get_spatial_tendon_stiffnesses,
        "limit_stiffness": root_view.get_spatial_tendon_limit_stiffnesses,
        "damping": root_view.get_spatial_tendon_dampings,
        "offset": root_view.get_spatial_tendon_offsets,
    }

    # Distinct per-environment values so that an environment mix-up cannot pass
    env_offset = torch.arange(_NUM_ENVS, dtype=torch.float32, device=device).unsqueeze(1)
    values = {"stiffness": 10.0 + env_offset, "limit_stiffness": 20.0 + env_offset, "damping": 3.0 + env_offset}
    values["offset"] = env_offset
    for name, value in values.items():
        getattr(articulation, f"set_spatial_tendon_{name}_index")(**{name: value})
    articulation.write_spatial_tendon_properties_to_sim_index()
    for name, getter in getters.items():
        torch.testing.assert_close(wp.to_torch(getter()).to(device), values[name])

    # Partial writes change environment 1 in the data and in the solver and preserve environment 0.
    env_ids = torch.tensor([1], dtype=torch.int32, device=device)
    partial = {"stiffness": 12.0, "limit_stiffness": 3.0, "damping": 1.5, "offset": 0.1}
    for name, value in partial.items():
        getattr(articulation, f"set_spatial_tendon_{name}_index")(
            **{name: torch.tensor([[value]], device=device), "env_ids": env_ids}
        )
        values[name][1] = value
        torch.testing.assert_close(getattr(articulation.data, f"spatial_tendon_{name}").torch, values[name])
    articulation.write_spatial_tendon_properties_to_sim_index(env_ids=env_ids)
    for name, getter in getters.items():
        torch.testing.assert_close(wp.to_torch(getter()).to(device), values[name])

    # Unsorted int64 selectors write each row to the environment it names.
    env_ids = torch.tensor([1, 0], dtype=torch.int64, device=device)
    for name, value in partial.items():
        rows = torch.tensor([[value + 1.0], [value + 2.0]], device=device)
        getattr(articulation, f"set_spatial_tendon_{name}_index")(**{name: rows, "env_ids": env_ids})
        values[name][env_ids] = rows
        torch.testing.assert_close(getattr(articulation.data, f"spatial_tendon_{name}").torch, values[name])
    articulation.write_spatial_tendon_properties_to_sim_index(env_ids=env_ids)
    for name, getter in getters.items():
        torch.testing.assert_close(wp.to_torch(getter()).to(device), values[name])
