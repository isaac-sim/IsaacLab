# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass as lab_configclass
from isaaclab.utils.math import (
    apply_delta_pose,
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

from isaaclab_assets import FRANKA_PANDA_CFG, G1_29DOF_CFG  # isort:skip

pytestmark = pytest.mark.integration

_NUM_ENVS = 16
_ARM_JOINTS = ["panda_joint.*"]


def _build_targets(device: str) -> dict[str, torch.Tensor]:
    """Build the target sets tracked by the Franka tests, keyed by the controller command layout."""
    abs_pos_b = torch.tensor([[0.5, 0.5, 0.7], [0.5, -0.4, 0.6], [0.5, 0.0, 0.5]], device=device)
    abs_quat_b = torch.tensor([[0.0, 0.707, 0.0, 0.707], [0.707, 0.0, 0.0, 0.707], [1.0, 0.0, 0.0, 0.0]], device=device)
    rel_pos = torch.tensor([[0.2, 0.0, 0.0], [0.2, 0.2, 0.0], [0.2, 0.2, -0.2]], device=device)
    rel_axis_angle = torch.tensor(
        [[0.0, torch.pi / 2, 0.0], [torch.pi / 2, 0.0, 0.0], [torch.pi / 2, torch.pi / 2, 0.0]], device=device
    )
    abs_wrench_b = torch.tensor(
        [[0.0, 0.0, 10.0, 0.0, -1.0, 0.0], [0.0, 10.0, 0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        device=device,
    )
    kp = torch.tensor([[200.0], [240.0], [160.0]], device=device).repeat(1, 6)
    d_ratio = torch.tensor([[2.0], [2.2], [1.8]], device=device).repeat(1, 6)
    # [x, y, z, qx, qy, qz, qw, force_xyz, torque_xyz]
    hybrid_b = torch.tensor(
        [
            [0.6, 0.2, 0.5, 0.707, 0.0, 0.707, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, -0.29, 0.6, 0.707, 0.0, 0.707, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.1, 0.8, 0.5774, 0.0, 0.8165, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        device=device,
    )
    # pose in root frame, wrench in the (tilted) task frame
    tilted_pose_b = torch.tensor([[0.6, 0.15, 0.3], [0.6, -0.3, 0.3], [0.8, 0.0, 0.5]], device=device)
    tilted_quat = torch.tensor([[0.92387953, 0.0, 0.38268343, 0.0]], device=device).repeat(3, 1)
    tilted_wrench_task = torch.tensor([[0.0, 0.0, 10.0, 0.0, 0.0, 0.0]], device=device).repeat(3, 1)

    abs_pose_b = torch.cat([abs_pos_b, abs_quat_b], dim=-1)
    return {
        "abs_pose": abs_pose_b,
        "rel_pose": torch.cat([rel_pos, rel_axis_angle], dim=-1),
        "abs_wrench": abs_wrench_b,
        "abs_pose_variable": torch.cat([abs_pose_b, kp, d_ratio], dim=-1),
        "hybrid": hybrid_b,
        "hybrid_variable_kp": torch.cat([hybrid_b, kp], dim=-1),
        "hybrid_tilted": torch.cat([tilted_pose_b, tilted_quat, tilted_wrench_task], dim=-1),
    }


@pytest.fixture
def scene():
    """Create a simulation with a cloned, gravity-free Franka scene and the tracked target sets."""
    stage = sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    # TODO: Remove this once we have a better way to handle this.
    sim._app_control_on_stop_handle = None

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)
    light_cfg = sim_utils.DistantLightCfg(intensity=5.0, exposure=10.0)
    light_cfg.func("/Light", light_cfg, translation=[0, 0, 1])

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    env_fmt = "/World/envs/env_{}"
    env_origins, _ = cloner.grid_transforms(_NUM_ENVS, spacing=2.0)
    stage.DefinePrim(env_fmt.format(0), "Xform")
    cloner.usd_replicate(stage, [env_fmt.format(0)], [env_fmt], np.arange(_NUM_ENVS), positions=env_origins)

    robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Explicit torque actuators enforce effort limits on the commands sent to the simulator.
    for actuator_name in ("panda_shoulder", "panda_forearm"):
        actuator_cfg = robot_cfg.actuators[actuator_name]
        robot_cfg.actuators[actuator_name] = IdealPDActuatorCfg(
            joint_names_expr=actuator_cfg.joint_names_expr,
            joint_effort_limit=actuator_cfg.joint_effort_limit,
            joint_velocity_limit=actuator_cfg.joint_velocity_limit,
            armature=actuator_cfg.armature,
            stiffness=0.0,
            damping=0.0,
        )
    robot_cfg.spawn.rigid_props.disable_gravity = True

    yield SimpleNamespace(
        sim=sim,
        num_envs=_NUM_ENVS,
        robot_cfg=robot_cfg,
        ee_marker=ee_marker,
        goal_marker=goal_marker,
        targets=_build_targets(sim.device),
    )

    sim.stop()
    sim.clear_instance()


"""
Franka tracking cases.
"""

# obstacle specs: (size, translation, orientation)
_CONTACT_PLANES = [
    ((0.7, 0.7, 0.01), (0.2, 0.0, 0.93), (0.0, -0.1736, 0.0, 0.9848)),
    ((0.7, 0.7, 0.01), (0.2, 0.35, 0.7), (0.707, 0.0, 0.0, 0.707)),
    ((0.7, 0.7, 0.01), (0.55, 0.0, 0.7), (0.0, 0.707, 0.0, 0.707)),
]


def _hybrid_plane(targets):
    return [((1.0, 1.0, 0.01), (targets["hybrid"][0, 0] + 0.05, 0.0, 0.7), (0.0, 0.707, 0.0, 0.707))]


def _tilted_plane(targets):
    x = targets["hybrid_tilted"][0, 0] + 0.085
    return [((2.0, 1.5, 0.01), (x, 0.0, 0.3), (0.0, -0.3826834324, 0.0, 0.9238795325))]


@dataclass
class _Case:
    """One Franka operational-space tracking scenario."""

    osc: dict
    target: str
    ee_frame: str = "panda_hand"
    frame: str = "root"
    obstacles: Callable[[dict], list] | None = None
    contact_history: int = 2
    gravity: bool = False
    convergence_steps: int = 500
    rotation_tolerance: float = 0.1
    marks: list = field(default_factory=list)


_FIXED_DECOUPLED = dict(
    impedance_mode="fixed",
    inertial_dynamics_decoupling=True,
    partial_inertial_dynamics_decoupling=False,
    gravity_compensation=False,
    motion_stiffness_task=500.0,
    motion_damping_ratio_task=1.0,
)
_FIXED_UNDECOUPLED = dict(
    impedance_mode="fixed",
    inertial_dynamics_decoupling=False,
    gravity_compensation=False,
    motion_stiffness_task=[400.0, 400.0, 400.0, 100.0, 100.0, 100.0],
    motion_damping_ratio_task=[5.0, 5.0, 5.0, 0.001, 0.001, 0.001],
)
_FIXED_PARTIAL = dict(
    impedance_mode="fixed",
    inertial_dynamics_decoupling=True,
    partial_inertial_dynamics_decoupling=True,
    gravity_compensation=False,
    motion_stiffness_task=1000.0,
    motion_damping_ratio_task=1.0,
)
_HYBRID_TILTED = dict(
    target_types=["pose_abs", "wrench_abs"],
    impedance_mode="fixed",
    inertial_dynamics_decoupling=True,
    partial_inertial_dynamics_decoupling=False,
    gravity_compensation=False,
    motion_stiffness_task=400.0,
    motion_damping_ratio_task=1.0,
    contact_wrench_stiffness_task=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
    motion_control_axes_task=[1, 1, 0, 1, 1, 1],
    contact_wrench_control_axes_task=[0, 0, 1, 0, 0, 0],
)
_WRENCH_ONLY = dict(
    target_types=["wrench_abs"], motion_control_axes_task=[0, 0, 0, 0, 0, 0], contact_wrench_control_axes_task=[1] * 6
)

_CASES = {
    "pose_abs_without_inertial_decoupling": _Case(dict(target_types=["pose_abs"], **_FIXED_UNDECOUPLED), "abs_pose"),
    "pose_abs_with_partial_inertial_decoupling": _Case(
        dict(target_types=["pose_abs"], **_FIXED_PARTIAL), "abs_pose", rotation_tolerance=0.12
    ),
    "pose_abs_fixed_impedance_with_gravity_compensation": _Case(
        dict(
            target_types=["pose_abs"],
            **_FIXED_DECOUPLED | dict(motion_damping_ratio_task=2.0, gravity_compensation=True),
        ),
        "abs_pose",
        gravity=True,
    ),
    "pose_abs": _Case(dict(target_types=["pose_abs"], **_FIXED_DECOUPLED), "abs_pose"),
    "pose_rel": _Case(dict(target_types=["pose_rel"], **_FIXED_DECOUPLED), "rel_pose"),
    "pose_abs_variable_impedance": _Case(
        dict(
            target_types=["pose_abs"],
            impedance_mode="variable",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=False,
        ),
        "abs_pose_variable",
    ),
    "wrench_abs_open_loop": _Case(
        _WRENCH_ONLY, "abs_wrench", obstacles=lambda targets: _CONTACT_PLANES, contact_history=50
    ),
    "wrench_abs_closed_loop": _Case(
        # zero torque feedback as the contact torque cannot be measured
        dict(_WRENCH_ONLY, contact_wrench_stiffness_task=[0.2, 0.2, 0.2, 0.0, 0.0, 0.0]),
        "abs_wrench",
        obstacles=lambda targets: _CONTACT_PLANES,
    ),
    "hybrid_decoupled_motion": _Case(
        dict(
            target_types=["pose_abs", "wrench_abs"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=True,
            gravity_compensation=False,
            motion_stiffness_task=300.0,
            motion_damping_ratio_task=1.0,
            contact_wrench_stiffness_task=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            motion_control_axes_task=[0, 1, 1, 1, 1, 1],
            contact_wrench_control_axes_task=[1, 0, 0, 0, 0, 0],
        ),
        "hybrid",
        ee_frame="panda_leftfinger",
        obstacles=_hybrid_plane,
    ),
    "hybrid_variable_kp_impedance": _Case(
        dict(
            target_types=["pose_abs", "wrench_abs"],
            impedance_mode="variable_kp",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=False,
            motion_damping_ratio_task=0.8,
            contact_wrench_stiffness_task=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            motion_control_axes_task=[0, 1, 1, 1, 1, 1],
            contact_wrench_control_axes_task=[1, 0, 0, 0, 0, 0],
        ),
        "hybrid_variable_kp",
        ee_frame="panda_leftfinger",
        obstacles=_hybrid_plane,
        # hybrid control is less precise, so allow more steps to converge
        convergence_steps=750,
        marks=[pytest.mark.flaky(max_runs=3, min_passes=1)],
    ),
    "taskframe_pose_abs": _Case(dict(target_types=["pose_abs"], **_FIXED_DECOUPLED), "abs_pose", frame="task"),
    "taskframe_pose_rel": _Case(dict(target_types=["pose_rel"], **_FIXED_DECOUPLED), "rel_pose", frame="task"),
    "taskframe_hybrid": _Case(
        _HYBRID_TILTED, "hybrid_tilted", ee_frame="panda_leftfinger", frame="task", obstacles=_tilted_plane
    ),
    "pose_abs_without_inertial_decoupling_with_nullspace_centering": _Case(
        dict(target_types=["pose_abs"], **_FIXED_UNDECOUPLED, nullspace_control="position"), "abs_pose"
    ),
    "pose_abs_with_partial_inertial_decoupling_nullspace_centering": _Case(
        dict(target_types=["pose_abs"], **_FIXED_PARTIAL, nullspace_control="position", nullspace_stiffness=1.0),
        "abs_pose",
        rotation_tolerance=0.12,
    ),
    "pose_abs_with_nullspace_centering": _Case(
        dict(target_types=["pose_abs"], **_FIXED_DECOUPLED, nullspace_control="position", nullspace_stiffness=1.0),
        "abs_pose",
    ),
    "taskframe_hybrid_with_nullspace_centering": _Case(
        dict(_HYBRID_TILTED, nullspace_control="position"),
        "hybrid_tilted",
        ee_frame="panda_leftfinger",
        frame="task",
        obstacles=_tilted_plane,
    ),
}


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("case", [pytest.param(case, id=name, marks=case.marks) for name, case in _CASES.items()])
def test_franka_tracking(scene, case: _Case):
    """The controller drives the Franka end-effector to each target of the case within tolerance."""
    scene.robot_cfg.spawn.rigid_props.disable_gravity = not case.gravity
    robot = Articulation(cfg=scene.robot_cfg)
    contact_forces = None
    if case.obstacles is not None:
        for index, (size, translation, orientation) in enumerate(case.obstacles(scene.targets), start=1):
            obstacle_cfg = sim_utils.CuboidCfg(
                size=size,
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.1),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                activate_contact_sensors=True,
            )
            obstacle_cfg.func(
                f"/World/envs/env_[^/]+/obstacle{index}", obstacle_cfg, translation=translation, orientation=orientation
            )
        contact_forces = ContactSensor(
            ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/obstacle[^/]*",
                update_period=0.0,
                history_length=case.contact_history,
                debug_vis=False,
                force_threshold=0.1,
            )
        )
    osc = OperationalSpaceController(
        OperationalSpaceControllerCfg(**case.osc), num_envs=scene.num_envs, device=scene.sim.device
    )

    _run_op_space_controller(
        scene,
        robot,
        osc,
        case.ee_frame,
        scene.targets[case.target],
        contact_forces,
        case.frame,
        convergence_steps=case.convergence_steps,
        rotation_tolerance=case.rotation_tolerance,
    )


@pytest.mark.isaacsim_ci
def test_task_frame_conversion_preserves_absolute_target():
    """A rounded pose command must resolve to the same target through either reference frame."""
    osc_cfg = OperationalSpaceControllerCfg(target_types=["pose_abs"])
    osc = OperationalSpaceController(osc_cfg, num_envs=1, device="cpu")
    target_b = torch.tensor([[0.5, -0.4, 0.6, 0.707, 0.0, 0.0, 0.707]])
    command = target_b.clone()
    resolved_targets = []
    for frame in ("root", "task"):
        converted_command, task_frame_pose_b = _convert_to_task_frame(osc, command, target_b, frame)
        osc.set_command(converted_command, current_task_frame_pose_b=task_frame_pose_b)
        resolved_targets.append(osc.desired_ee_pose_b.clone())

    torch.testing.assert_close(resolved_targets[0], resolved_targets[1], atol=1e-6, rtol=0.0)
    torch.testing.assert_close(command, target_b, atol=0.0, rtol=0.0)


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("feedback_source", ["test_helper", "action"])
def test_franka_velocity_feedback_matches_jacobian(scene, feedback_source):
    """Both OSC callers must measure velocity at the link origin used by the Jacobian."""
    sim = scene.sim
    robot = Articulation(cfg=scene.robot_cfg)
    sim.reset()
    arm_joint_ids, _ = robot.find_joints(_ARM_JOINTS)
    ee_frame_idx = robot.find_bodies("panda_hand")[0][0]

    joint_vel = torch.zeros_like(robot.data.default_joint_vel.torch)
    joint_vel[:, arm_joint_ids] = torch.linspace(0.1, 0.7, len(arm_joint_ids), device=sim.device)
    robot.write_joint_state_to_sim_index(position=robot.data.default_joint_pos.torch, velocity=joint_vel)
    sim.step(render=False)
    robot.update(sim.get_physics_dt())

    # Angular motion and the hand's COM offset must expose the reference-point mismatch.
    assert not torch.allclose(
        robot.data.body_com_vel_w.torch[:, ee_frame_idx, :3],
        robot.data.body_link_vel_w.torch[:, ee_frame_idx, :3],
        atol=1e-4,
        rtol=1e-4,
    )
    if feedback_source == "test_helper":
        states = _update_states(robot, ee_frame_idx, arm_joint_ids, sim, None, scene.num_envs)
        jacobian_b, _, _, _, ee_vel_b, _, _, _, _, joint_vel = states
    else:
        env = SimpleNamespace(scene={"robot": robot}, sim=sim, num_envs=scene.num_envs, device=sim.device)
        action_cfg = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=_ARM_JOINTS,
            body_name="panda_hand",
            controller_cfg=OperationalSpaceControllerCfg(target_types=["pose_abs"]),
        )
        action_term = OperationalSpaceControllerAction(action_cfg, env)
        action_term._compute_ee_jacobian()
        action_term._compute_ee_velocity()
        jacobian_b, ee_vel_b = action_term._jacobian_b, action_term._ee_vel_b
        joint_vel = robot.data.joint_vel.torch[:, arm_joint_ids]

    # With a stationary fixed base, the link twist must equal J(q) * q_dot.
    expected_vel_b = torch.bmm(jacobian_b, joint_vel.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(ee_vel_b, expected_vel_b, atol=1e-4, rtol=1e-4)


"""
Floating-base regression test configs (PR #5107).
"""

_G1_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
]


@lab_configclass
class _FloatingBaseOscSceneCfg(InteractiveSceneCfg):
    """Minimal scene with a floating-base G1 humanoid."""

    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)
    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.fix_root_link = False
        self.robot.spawn.rigid_props.disable_gravity = True


@lab_configclass
class _FloatingBaseOscActionsCfg:
    arm_action: OperationalSpaceControllerActionCfg = OperationalSpaceControllerActionCfg(
        asset_name="robot",
        joint_names=_G1_ARM_JOINT_NAMES,
        body_name="left_elbow_link",
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="fixed",
            # Both flags enabled so the action term fetches mass matrix AND gravity each step,
            # exercising the floating-base +6 indexing on both quantities.
            inertial_dynamics_decoupling=True,
            gravity_compensation=True,
            motion_stiffness_task=500.0,
            motion_damping_ratio_task=1.0,
        ),
    )


@lab_configclass
class _FloatingBaseOscObsCfg:
    @lab_configclass
    class _PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})

    policy: _PolicyCfg = _PolicyCfg()


@lab_configclass
class _FloatingBaseOscEnvCfg(ManagerBasedEnvCfg):
    scene: _FloatingBaseOscSceneCfg = _FloatingBaseOscSceneCfg(num_envs=4, env_spacing=4.0)
    actions: _FloatingBaseOscActionsCfg = _FloatingBaseOscActionsCfg()
    observations: _FloatingBaseOscObsCfg = _FloatingBaseOscObsCfg()
    decimation: int = 1
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=0.01)


@pytest.mark.isaacsim_ci
def test_floating_base_osc_action_term_indexing():
    """Regression test for #4999 / PR #5107: the action term must offset the mass-matrix and gravity
    indices by the floating-base DoFs (``_jacobi_joint_idx``) instead of using the raw ``_joint_ids``."""
    env_cfg = _FloatingBaseOscEnvCfg()
    env_cfg.sim.device = "cuda:0"
    env = ManagerBasedEnv(cfg=env_cfg)

    try:
        robot: Articulation = env.scene["robot"]
        assert not robot.is_fixed_base, "G1_29DOF_CFG must be floating-base for this test"
        action_term = env.action_manager._terms["arm_action"]
        num_arm_joints = action_term._num_DoF

        # step the action term so its dynamic quantities are populated
        action_term.process_actions(torch.zeros(env.num_envs, action_term.action_dim, device=env.device))
        action_term.apply_actions()
        term_mass = action_term._mass_matrix.clone()
        term_gravity = action_term._gravity.clone()

        # the extracted quantities must match a manual extraction with the base-offset indices
        jacobi_joint_idx = action_term._jacobi_joint_idx
        full_mass_matrix = robot.data.mass_matrix.torch
        full_gravity = robot.data.gravity_compensation_forces.torch
        torch.testing.assert_close(term_mass, full_mass_matrix[:, jacobi_joint_idx, :][:, :, jacobi_joint_idx])
        torch.testing.assert_close(term_gravity, full_gravity[:, jacobi_joint_idx])
        assert term_mass.shape == (env.num_envs, num_arm_joints, num_arm_joints)
        assert term_gravity.shape == (env.num_envs, num_arm_joints)

        # the data-layer tensor exposes the full DoF axis and differs from the raw joint-id extraction (old bug)
        assert full_mass_matrix.shape[1] == robot.num_joints + robot.num_base_dofs
        original_joint_ids, _ = robot.find_joints(_G1_ARM_JOINT_NAMES)
        buggy_mass = full_mass_matrix[:, original_joint_ids, :][:, :, original_joint_ids]
        assert not torch.allclose(term_mass, buggy_mass, atol=1e-6)

        # physically reasonable, symmetric arm inertia
        diag = torch.diagonal(term_mass, dim1=-2, dim2=-1)
        assert (diag > 0).all()
        assert diag.max().item() < 100.0, "mass matrix diagonal too large, possibly contaminated by base DOFs"
        torch.testing.assert_close(term_mass, term_mass.mT, atol=1e-5, rtol=0.0)
    finally:
        env.close()


"""
Helpers.
"""


def _run_op_space_controller(
    scene: SimpleNamespace,
    robot: Articulation,
    osc: OperationalSpaceController,
    ee_frame_name: str,
    target_set: torch.Tensor,
    contact_forces: ContactSensor | None,
    frame: str,
    convergence_steps: int = 500,
    position_tolerance: float = 0.1,
    rotation_tolerance: float = 0.1,
):
    """Track every target of ``target_set`` for ``convergence_steps`` steps and check convergence."""
    sim, num_envs = scene.sim, scene.num_envs
    # masks for evaluating target convergence according to the selection matrices
    pos_mask = torch.tensor(osc.cfg.motion_control_axes_task[:3], device=sim.device).view(1, 3)
    rot_mask = torch.tensor(osc.cfg.motion_control_axes_task[3:], device=sim.device).view(1, 3)
    # only the force components can be measured
    force_mask = torch.tensor(osc.cfg.contact_wrench_control_axes_task[:3], device=sim.device).view(1, 3)

    sim_dt = sim.get_physics_dt()
    sim.reset()
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    arm_joint_ids = robot.find_joints(_ARM_JOINTS)[0]
    # buffers must be updated before the first controller step
    robot.update(dt=sim_dt)
    joint_centers = torch.mean(robot.data.soft_joint_pos_limits.torch[:, arm_joint_ids, :], dim=-1)

    current_goal_idx = 0
    command = torch.zeros(num_envs, osc.action_dim, device=sim.device)
    ee_target_pose_b = torch.zeros(num_envs, 7, device=sim.device)
    ee_target_pose_w = torch.zeros(num_envs, 7, device=sim.device)
    zero_joint_efforts = torch.zeros(num_envs, robot.num_joints, device=sim.device)

    (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        ee_force_b,
        joint_pos,
        joint_vel,
    ) = _update_states(robot, ee_frame_idx, arm_joint_ids, sim, contact_forces, num_envs)

    # run for 3 target cycles plus 1 step to trigger the final convergence check
    for count in range(3 * convergence_steps + 1):
        if count % convergence_steps == 0:
            if count > 0:
                _check_convergence(
                    osc,
                    ee_pose_b,
                    ee_target_pose_b,
                    ee_force_b,
                    command,
                    pos_mask,
                    rot_mask,
                    force_mask,
                    frame,
                    position_tolerance,
                    rotation_tolerance,
                )
            # reset the joint state to the defaults with zero torques
            robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
            robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
            robot.set_joint_effort_target_index(target=zero_joint_efforts)
            robot.write_data_to_sim()
            robot.reset()
            if contact_forces is not None:
                contact_forces.reset()
            # at reset, the jacobians are not updated to the latest state; only the pose is needed
            robot.update(sim_dt)
            ee_pose_b = _update_states(robot, ee_frame_idx, arm_joint_ids, sim, contact_forces, num_envs)[3]
            command, ee_target_pose_b, ee_target_pose_w, current_goal_idx = _update_target(
                osc, root_pose_w, ee_pose_b, target_set, current_goal_idx
            )
            osc.reset()
            command, task_frame_pose_b = _convert_to_task_frame(osc, command, ee_target_pose_b, frame)
            osc.set_command(command=command, current_ee_pose_b=ee_pose_b, current_task_frame_pose_b=task_frame_pose_b)
        else:
            (
                jacobian_b,
                mass_matrix,
                gravity,
                ee_pose_b,
                ee_vel_b,
                root_pose_w,
                ee_pose_w,
                ee_force_b,
                joint_pos,
                joint_vel,
            ) = _update_states(robot, ee_frame_idx, arm_joint_ids, sim, contact_forces, num_envs)
            joint_efforts = osc.compute(
                jacobian_b=jacobian_b,
                current_ee_pose_b=ee_pose_b,
                current_ee_vel_b=ee_vel_b,
                current_ee_force_b=ee_force_b,
                mass_matrix=mass_matrix,
                gravity=gravity,
                current_joint_pos=joint_pos,
                current_joint_vel=joint_vel,
                nullspace_joint_pos_target=joint_centers,
            )
            robot.set_joint_effort_target_index(target=joint_efforts, joint_ids=arm_joint_ids)
            robot.write_data_to_sim()

        scene.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        scene.goal_marker.visualize(ee_target_pose_w[:, 0:3], ee_target_pose_w[:, 3:7])
        sim.step(render=False)
        robot.update(sim_dt)


def _update_states(
    robot: Articulation,
    ee_frame_idx: int,
    arm_joint_ids: list[int],
    sim: sim_utils.SimulationContext,
    contact_forces: ContactSensor | None,
    num_envs: int,
):
    """Read the robot quantities consumed by the operational space controller, expressed in the root frame.

    Returns:
        ``(jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, root_pose_w, ee_pose_w, ee_force_b, joint_pos,
        joint_vel)``.
    """
    ee_jacobi_idx = ee_frame_idx - 1
    jacobian_w = robot.data.body_link_jacobian_w.torch[:, ee_jacobi_idx, :, arm_joint_ids]
    mass_matrix = robot.data.mass_matrix.torch[:, arm_joint_ids, :][:, :, arm_joint_ids]
    gravity = robot.data.gravity_compensation_forces.torch[:, arm_joint_ids]
    # Convert the Jacobian from world to root frame
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w.torch))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    root_pose_w = robot.data.root_pose_w.torch
    ee_pose_w = robot.data.body_pose_w.torch[:, ee_frame_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    # Match the link-origin reference point used by the pose and Jacobian.
    relative_vel_w = robot.data.body_link_vel_w.torch[:, ee_frame_idx, :] - robot.data.root_link_vel_w.torch
    ee_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w.torch, relative_vel_w[:, 0:3])
    ee_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w.torch, relative_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    # Contact force: average the history to smooth, then take the max over the surfaces since only
    # one is the contact of interest. Using the world-frame force is a simplification for testing.
    ee_force_b = torch.zeros(num_envs, 3, device=sim.device)
    if contact_forces is not None:
        contact_forces.update(sim.get_physics_dt())
        ee_force_b, _ = torch.max(torch.mean(contact_forces.data.net_normal_forces_w_history.torch, dim=1), dim=1)

    joint_pos = robot.data.joint_pos.torch[:, arm_joint_ids]
    joint_vel = robot.data.joint_vel.torch[:, arm_joint_ids]

    return (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        ee_force_b,
        joint_pos,
        joint_vel,
    )


def _update_target(
    osc: OperationalSpaceController,
    root_pose_w: torch.Tensor,
    ee_pose_b: torch.Tensor,
    target_set: torch.Tensor,
    current_goal_idx: int,
):
    """Select the next target and return ``(command, ee_target_pose_b, ee_target_pose_w, next_goal_idx)``."""
    command = target_set[current_goal_idx].expand(osc.num_envs, -1).clone()

    ee_target_pose_b = torch.zeros(osc.num_envs, 7, device=osc._device)
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            ee_target_pose_b[:] = command[:, :7]
        elif target_type == "pose_rel":
            ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7] = apply_delta_pose(
                ee_pose_b[:, :3], ee_pose_b[:, 3:], command[:, :7]
            )
        elif target_type == "wrench_abs":
            pass  # the target pose may stay at the root frame for force control
        else:
            raise ValueError("Undefined target_type within _update_target().")

    # target pose in the world frame (for the marker)
    ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
    )
    ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)

    return command, ee_target_pose_b, ee_target_pose_w, (current_goal_idx + 1) % len(target_set)


def _convert_to_task_frame(
    osc: OperationalSpaceController, command: torch.Tensor, ee_target_pose_b: torch.Tensor, frame: str
):
    """Express the target command in the task frame when ``frame == "task"``.

    Returns:
        The converted command and the task frame pose in the root frame (None for the root frame).
    """
    command = command.clone()
    if frame == "root":
        return command, None
    if frame != "task":
        raise ValueError("Invalid frame selection for target setting inside the test_operational_space.")

    task_frame_pose_b = ee_target_pose_b.clone()
    # Rounded goal quaternions must define a unit rotation when used as a reference frame.
    task_frame_pose_b[:, 3:] /= torch.linalg.vector_norm(task_frame_pose_b[:, 3:], dim=-1, keepdim=True)
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            command[:, :3], command[:, 3:7] = subtract_frame_transforms(
                task_frame_pose_b[:, :3], task_frame_pose_b[:, 3:], command[:, :3], command[:, 3:7]
            )
        elif target_type == "pose_rel":
            # rotate the delta position and orientation from the base to the task frame
            R_b_task = matrix_from_quat(task_frame_pose_b[:, 3:]).mT
            command[:, :3] = (R_b_task @ command[:, :3].unsqueeze(-1)).squeeze(-1)
            command[:, 3:7] = (R_b_task @ command[:, 3:7].unsqueeze(-1)).squeeze(-1)
        elif target_type == "wrench_abs":
            pass  # the wrench targets of the tilted set are already defined in the task frame
        else:
            raise ValueError("Undefined target_type within _convert_to_task_frame().")

    return command, task_frame_pose_b


def _check_convergence(
    osc: OperationalSpaceController,
    ee_pose_b: torch.Tensor,
    ee_target_pose_b: torch.Tensor,
    ee_force_b: torch.Tensor,
    ee_target_b: torch.Tensor,
    pos_mask: torch.Tensor,
    rot_mask: torch.Tensor,
    force_mask: torch.Tensor,
    frame: str,
    position_tolerance: float,
    rotation_tolerance: float,
):
    """Assert that the controlled pose axes and measured forces reached their targets."""
    cmd_idx = 0
    for target_type in osc.cfg.target_types:
        if target_type in ("pose_abs", "pose_rel"):
            pos_error, rot_error = compute_pose_error(
                ee_pose_b[:, 0:3], ee_pose_b[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
            )
            pos_error_norm = torch.linalg.norm(pos_error * pos_mask, dim=-1)
            rot_error_norm = torch.linalg.norm(rot_error * rot_mask, dim=-1)
            des_error = torch.zeros_like(pos_error_norm)
            torch.testing.assert_close(pos_error_norm, des_error, rtol=0.0, atol=position_tolerance)
            torch.testing.assert_close(rot_error_norm, des_error, rtol=0.0, atol=rotation_tolerance)
            cmd_idx += 7 if target_type == "pose_abs" else 6
        elif target_type == "wrench_abs":
            force_target_b = ee_target_b[:, cmd_idx : cmd_idx + 3].clone()
            if frame == "task":
                R_task_b = matrix_from_quat(ee_target_pose_b[:, 3:])
                force_target_b = (R_task_b @ force_target_b.unsqueeze(-1)).squeeze(-1)
            force_error_norm = torch.linalg.norm((ee_force_b - force_target_b) * force_mask, dim=-1)
            # Contact force steady-state is sensitive to physics engine internals, which causes outlier
            # environments. A tight median check catches controller regressions while a loose max check
            # catches catastrophic failures without breaking on single-environment noise.
            median_error = torch.median(force_error_norm).item()
            max_error = torch.max(force_error_norm).item()
            assert median_error < 5.0, (
                f"Median force error {median_error:.1f} N exceeds 5.0 N threshold"
                f" (max: {max_error:.1f} N, per-env: {force_error_norm.tolist()})"
            )
            assert max_error < 50.0, (
                f"Max force error {max_error:.1f} N exceeds 50.0 N sanity threshold"
                f" (median: {median_error:.1f} N, per-env: {force_error_norm.tolist()})"
            )
            cmd_idx += 6
        else:
            raise ValueError("Undefined target_type within _check_convergence().")
