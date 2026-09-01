# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import tempfile

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import DampingTaskCfg, FrameTaskCfg, NullSpacePostureTaskCfg, PinkIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg, preset

from . import mdp

from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG  # isort: skip
from isaaclab_teleop.haptic_feedback import GloveHapticFeedbackCfg  # isort: skip
from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg, XrCameraFeedCfg  # isort: skip
from isaaclab_teleop.xr_cfg import XrCfg  # isort: skip
from isaaclab_tasks.contrib.robot_pov_camera_cfg import robot_pov_camera_cfg  # isort: skip


def _build_gr1t2_pickplace_pipeline():
    """Build an IsaacTeleop retargeting pipeline for GR1T2 pick-place teleoperation.

    Creates two Se3AbsRetargeters for left and right wrist pose tracking and
    two DexHandRetargeters for left and right dexterous hand finger control
    from hand tracking data. All outputs are flattened into a single action
    tensor via TensorReorderer.
    """
    from isaacteleop.retargeters import (
        DexHandRetargeter,
        DexHandRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    # Create input sources (trackers are auto-discovered from pipeline)
    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")

    # External input: world-to-anchor 4x4 transform matrix provided by IsaacTeleopDevice
    transform_input = ValueInput("world_T_anchor", TransformMatrix())

    # Apply the coordinate-frame transform to controller poses so that
    # downstream retargeters receive data in the simulation world frame.
    _transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    # -------------------------------------------------------------------------
    # SE3 Absolute Pose Retargeters (left and right wrists)
    # -------------------------------------------------------------------------
    # Left wrist: identity rotation offset (passes through as-is in original retargeter)
    left_se3_cfg = Se3RetargeterConfig(
        input_device=HandsSource.LEFT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        target_offset_roll=0.0,
        target_offset_pitch=0.0,
        target_offset_yaw=0.0,
    )
    left_se3 = Se3AbsRetargeter(left_se3_cfg, name="left_ee_pose")
    connected_left_se3 = left_se3.connect(
        {
            HandsSource.LEFT: transformed_hands.output(HandsSource.LEFT),
        }
    )

    # Right wrist: 180-degree Z rotation offset
    # From GR1T2Retargeter._retarget_abs: the USD control frame is 180 degrees
    # rotated around the Z axis w.r.t. the OpenXR frame.
    right_se3_cfg = Se3RetargeterConfig(
        input_device=HandsSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        target_offset_roll=0.0,
        target_offset_pitch=0.0,
        target_offset_yaw=180.0,
    )
    right_se3 = Se3AbsRetargeter(right_se3_cfg, name="right_ee_pose")
    connected_right_se3 = right_se3.connect(
        {
            HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT),
        }
    )

    # -------------------------------------------------------------------------
    # DexHand Retargeters (left and right hands)
    # -------------------------------------------------------------------------
    # Resolve dex-retargeting YAML config paths from IsaacLab's retargeter data directory
    import isaaclab_teleop.isaac_teleop_cfg as _teleop_cfg_mod

    _teleop_cfg_file = _teleop_cfg_mod.__file__
    if _teleop_cfg_file is None:
        raise RuntimeError("Could not resolve isaaclab_teleop package path for dex-retargeting configs.")
    _teleop_pkg_dir = os.path.dirname(_teleop_cfg_file)
    _data_dir = os.path.join(
        _teleop_pkg_dir,
        "deprecated",
        "openxr",
        "retargeters",
        "humanoid",
        "fourier",
        "data",
    )
    _config_dir = os.path.join(_data_dir, "configs", "dex-retargeting")
    left_yaml_path = os.path.join(_config_dir, "fourier_hand_left_dexpilot.yml")
    right_yaml_path = os.path.join(_config_dir, "fourier_hand_right_dexpilot.yml")

    # Resolve URDF paths (downloads from Omniverse if needed)
    local_left_urdf = retrieve_file_path(f"{ISAACLAB_NUCLEUS_DIR}/Mimic/GR1T2_assets/GR1_T2_left_hand.urdf")
    local_right_urdf = retrieve_file_path(f"{ISAACLAB_NUCLEUS_DIR}/Mimic/GR1T2_assets/GR1_T2_right_hand.urdf")

    # Hand-tracking to base-link frame transform (OPERATOR2MANO matrix)
    # From gr1_t2_dex_retargeting_utils: [[0,-1,0],[-1,0,0],[0,0,-1]]
    operator2mano = (0, -1, 0, -1, 0, 0, 0, 0, -1)

    # Joint names for each hand (11 DOF per hand)
    left_hand_joint_names = [
        "L_index_proximal_joint",
        "L_index_intermediate_joint",
        "L_middle_proximal_joint",
        "L_middle_intermediate_joint",
        "L_pinky_proximal_joint",
        "L_pinky_intermediate_joint",
        "L_ring_proximal_joint",
        "L_ring_intermediate_joint",
        "L_thumb_proximal_yaw_joint",
        "L_thumb_proximal_pitch_joint",
        "L_thumb_distal_joint",
    ]

    right_hand_joint_names = [
        "R_index_proximal_joint",
        "R_index_intermediate_joint",
        "R_middle_proximal_joint",
        "R_middle_intermediate_joint",
        "R_pinky_proximal_joint",
        "R_pinky_intermediate_joint",
        "R_ring_proximal_joint",
        "R_ring_intermediate_joint",
        "R_thumb_proximal_yaw_joint",
        "R_thumb_proximal_pitch_joint",
        "R_thumb_distal_joint",
    ]

    left_dex_cfg = DexHandRetargeterConfig(
        hand_retargeting_config=left_yaml_path,
        hand_urdf=local_left_urdf,
        hand_joint_names=left_hand_joint_names,
        hand_side="left",
        handtracking_to_baselink_frame_transform=operator2mano,
    )
    left_dex = DexHandRetargeter(left_dex_cfg, name="left_hand")
    connected_left_dex = left_dex.connect(
        {
            HandsSource.LEFT: hands.output(HandsSource.LEFT),
        }
    )

    right_dex_cfg = DexHandRetargeterConfig(
        hand_retargeting_config=right_yaml_path,
        hand_urdf=local_right_urdf,
        hand_joint_names=right_hand_joint_names,
        hand_side="right",
        handtracking_to_baselink_frame_transform=operator2mano,
    )
    right_dex = DexHandRetargeter(right_dex_cfg, name="right_hand")
    connected_right_dex = right_dex.connect(
        {
            HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
        }
    )

    # -------------------------------------------------------------------------
    # TensorReorderer: flatten into a 36D action tensor
    # -------------------------------------------------------------------------
    # Se3AbsRetargeter outputs 7D arrays: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
    left_ee_elements = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w"]
    right_ee_elements = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w"]

    # Output order must match the PinkInverseKinematicsActionCfg expected tensor layout:
    #   [left_wrist(7), right_wrist(7), hand_joints(22)]
    # Hand joints follow hand_joint_names order from ActionsCfg.upper_body_ik.
    output_order = (
        left_ee_elements
        + right_ee_elements
        + [
            # hand_joint_names indices 0-4 (left proximal + thumb yaw)
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_thumb_proximal_yaw_joint",
            # hand_joint_names indices 5-9 (right proximal + thumb yaw)
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            # hand_joint_names indices 10-14 (left intermediate + thumb pitch)
            "L_index_intermediate_joint",
            "L_middle_intermediate_joint",
            "L_pinky_intermediate_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_pitch_joint",
            # hand_joint_names indices 15-19 (right intermediate + thumb pitch)
            "R_index_intermediate_joint",
            "R_middle_intermediate_joint",
            "R_pinky_intermediate_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_pitch_joint",
            # hand_joint_names indices 20-21 (thumb distal)
            "L_thumb_distal_joint",
            "R_thumb_distal_joint",
        ]
    )

    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": left_ee_elements,
            "right_ee_pose": right_ee_elements,
            "left_hand_joints": left_hand_joint_names,
            "right_hand_joints": right_hand_joint_names,
        },
        output_order=output_order,
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_hand_joints": "scalar",
            "right_hand_joints": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "left_ee_pose": connected_left_se3.output("ee_pose"),
            "right_ee_pose": connected_right_se3.output("ee_pose"),
            "left_hand_joints": connected_left_dex.output("hand_joints"),
            "right_hand_joints": connected_right_dex.output("hand_joints"),
        }
    )

    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    return pipeline, [left_dex, right_dex]


##
# Scene definition
##

# The steering wheel USD authors its rigid body on a nested prim rather than at the
# spawned ``Object`` root, so contact filtering must target that actor: filtering
# against ``Object`` matches an empty Xform and force_matrix_w always reads zero.
_STEERING_WHEEL_BODY = "{ENV_REGEX_NS}/Object/Geometry/sm_steeringwheel_a01_01"


# Steering-wheel collision meshes that must keep their convex decomposition under MJWarp.
# The rim is the tube the hands actually grasp and the spokes bound the gap the fingers wrap
# through, so collapsing either into a single hull would fill the wheel and make it ungraspable.
_STEERING_WHEEL_DECOMPOSED_MESHES = frozenset(
    {
        "sm_steeringwheel_a01_wheel_rim_01",
        "sm_steeringwheel_a01_wheel_spokes_01",
    }
)


def _spawn_steering_wheel_for_mjwarp(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the steering wheel with an MJWarp-compatible collision approximation.

    Every collision mesh in the asset is authored as ``convexDecomposition``. MuJoCo derives
    each convex piece's inertia from its volume and rejects the model outright when a piece is
    degenerate (``mesh volume is too small``); the wheel's hub and quick-release parts decompose
    into such slivers, which makes the MJWarp model fail to compile.

    Collapsing every mesh to a single convex hull compiles but turns the wheel into a solid disc.
    Instead only the non-graspable detail meshes are hulled, and the meshes in
    :data:`_STEERING_WHEEL_DECOMPOSED_MESHES` keep their decomposition so the wheel stays
    graspable. PhysX is unaffected: this spawner is only used by the ``newton_mjwarp`` preset.

    Args:
        prim_path: The prim path or regex pattern to spawn the asset at.
        cfg: The spawner configuration.
        translation: Optional translation applied to the spawned prim.
        orientation: Optional ``(x, y, z, w)`` orientation applied to the spawned prim.
        **kwargs: Forwarded to :func:`~isaaclab.sim.spawn_from_usd`.

    Returns:
        The spawned source prim.
    """
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)

    stage = sim_utils.get_current_stage()
    for root_path in sim_utils.find_matching_prim_paths(prim_path):
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            continue
        for mesh_prim in Usd.PrimRange(root):
            if not mesh_prim.IsA(UsdGeom.Mesh) or mesh_prim.GetName() in _STEERING_WHEEL_DECOMPOSED_MESHES:
                continue
            approximation = mesh_prim.GetAttribute("physics:approximation")
            if approximation and approximation.Get() is not None:
                approximation.Set("convexHull")
    return prim


def _gr1t2_robot_spawn() -> UsdFileCfg:
    """Build the GR1T2 spawn with backend-appropriate gravity handling.

    ``GR1T2_HIGH_PD_CFG`` disables gravity on every body, which is what lets this task actuate
    only the trunk, arms and hands. Newton does not read ``physxRigidBody:disableGravity`` off a
    rigid body -- it only honours the flag on the physics scene -- so under MJWarp the unactuated
    legs and head would be pulled by gravity. ``mjc:gravcomp`` is the MuJoCo-solver equivalent, so
    the ``newton_mjwarp`` preset asks for full per-body gravity compensation instead.

    The spawn is copied so the per-backend ``rigid_props`` override stays local to this task and
    does not leak into the shared :data:`GR1T2_HIGH_PD_CFG`. Only ``rigid_props`` is preset-backed:
    ``__post_init__`` reads ``scene.robot.spawn.usd_path`` before presets are resolved, so ``spawn``
    itself has to stay a concrete config.

    Returns:
        The spawn configuration for the task's GR1T2 robot.
    """
    spawn = GR1T2_HIGH_PD_CFG.spawn.copy()
    spawn.rigid_props = preset(
        default=GR1T2_HIGH_PD_CFG.spawn.rigid_props,
        # The remaining PhysX damping and velocity-limit fields are not consumed by Newton.
        newton_mjwarp=sim_utils.MujocoRigidBodyPropertiesCfg(disable_gravity=True, gravcomp=1.0),
    )
    return spawn


# Finger drive gains for MJWarp.
#
# ``GR1T2_HIGH_PD_CFG`` leaves the hands' stiffness and damping as ``None``, i.e. "use whatever
# the USD authors". PhysX resolves that to 17184 / 558.48 and closes a finger onto its target in
# five steps. Newton does not pick those drives up, and a moving finger target then sends the
# articulation to NaN on the first step, so the gains have to be authored here.
#
# They cannot simply be copied across. MJWarp will not run 17184: it goes non-finite with 2, 8
# and 16 substeps, with ``implicitfast`` and ``implicit``, and with the ``newton`` and ``cg``
# solvers.
#
# The value is chosen to reproduce PhysX's *approach*, not just to reach the target. Actions
# arrive at 20 Hz (``decimation=6`` at 120 Hz) while the scene renders every second physics
# step, so a drive stiff enough to snap onto its target within one step is rendered as a hold
# followed by a jump -- the fingers visibly move in stairsteps. 3000 did exactly that. At 200
# the joint approaches over roughly five steps, matching PhysX almost sample for sample
# (-1.067 then -1.099, against PhysX's -1.047 then -1.098), and the motion reads as continuous.
# Contact behaviour is unchanged or slightly better: closing on the steering wheel peaks at
# 0.377 m/s rather than 0.436 m/s.
#
# The mimic followers -- the ``*_intermediate_joint`` and ``*_thumb_distal_joint`` -- are
# authored with zero stiffness and damping and are meant to be carried by the mimic coupling
# rather than driven, so they get an explicit passive group.
_HAND_DRIVE_STIFFNESS = 200.0
_HAND_DRIVE_DAMPING = 6.0
_HAND_DRIVE_ARMATURE = 0.01
_HAND_DRIVE_EFFORT = 2000.0


def _gr1t2_actuators():
    """Build the task's actuator set, adapting the hands and posture joints for MJWarp.

    Two things differ under Newton:

    * **Mimic followers must not be driven.** Newton lowers GR1T2's ten hand mimic couplings
      to ``mjEQ_JOINT`` equality constraints, but ``GR1T2_HIGH_PD_CFG`` drives every ``R_.*`` /
      ``L_.*`` joint, so the follower joints are position-driven *and* constrained. The two
      fight and the fingers oscillate at tens of rad/s even while holding a static pose. Drive
      only the leader (proximal) joints and leave the followers passive, mirroring the
      ``panda_finger2_passive`` group used by the Newton-validated Franka lift task.
    * **Undriven joints need damping.** The config actuates 39 of 54 joints, leaving the legs
      and head free. PhysX never excites them because gravity is disabled, but under MJWarp
      they pick up energy and spin (``head_yaw_joint`` reaches ~47 rad/s). The robot is
      fixed-base here and these joints only have to hold their default pose, so give them a
      modest posture drive.

    The actuator configs are copied so the overrides stay local to this task and do not leak
    into the shared :data:`GR1T2_HIGH_PD_CFG`.

    Returns:
        A preset selecting the PhysX or MJWarp actuator set.
    """
    physx = {name: cfg.copy() for name, cfg in GR1T2_HIGH_PD_CFG.actuators.items()}

    newton = {name: cfg.copy() for name, cfg in GR1T2_HIGH_PD_CFG.actuators.items()}
    for side in ("R", "L"):
        key = "right-hand" if side == "R" else "left-hand"
        newton[key] = ImplicitActuatorCfg(
            joint_names_expr=[f"{side}_.*_proximal_.*"],
            stiffness=_HAND_DRIVE_STIFFNESS,
            damping=_HAND_DRIVE_DAMPING,
            armature=_HAND_DRIVE_ARMATURE,
            joint_effort_limit=_HAND_DRIVE_EFFORT,
        )
        newton[f"{key}-passive"] = ImplicitActuatorCfg(
            joint_names_expr=[f"{side}_.*_intermediate_joint", f"{side}_thumb_distal_joint"],
            stiffness=0.0,
            damping=0.0,
        )
    newton["posture"] = ImplicitActuatorCfg(
        joint_names_expr=["head_.*", ".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"],
        stiffness=200.0,
        damping=20.0,
        armature=0.01,
    )
    return preset(default=physx, newton_mjwarp=newton)


# World-space bounds of the packing table's authored collider, measured from the composed
# stage (``.../SM_CratePacking_Table_A1/SM_HeavyDutyPackingTable_C02_01``). The table prim
# sits at y = 0.55, so these are centred on that in the proxy below.
_PACKING_TABLE_COLLIDER_SIZE = (2.4736, 0.762, 0.9941)
_PACKING_TABLE_COLLIDER_POS = (0.0, 0.55, 0.49705)


def _steering_wheel_spawn(func=None) -> UsdFileCfg:
    """Build the steering-wheel spawn, optionally overriding the spawner function."""
    spawn = UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
        scale=(0.75, 0.75, 0.75),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    )
    if func is not None:
        spawn.func = func
    return spawn


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the GR1T2 Pick Place Base Scene."""

    # Table
    packing_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # Static collision proxy for the tabletop, used by the Newton backend only.
    #
    # ``packing_table.usd`` authors its tabletop collider as a ``boundingCube``
    # ``PhysicsCollisionAPI`` on an Xform rather than on mesh prims. PhysX resolves that and
    # builds the collider; Newton emits no shape for it, so anything resting on the table
    # falls straight through. This invisible box reproduces the same bounding volume.
    #
    # The box is always spawned, but its collider is only enabled under ``newton_mjwarp`` so
    # PhysX keeps colliding solely with the asset's own (correctly imported) collider.
    packing_table_collider = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PackingTableCollider",
        init_state=AssetBaseCfg.InitialStateCfg(pos=list(_PACKING_TABLE_COLLIDER_POS)),
        spawn=sim_utils.CuboidCfg(
            size=_PACKING_TABLE_COLLIDER_SIZE,
            visible=False,
            collision_props=preset(
                default=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                newton_mjwarp=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            ),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.45, 0.45, 0.9996], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=preset(
            default=_steering_wheel_spawn(),
            newton_mjwarp=_steering_wheel_spawn(func=_spawn_steering_wheel_for_mjwarp),
        ),
    )

    # Humanoid robot configured for pick-place manipulation tasks
    robot: ArticulationCfg = GR1T2_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=_gr1t2_robot_spawn(),
        actuators=_gr1t2_actuators(),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.93),
            rot=(0.0, 0.0, 0.7071, 0.7071),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": -1.5708,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": -1.5708,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # --
                "head_.*": 0.0,
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                "R_.*": 0.0,
                "L_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Per-finger contact sensors on all finger links of each hand, filtered against
    # the wheel body so force_matrix_w reports each finger's grip force. This drives
    # the per-finger haptic glove feedback (see GloveHapticFeedbackCfg below).
    # Contact reporting is already enabled on the robot by GR1T2_HIGH_PD_CFG
    # (``spawn.activate_contact_sensors=True``), so it is not set again here.
    left_hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/[^/]*L_(index|middle|ring|pinky|thumb)[^/]*_link",
        filter_prim_paths_expr=[_STEERING_WHEEL_BODY],
        update_period=0.0,
        history_length=3,
    )
    right_hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/[^/]*R_(index|middle|ring|pinky|thumb)[^/]*_link",
        filter_prim_paths_expr=[_STEERING_WHEEL_BODY],
        update_period=0.0,
        history_length=3,
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class PickPlaceGR1T2SceneCfg(ObjectTableSceneCfg):
    """GR1T2 pick-place scene with the camera observation shown in XR PiP."""

    robot_pov_cam = robot_pov_camera_cfg(
        parent_prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset_pos=(0.11999996, -0.00000233, 0.74674994),
        offset_rot=(-0.69303199, 0.69304552, -0.14034840, 0.14034565),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
        ],
        hand_joint_names=[
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_thumb_proximal_yaw_joint",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            "L_index_intermediate_joint",
            "L_middle_intermediate_joint",
            "L_pinky_intermediate_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_pitch_joint",
            "R_index_intermediate_joint",
            "R_middle_intermediate_joint",
            "R_pinky_intermediate_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_pitch_joint",
            "L_thumb_distal_joint",
            "R_thumb_distal_joint",
        ],
        target_eef_link_names={
            "left_wrist": "left_hand_pitch_link",
            "right_wrist": "right_hand_pitch_link",
        },
        # the robot in the sim scene we are controlling
        asset_name="robot",
        # Configuration for the IK controller
        # The frames names are the ones present in the URDF file
        # The urdf has to be generated from the USD that is being used in the scene
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="base_link",
            num_hand_joints=22,
            show_ik_warnings=False,
            # Determines whether Pink IK solver will fail due to a joint limit violation
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTaskCfg(
                    frame="left_hand_pitch_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=12,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                FrameTaskCfg(
                    frame="right_hand_pitch_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=12,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                DampingTaskCfg(
                    cost=0.5,  # [cost] * [s] / [rad]
                ),
                NullSpacePostureTaskCfg(
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        "left_hand_pitch_link",
                        "right_hand_pitch_link",
                    ],
                    controlled_joints=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_pitch_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "right_elbow_pitch_joint",
                        "waist_yaw_joint",
                        "waist_pitch_joint",
                        "waist_roll_joint",
                    ],
                ),
            ],
            fixed_input_tasks=[],
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "left_hand_roll_link"})
        left_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "left_hand_roll_link"})
        right_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "right_hand_roll_link"})
        right_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "right_hand_roll_link"})

        hand_joint_state = ObsTerm(func=mdp.get_robot_joint_state, params={"joint_names": ["R_.*", "L_.*"]})
        head_joint_state = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": ["head_pitch_joint", "head_roll_joint", "head_yaw_joint"]},
        )

        object = ObsTerm(
            func=mdp.object_obs,
            params={"left_eef_link_name": "left_hand_roll_link", "right_eef_link_name": "right_hand_roll_link"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class PickPlaceGR1T2ObservationsCfg(ObservationsCfg):
    """GR1T2 pick-place observations including the camera shown in XR PiP."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        robot_pov_cam = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("robot_pov_cam"),
                "data_type": "rgb",
                "normalize": False,
                "clone": False,
            },
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(func=mdp.task_done_pick_place, params={"task_link_name": "right_hand_roll_link"})


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],
                "y": [-0.01, 0.01],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PhysicsCfg(PresetCfg):
    """Physics backend presets for the GR1T2 pick-place task.

    ``default`` keeps the bare :class:`~isaaclab_physx.physics.PhysxCfg` this task ran with
    before presets were exposed, so PhysX behavior is unchanged.

    The ``newton_mjwarp`` variant targets a two-armed humanoid with articulated hands: the
    contact budget is raised well above the single-arm manipulation tasks because both
    multi-finger hands can contact the object and the table at once, and the elliptic friction
    cone with a high ``impratio`` is what keeps fingertip grasps from slipping.

    ``num_substeps`` and ``ccd_iterations`` follow Newton's own dexterous-hand example rather
    than the parallel-gripper tasks. Closing a hand onto the steering wheel with the
    gripper-oriented values threw the object at 2.2 m/s and shook the fingers at 6.2 rad/s;
    with eight substeps and 50 CCD iterations the same grasp moves the object 2 mm and the
    fingers settle to 0.01 rad/s.

    The per-shape contact stiffness is deliberately left at Newton's default. Raising it to
    2.5e5, which is the published remedy for grasp *drift*, made this scene markedly worse --
    peak object speed 5.0 m/s and half a metre of travel -- because a stiffer contact ejects
    harder on the first overlap.
    """

    isaacsim_physx = PhysxCfg()
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=800,
            nconmax=600,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
            iterations=100,
            ls_iterations=15,
            ls_parallel=False,
            use_mujoco_contacts=False,
            ccd_iterations=50,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(),
        default_shape_cfg=NewtonShapeCfg(),
        num_substeps=8,
        debug_mode=False,
    )
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)
    default = isaacsim_physx


@configclass
class PickPlaceGR1T2EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 environment."""

    # Scene settings
    scene: PickPlaceGR1T2SceneCfg = PickPlaceGR1T2SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: PickPlaceGR1T2ObservationsCfg = PickPlaceGR1T2ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    # Idle action to hold robot in default pose
    # Action format: [left arm pos (3), left arm quat (4), right arm pos (3), right arm quat (4),
    #                 left hand joint pos (11), right hand joint pos (11)]
    idle_action = [
        -0.22878,
        0.2536,
        1.0953,
        0.5,
        -0.5,
        0.5,
        0.5,
        0.22878,
        0.2536,
        1.0953,
        0.5,
        -0.5,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 6
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 120  # 120Hz
        self.sim.render_interval = 2
        self.sim.physics = PhysicsCfg()
        self.num_rerenders_on_reset = 3

        # Defer USD→URDF conversion to controller initialization (requires Isaac Sim at runtime).
        self.actions.upper_body_ik.controller.usd_path = self.scene.robot.spawn.usd_path
        self.actions.upper_body_ik.controller.urdf_output_dir = self.temp_urdf_dir

        # IsaacTeleop-based teleoperation pipeline.
        self.xr = XrCfg(
            anchor_pos=(0.0, 0.0, 0.0),
            anchor_rot=(0.0, 0.0, 0.0, 1.0),
        )
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: _build_gr1t2_pickplace_pipeline()[0],
            sim_device=self.sim.device,
            xr_cfg=self.xr,
            xr_camera_feeds=[
                XrCameraFeedCfg(
                    camera_name="robot_pov_cam",
                    enable_dlss_ray_reconstruction=True,
                    dlss_exec_mode="quality",
                    offset_m=(0.0, -0.15),
                    max_update_hz=0.0,
                )
            ],
        )
        self.image_obs_list = ["robot_pov_cam"]

        # Per-finger haptic glove feedback: vibrate each finger of the operator's
        # glove in proportion to how tightly it grips the object. The session
        # always requests the push-tensor extension the glove device needs, so
        # this stays inert (no glove connected) rather than failing.
        self.haptic_feedback = GloveHapticFeedbackCfg(
            left_sensor_name="left_hand_contact",
            right_sensor_name="right_hand_contact",
        )
