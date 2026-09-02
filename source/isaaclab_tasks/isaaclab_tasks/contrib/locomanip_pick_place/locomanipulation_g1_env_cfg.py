# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_teleop import (
    ControllerHapticFeedbackCfg,
    IsaacTeleopCfg,
    XrAnchorRotationMode,
    XrCameraFeedCfg,
    XrCfg,
)

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.locomanip_pick_place import mdp as locomanip_mdp
from isaaclab_tasks.contrib.locomanip_pick_place.configs.action_cfg import AgileBasedLowerBodyActionCfg
from isaaclab_tasks.contrib.locomanip_pick_place.configs.agile_locomotion_observation_cfg import (
    AgileTeacherPolicyObservationsCfg,
)
from isaaclab_tasks.contrib.pick_place import mdp as manip_mdp
from isaaclab_tasks.utils import PresetCfg, preset

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

from isaaclab_tasks.contrib.locomanip_pick_place.configs.pink_controller_cfg import (  # isort: skip
    G1_UPPER_BODY_IK_ACTION_CFG,
)
from isaaclab_tasks.contrib.robot_pov_camera_cfg import robot_pov_camera_cfg  # isort: skip


def _build_g1_locomanipulation_pipeline():
    """Build an IsaacTeleop retargeting pipeline for G1 locomanipulation teleoperation.

    Creates two Se3AbsRetargeters for left and right wrist pose tracking,
    two TriHandMotionControllerRetargeters for left and right hand joint
    control from VR controller buttons, and a LocomotionRootCmdRetargeter
    for base velocity commands from controller thumbsticks. All outputs
    are flattened into a single action tensor via TensorReorderer.

    Returns:
        OutputCombiner with a single "action" output containing the flattened
        32D action tensor: [left_wrist(7), right_wrist(7), hand_joints(14), locomotion(4)].
    """
    from isaacteleop.retargeters import (
        LocomotionRootCmdRetargeter,
        LocomotionRootCmdRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
        TriHandMotionControllerConfig,
        TriHandMotionControllerRetargeter,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    # Create input sources (trackers are auto-discovered from pipeline)
    controllers = ControllersSource(name="controllers")

    # External input: world-to-anchor 4x4 transform matrix provided by IsaacTeleopDevice
    transform_input = ValueInput("world_T_anchor", TransformMatrix())

    # Apply the coordinate-frame transform to controller poses so that
    # downstream retargeters receive data in the simulation world frame.
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    # -------------------------------------------------------------------------
    # SE3 Absolute Pose Retargeters (left and right wrists)
    # -------------------------------------------------------------------------
    # Rotation offsets from G1TriHandUpperBodyRetargeter._retarget_abs:
    #   Left:  (-0.2706, 0.6533, 0.2706, 0.6533) xyzw  -- 90 deg about Y then -45 deg about X
    #   Right: (-0.7071, 0, 0.7071, 0) xyzw

    left_se3_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.LEFT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        target_offset_roll=45.0,
        target_offset_pitch=180.0,
        target_offset_yaw=-90.0,
    )
    left_se3 = Se3AbsRetargeter(left_se3_cfg, name="left_ee_pose")
    connected_left_se3 = left_se3.connect(
        {
            ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT),
        }
    )

    right_se3_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        target_offset_roll=-135.0,
        target_offset_pitch=0.0,
        target_offset_yaw=90.0,
    )
    right_se3 = Se3AbsRetargeter(right_se3_cfg, name="right_ee_pose")
    connected_right_se3 = right_se3.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
        }
    )

    # -------------------------------------------------------------------------
    # TriHand Motion Controller Retargeters (left and right hands)
    # -------------------------------------------------------------------------
    # Generic joint names matching TriHand 7-DOF output order:
    #   [thumb_rotation, thumb_proximal, thumb_distal,
    #    index_proximal, index_distal, middle_proximal, middle_distal]
    hand_joint_names = [
        "thumb_rotation",
        "thumb_proximal",
        "thumb_distal",
        "index_proximal",
        "index_distal",
        "middle_proximal",
        "middle_distal",
    ]

    left_trihand_cfg = TriHandMotionControllerConfig(
        hand_joint_names=hand_joint_names,
        controller_side="left",
    )
    left_trihand = TriHandMotionControllerRetargeter(left_trihand_cfg, name="trihand_left")
    connected_left_trihand = left_trihand.connect(
        {
            ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT),
        }
    )

    right_trihand_cfg = TriHandMotionControllerConfig(
        hand_joint_names=hand_joint_names,
        controller_side="right",
    )
    right_trihand = TriHandMotionControllerRetargeter(right_trihand_cfg, name="trihand_right")
    connected_right_trihand = right_trihand.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
        }
    )

    # -------------------------------------------------------------------------
    # Locomotion Root Command Retargeter (base velocity from thumbsticks)
    # -------------------------------------------------------------------------
    locomotion_cfg = LocomotionRootCmdRetargeterConfig(
        initial_hip_height=0.72,
        movement_scale=0.5,
        rotation_scale=0.35,
        dt=1.0 / 100.0,  # Must match rendering dt: sim.dt (1/200) * render_interval (2)
    )
    locomotion = LocomotionRootCmdRetargeter(locomotion_cfg, name="locomotion")
    connected_locomotion = locomotion.connect(
        {
            "controller_left": controllers.output(ControllersSource.LEFT),
            "controller_right": controllers.output(ControllersSource.RIGHT),
        }
    )

    # -------------------------------------------------------------------------
    # TensorReorderer: flatten into a 32D action tensor
    # -------------------------------------------------------------------------
    # Se3AbsRetargeter outputs 7D arrays: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
    left_ee_elements = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w"]
    right_ee_elements = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w"]

    # TriHand outputs 7 scalars per hand (positionally mapped):
    #   [thumb_rotation, thumb_proximal, thumb_distal,
    #    index_proximal, index_distal, middle_proximal, middle_distal]
    left_hand_elements = [
        "l_thumb_rotation",
        "l_thumb_proximal",
        "l_thumb_distal",
        "l_index_proximal",
        "l_index_distal",
        "l_middle_proximal",
        "l_middle_distal",
    ]
    right_hand_elements = [
        "r_thumb_rotation",
        "r_thumb_proximal",
        "r_thumb_distal",
        "r_index_proximal",
        "r_index_distal",
        "r_middle_proximal",
        "r_middle_distal",
    ]

    # Locomotion outputs 4D array: [vel_x, vel_y, rot_vel_z, hip_height]
    locomotion_elements = ["loco_vel_x", "loco_vel_y", "loco_rot_vel_z", "loco_hip_height"]

    # Output order must match the action space layout expected by the environment:
    #   [left_wrist(7), right_wrist(7), hand_joints(14), locomotion(4)]
    # Hand joints follow hand_joint_names order from G1_UPPER_BODY_IK_ACTION_CFG.
    # Locomotion (4D) is consumed by AgileBasedLowerBodyAction.
    output_order = (
        left_ee_elements
        + right_ee_elements
        + [
            # hand_joint_names indices 0-5  (proximal / 0-joints)
            "l_index_proximal",
            "l_middle_proximal",
            "l_thumb_rotation",
            "r_index_proximal",
            "r_middle_proximal",
            "r_thumb_rotation",
            # hand_joint_names indices 6-11 (distal / 1-joints)
            "l_index_distal",
            "l_middle_distal",
            "l_thumb_proximal",
            "r_index_distal",
            "r_middle_distal",
            "r_thumb_proximal",
            # hand_joint_names indices 12-13 (thumb tip / 2-joints)
            "l_thumb_distal",
            "r_thumb_distal",
        ]
        + locomotion_elements
    )

    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": left_ee_elements,
            "right_ee_pose": right_ee_elements,
            "left_hand_joints": left_hand_elements,
            "right_hand_joints": right_hand_elements,
            "locomotion": locomotion_elements,
        },
        output_order=output_order,
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_hand_joints": "scalar",
            "right_hand_joints": "scalar",
            "locomotion": "array",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "left_ee_pose": connected_left_se3.output("ee_pose"),
            "right_ee_pose": connected_right_se3.output("ee_pose"),
            "left_hand_joints": connected_left_trihand.output("hand_joints"),
            "right_hand_joints": connected_right_trihand.output("hand_joints"),
            "locomotion": connected_locomotion.output("root_command"),
        }
    )

    return OutputCombiner({"action": connected_reorderer.output("output")})


##
# Scene definition
##
# Newton resolves an omitted friction value to zero, so a body with no authored material has no
# friction at all and slides out of the grasp. Author it on both the robot and the grasped object.
_ROBOT_CONTACT_MATERIAL = preset(
    default=None,
    newton_mjwarp=[
        UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        NewtonMaterialCfg(contact_stiffness=1.0e6, contact_damping=2000.0),
    ],
)

# Object the hands grasp, selected per backend.
#
# PhysX keeps the authored steering wheel. MJWarp cannot use it: the wheel's rim is a torus and
# MuJoCo has no concave mesh-mesh collision, so however the rim is approximated the hand either
# passes through it (imported as a mesh) or the ring fills in and the wheel becomes a solid disc
# (collapsed to a convex hull). The stand-in is the graspable primitive from
# ``Isaac-Lift-Franka``, which runs MJWarp as its default backend. Replace it once the asset
# ships a convex-decomposed rim.
_NEWTON_OBJECT_SIZE = 0.035
"""Edge length of the MJWarp stand-in object [m]."""

_NEWTON_OBJECT_MASS = 0.2
"""Mass of the MJWarp stand-in object [kg], matching the validated lift-task primitive."""


def _steering_wheel_spawn() -> UsdFileCfg:
    """Build the authored steering-wheel spawn used by the PhysX path."""
    return UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
        scale=(0.75, 0.75, 0.75),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    )


def _newton_object_spawn() -> sim_utils.CuboidCfg:
    """Build the graspable primitive used in place of the steering wheel under MJWarp."""
    return sim_utils.CuboidCfg(
        size=(_NEWTON_OBJECT_SIZE, _NEWTON_OBJECT_SIZE, _NEWTON_OBJECT_SIZE),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=_NEWTON_OBJECT_MASS),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.2)),
    )


# Newton matches contact sensors against the model's body labels, which keep the asset's
# intermediate grouping prim: ``/Robot/left_hand/left_hand_index_0_link``. The PhysX pattern
# below stops at ``/Robot/`` and ``[^/]*`` cannot cross a path separator, so it full-matches
# nothing under MJWarp and sensor init fails with "No bodies matched the sensing object
# pattern(s)". Always confirm a sensor pattern against ``model.body_label`` when porting.
def _hand_contact_prim_path(side: str) -> object:
    """Per-backend prim-path pattern for a hand's contact sensor."""
    return preset(
        default="{ENV_REGEX_NS}/Robot/" + f"{side}_hand_[^/]*_link",
        newton_mjwarp="{ENV_REGEX_NS}/Robot/" + f"{side}_hand/{side}_hand_[^/]*_link",
    )


def _g1_robot_spawn():
    """Copy the shared G1 spawn and author a contact material for MJWarp.

    The spawn is copied so the override stays local to this task and does not leak into
    :data:`G1_29DOF_CFG`. Gravity is deliberately left enabled: this robot walks, so the
    lower-body policy needs real ground contact.
    """
    spawn = G1_29DOF_CFG.spawn.copy()
    spawn.physics_material = _ROBOT_CONTACT_MATERIAL
    return spawn


@configclass
class PhysicsCfg(PresetCfg):
    """Physics backend presets for the G1 locomanipulation task.

    ``default`` keeps the bare :class:`~isaaclab_physx.physics.PhysxCfg` this task ran with before
    presets were exposed, so PhysX behavior is unchanged.

    The ``newton_mjwarp`` profile follows the locomotion tasks rather than the fixed-base
    dexterous ones: this robot walks, so the contact budget has to cover two feet on the ground
    plus a two-handed grasp, and the friction cone stays pyramidal as it is for the other Unitree
    locomotion presets. Unlike a fixed-base task, gravity must stay enabled -- the lower-body
    policy needs real ground contact, so the ``gravcomp`` trick used by fixed-base humanoids does
    not apply here.
    """

    isaacsim_physx = PhysxCfg()
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=400,
            nconmax=250,
            impratio=1.0,
            cone="pyramidal",
            update_data_interval=2,
            iterations=100,
            ls_iterations=15,
            ls_parallel=False,
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(),
        default_shape_cfg=NewtonShapeCfg(),
        num_substeps=2,
    )
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)
    default = isaacsim_physx


@configclass
class LocomanipulationG1SceneCfg(InteractiveSceneCfg):
    """Scene configuration for locomanipulation environment with G1 robot.

    This configuration sets up the G1 humanoid robot for locomanipulation tasks,
    allowing both locomotion and manipulation capabilities. The robot can move its
    base and use its arms for manipulation tasks.
    """

    # Table
    packing_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, -0.3], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.45, 0.6996], rot=[0, 0, 0, 1]),
        spawn=preset(default=_steering_wheel_spawn(), newton_mjwarp=_newton_object_spawn()),
    )

    # Humanoid robot w/ arms higher
    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", spawn=_g1_robot_spawn())

    # Use the calibrated G1 head-camera view shared with IsaacLab-Arena.
    robot_pov_cam = robot_pov_camera_cfg(
        parent_prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_link",
        offset_pos=(0.04485, 0.0, 0.35325),
        offset_rot=(-0.62721, 0.62721, -0.32651, 0.32651),
    ).replace(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_link/RobotHeadCam",
        height=480,
        width=640,
        spawn=sim_utils.PinholeCameraCfg(focal_length=15.0, horizontal_aperture=20.955, clipping_range=(0.1, 5.0)),
    )

    # Per-hand contact sensors over all finger links, used to drive controller
    # haptics (see HapticFeedbackCfg below). Requires activate_contact_sensors
    # on the robot spawn, enabled in the env __post_init__.
    left_hand_contact = ContactSensorCfg(
        prim_path=_hand_contact_prim_path("left"),
        update_period=0.0,
        history_length=3,
    )
    right_hand_contact = ContactSensorCfg(
        prim_path=_hand_contact_prim_path("right"),
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
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = G1_UPPER_BODY_IK_ACTION_CFG

    lower_body_joint_pos = AgileBasedLowerBodyActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_hip_.*_joint",
            ".*_knee_joint",
            ".*_ankle_.*_joint",
        ],
        policy_output_scale=0.25,
        obs_group_name="lower_body_policy",  # need to be the same name as the on in ObservationCfg
        policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt",
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.
    This class is required by the environment configuration but not used in this implementation
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=manip_mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        robot_links_state = ObsTerm(func=manip_mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "left_wrist_yaw_link"})
        left_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "left_wrist_yaw_link"})
        right_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "right_wrist_yaw_link"})
        right_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "right_wrist_yaw_link"})

        hand_joint_state = ObsTerm(func=manip_mdp.get_robot_joint_state, params={"joint_names": [".*_hand.*"]})

        object = ObsTerm(
            func=manip_mdp.object_obs,
            params={"left_eef_link_name": "left_wrist_yaw_link", "right_eef_link_name": "right_wrist_yaw_link"},
        )

        robot_pov_cam = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("robot_pov_cam"),
                "data_type": "rgb",
                "normalize": False,
                "clone": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    lower_body_policy: AgileTeacherPolicyObservationsCfg = AgileTeacherPolicyObservationsCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    object_too_far = DoneTerm(
        func=locomanip_mdp.object_too_far_from_robot,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
            "max_distance": 1.0,
        },
    )

    success = DoneTerm(
        func=manip_mdp.task_done_pick_place,
        params={
            "task_link_name": "right_wrist_yaw_link",
        },
    )


##
# MDP settings
##


@configclass
class LocomanipulationG1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 locomanipulation environment.

    This environment is designed for locomanipulation tasks where the G1 humanoid robot
    can perform both locomotion and manipulation simultaneously. The robot can move its
    base and use its arms for manipulation tasks, enabling complex mobile manipulation
    behaviors.
    """

    # Scene settings
    scene: LocomanipulationG1SceneCfg = LocomanipulationG1SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    rewards = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 200  # 200Hz
        self.sim.render_interval = 2
        self.sim.physics = PhysicsCfg()
        self.num_rerenders_on_reset = 3

        # Set the URDF path for the IK controller. Path resolution (Nucleus → local) happens at runtime.
        self.actions.upper_body_ik.controller.urdf_path = f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/unitree_g1_kinematics_asset/g1_29dof_with_hand_only_kinematics.urdf"  # noqa: E501

        self.xr = XrCfg(
            anchor_pos=(0.0, 0.0, -0.95),
            anchor_rot=(0.0, 0.0, 0.0, 1.0),
        )
        self.xr.anchor_prim_path = "/World/envs/env_0/Robot/pelvis"
        self.xr.fixed_anchor_height = True
        self.xr.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED

        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=_build_g1_locomanipulation_pipeline,
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

        # Enable contact reporting on the robot so the per-hand ContactSensors
        # report finger forces, and drive controller haptics from them.
        self.scene.robot.spawn.activate_contact_sensors = True
        self.haptic_feedback = ControllerHapticFeedbackCfg(
            left_sensor_name="left_hand_contact",
            right_sensor_name="right_hand_contact",
        )
