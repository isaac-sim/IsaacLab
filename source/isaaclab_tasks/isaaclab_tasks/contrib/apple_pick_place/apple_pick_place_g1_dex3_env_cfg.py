# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 with Dex3 hands moves an apple onto a plate on a table, seen by a head camera.

The scene, start pose, actuator gains and timing reproduce the evaluation of a GR00T N1.7 policy fine-tuned on
real demonstrations of this task. The observation and action terms are named and ordered like that policy's
inputs and outputs, so its exported pipeline can read and write them by name.
"""

import json
import os

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg, NewtonSDFCollisionCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxConvexDecompositionCfg, PhysxRigidBodyCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.contrib.apple_pick_place import mdp
from isaaclab_tasks.contrib.robot_pov_camera_cfg import g1_robot_pov_camera_cfg
from isaaclab_tasks.utils import PresetCfg, preset

# Joint groups in the order of the policy's state inputs.
LEFT_LEG = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]
RIGHT_LEG = [name.replace("left_", "right_") for name in LEFT_LEG]
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
LEFT_ARM = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM = [name.replace("left_", "right_") for name in LEFT_ARM]
LEFT_HAND_STATE = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
]
RIGHT_HAND_STATE = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]
# The policy's action outputs order the hand joints differently from its state inputs.
LEFT_HAND_ACTION = [
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
]
RIGHT_HAND_ACTION = [name.replace("left_", "right_") for name in LEFT_HAND_ACTION]
# Joints the policy commands, in the order of its outputs: positions, then the same joints' feedforward efforts.
ACTION_JOINTS = LEFT_ARM + RIGHT_ARM + LEFT_HAND_ACTION + RIGHT_HAND_ACTION + WAIST

# Saved starts of 16 recorded real demonstrations: robot, apple and plate poses. Environment i takes start i.
START_STATES_FILE = os.path.join(os.path.dirname(__file__), "start_states.json")

# Default start: saved start 0, for anything that resets without the saved starts.
with open(START_STATES_FILE) as _f:
    _start_states = json.load(_f)
_start = _start_states["states"][0]
START_JOINT_POS = dict(zip(_start_states["joint_names"], _start["joint_positions"]))

# The robot stands on the floor with its root fixed; the legs hold the start pose.
G1_DEX3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1_29DOF_Dex3/G1_29DOF_Dex3.usd",
        rigid_props=PhysxRigidBodyCfg(disable_gravity=False),
        # Newton's self-collisions push the thighs at the hips, so only PhysX, as evaluated, keeps them.
        articulation_props=[
            PhysxArticulationCfg(enabled_self_collisions=True),
            NewtonArticulationCfg(self_collision_enabled=False),
        ],
        fix_root_link=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=tuple(_start["robot_root_pose"][:3]), joint_pos=START_JOINT_POS),
    # Gains, armature and friction of the evaluation; the policy's feedforward efforts are added to these PD efforts.
    # Without the armature, explicit PD on the light wrist and finger links is unstable at 240 Hz.
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint"],
            stiffness={".*_hip_(pitch|yaw)_joint": 40.179, ".*_hip_roll_joint": 99.098, ".*_knee_joint": 99.098},
            damping={".*_hip_(pitch|yaw)_joint": 2.558, ".*_hip_roll_joint": 6.309, ".*_knee_joint": 6.309},
            actuator_effort_limit={".*_hip_.*_joint": 88.0, ".*_knee_joint": 139.0},
            actuator_velocity_limit={".*_hip_.*_joint": 32.0, ".*_knee_joint": 20.0},
            armature={".*_hip_(pitch|yaw)_joint": 0.010178, ".*_hip_roll_joint": 0.025102, ".*_knee_joint": 0.025102},
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_.*_joint"],
            stiffness=28.501,
            damping=1.814,
            actuator_effort_limit=50.0,
            actuator_velocity_limit=37.0,
            armature=0.0072195,
            friction=0.03,
        ),
        "waist": IdealPDActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            stiffness=300.0,
            damping=5.0,
            actuator_effort_limit={"waist_yaw_joint": 88.0, "waist_(roll|pitch)_joint": 50.0},
            actuator_velocity_limit={"waist_yaw_joint": 32.0, "waist_(roll|pitch)_joint": 37.0},
            armature=0.03,
            friction=0.03,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
            stiffness={
                ".*_shoulder_(pitch|roll)_joint": 100.0,
                ".*_(shoulder_yaw|elbow)_joint": 40.0,
                ".*_wrist_.*": 20.0,
            },
            damping={".*_shoulder_(pitch|roll)_joint": 5.0, ".*_(shoulder_yaw|elbow)_joint": 2.0, ".*_wrist_.*": 2.0},
            actuator_effort_limit={".*_(shoulder_.*|elbow|wrist_roll)_joint": 25.0, ".*_wrist_(pitch|yaw)_joint": 5.0},
            actuator_velocity_limit={
                ".*_(shoulder_.*|elbow|wrist_roll)_joint": 37.0,
                ".*_wrist_(pitch|yaw)_joint": 22.0,
            },
            armature=0.03,
            friction=0.03,
        ),
        "hands": IdealPDActuatorCfg(
            joint_names_expr=[".*_hand_.*_joint"],
            stiffness=4.0,
            damping=0.5,
            actuator_effort_limit={
                ".*_hand_(index_[01]|middle_[01]|thumb_[12])_joint": 1.4,
                ".*_hand_thumb_0_joint": 2.45,
            },
            actuator_velocity_limit=10.0,
            armature=0.03,
            friction=0.03,
        ),
    },
)


# The G1 head camera of the real demonstrations, but held still at the pose of each environment's first reset (see
# ``reset_fixed_head_camera``), so it is not a child of the robot. Its offset is applied from the robot's head then.
HEAD_CAMERA_CFG = g1_robot_pov_camera_cfg().replace(prim_path="{ENV_REGEX_NS}/HeadCamera")


@configclass
class ApplePhysicsCfg(PresetCfg):
    """Physics backends: ``physics=ovphysx`` runs PhysX without Isaac Sim, ``physics=isaacsim_physx`` inside it, and
    ``physics=newton_mjwarp`` runs Newton, which every Isaac Lab install includes."""

    isaacsim_physx = PhysxCfg()
    ovphysx = OvPhysxCfg()
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=1000,
            nconmax=1000,
            cone="elliptic",
            impratio=10.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(),
    )
    default = isaacsim_physx


# The apple, plate and table assets author convex decomposition, which Newton computes on the CPU at startup, taking
# minutes per mesh; Newton uses convex hulls for the apple and plate instead. PhysX keeps the authored decomposition.
NEWTON_CONVEX_HULLS = preset(
    default=None, newton_mjwarp=sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull")
)


@configclass
class ApplePickPlaceSceneCfg(InteractiveSceneCfg):
    """Warehouse, table, apple and plate in front of the robot."""

    robot: ArticulationCfg = G1_DEX3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # The warehouse provides the floor, walls and lighting. It is 24 m x 39 m, so all environments share one;
    # a copy per environment would put its walls and shelves through the neighboring environments.
    warehouse = AssetBaseCfg(
        prim_path="/World/Warehouse",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Environments/E2E_RA_Warehouse/warehouse.usd"),
    )

    # Scaled down to the size of the table in the real demonstrations, with its colliders shrink-wrapped. The asset's
    # top is a dynamic rigid body; kinematic, the table stays where it is put instead of sliding when touched.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6163, 0.0788, 0.0411), rot=(0.0, 0.0, 0.5486, 0.8361)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Table/table.usd",
            scale=(0.5, 0.9125, 1.0),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            mesh_collision_props=preset(
                default=PhysxConvexDecompositionCfg(shrink_wrap=True),
                # Newton collides with the table's exact shape through a signed distance field; one hull around its
                # four legs would fill the space under it.
                newton_mjwarp=NewtonSDFCollisionCfg(
                    sdf_max_resolution=256, sdf_narrow_band_inner=-0.01, sdf_narrow_band_outer=0.01
                ),
            ),
        ),
    )

    apple = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Apple",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(_start["apple_pose"][:3]), rot=tuple(_start["apple_pose"][3:])
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Apple002/apple.usd",
            rigid_props=PhysxRigidBodyCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=8),
            mass_props=sim_utils.MassCfg(mass=0.05),
            mesh_collision_props=NEWTON_CONVEX_HULLS,
            activate_contact_sensors=True,
        ),
    )

    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(_start["plate_pose"][:3]), rot=tuple(_start["plate_pose"][3:])
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Clay_Plate/clay_plate.usd",
            scale=(0.7, 0.7, 0.7),
            rigid_props=PhysxRigidBodyCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=8),
            mass_props=sim_utils.MassCfg(mass=0.2),
            mesh_collision_props=NEWTON_CONVEX_HULLS,
        ),
    )

    # The apple asset nests its rigid body below the spawned prim.
    apple_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Apple/Geometry/apple_15_model_normalized_0_vis",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Plate"],
    )

    head_camera: CameraCfg = HEAD_CAMERA_CFG


@configclass
class ActionsCfg:
    """Joint position targets with feedforward efforts on the arms, hands and waist; the legs are held up."""

    joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ACTION_JOINTS, preserve_order=True, use_default_offset=False
    )
    joint_effort = base_mdp.JointEffortActionCfg(asset_name="robot", joint_names=ACTION_JOINTS, preserve_order=True)
    # The policy leaves the legs at their start pose; gravity compensation keeps the PD from sagging.
    leg_gravity = mdp.GravityCompensationActionCfg(asset_name="robot", joint_names=LEFT_LEG + RIGHT_LEG)


def _joints(names: list[str]) -> dict:
    return {"asset_cfg": SceneEntityCfg("robot", joint_names=names, preserve_order=True)}


@configclass
class ObservationsCfg:
    """Head camera image and joint positions, named like the policy's inputs."""

    @configclass
    class PolicyCfg(ObsGroup):
        ego_view = ObsTerm(
            func=base_mdp.image_rgb, params={"sensor_cfg": SceneEntityCfg("head_camera"), "normalize": False}
        )
        left_leg = ObsTerm(func=base_mdp.joint_pos, params=_joints(LEFT_LEG))
        right_leg = ObsTerm(func=base_mdp.joint_pos, params=_joints(RIGHT_LEG))
        waist = ObsTerm(func=base_mdp.joint_pos, params=_joints(WAIST))
        left_arm = ObsTerm(func=base_mdp.joint_pos, params=_joints(LEFT_ARM))
        right_arm = ObsTerm(func=base_mdp.joint_pos, params=_joints(RIGHT_ARM))
        left_hand = ObsTerm(func=base_mdp.joint_pos, params=_joints(LEFT_HAND_STATE))
        right_hand = ObsTerm(func=base_mdp.joint_pos, params=_joints(RIGHT_HAND_STATE))

        def __post_init__(self):
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Start from the saved starts, with the fingertip and apple friction of the evaluation."""

    reset_scene = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_to_start_states = EventTerm(
        func=mdp.reset_to_start_states, mode="reset", params={"states_file": START_STATES_FILE}
    )
    # After the start is set, so the camera is placed from the robot's start pose.
    reset_head_camera = EventTerm(func=mdp.reset_fixed_head_camera, mode="reset")

    right_hand_friction = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_hand_.*"),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.32, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    left_hand_friction = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_hand_.*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )
    apple_friction = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("apple"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )


@configclass
class TerminationsCfg:
    """End on success, a dropped apple, or the time limit."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    # The apple's center is inside the plate's rim, near its height, pressing on it at nearly the apple's weight
    # (0.49 N), nearly still, and clear of both thumb tips.
    success = DoneTerm(
        func=mdp.apple_on_plate,
        params={
            "radius": 0.089,
            "max_height_difference": 0.103,
            "min_contact_force": 0.44,
            "max_speed": 0.1,
            "release_distance": 0.1,
            "hand_cfg": SceneEntityCfg("robot", body_names=["left_hand_thumb_2_link", "right_hand_thumb_2_link"]),
        },
    )
    # The table top is at 0.76 m; an apple below 0.5 m has fallen off.
    apple_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("apple")}
    )


@configclass
class RewardsCfg:
    """No rewards: the task evaluates a pretrained policy."""


@configclass
class G1ApplePickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """G1 with Dex3 hands picks an apple from a table and places it on a plate."""

    scene: ApplePickPlaceSceneCfg = ApplePickPlaceSceneCfg(num_envs=1, env_spacing=3.0)
    sim: SimulationCfg = SimulationCfg(physics=ApplePhysicsCfg())
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        # The policy acts at 30 Hz, the rate of the real demonstrations, on 240 Hz physics.
        self.sim.dt = 1 / 240
        self.decimation = 8
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        # The policy's first image after a reset must show the new start, from the head camera's new pose.
        self.num_rerenders_on_reset = 1
