# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka lift and reorient environments."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import MeshCapsuleCfg, MeshCuboidCfg, MeshSphereCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.core.lift.lift_env_cfg as lift
import isaaclab_tasks.core.lift.mdp as mdp

from isaaclab_assets.robots import FRANKA_PANDA_CFG

##
# Scene assets
##

# The lift tasks run the menagerie-converted asset (identified inertials, authored finger coupling) with
# actuators calibrated for it, while the other Franka tasks keep the stock asset.
FRANKA_PANDA_LIFT_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_PANDA_LIFT_CFG.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/franka_panda.usda"
# Reset clearance was calibrated for these arm meshes; the asset's primitive colliders intersect the ground.
FRANKA_PANDA_LIFT_CFG.spawn.variants = {"Colliders": "convex_hulls"}
FRANKA_PANDA_LIFT_CFG.actuators = {
    # inspired by libfranka's joint_impedance_control.cpp; ``actuator_velocity_limit`` is the soft task
    # limit and ``joint_velocity_limit`` the separate solver request
    "panda_arm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-7]"],
        joint_effort_limit={"panda_joint[1-4]": 87.0, "panda_joint[5-7]": 12.0},
        actuator_velocity_limit={"panda_joint[1-4]": 2.175, "panda_joint[5-7]": 2.61},
        joint_velocity_limit={"panda_joint[1-4]": 20.0, "panda_joint[5-7]": 25.0},
        stiffness={
            "panda_joint[1-4]": 600.0,
            "panda_joint5": 250.0,
            "panda_joint6": 150.0,
            "panda_joint7": 50.0,
        },
        damping={
            "panda_joint[1-4]": 50.0,
            "panda_joint5": 30.0,
            "panda_joint6": 25.0,
            "panda_joint7": 15.0,
        },
        armature={
            "panda_joint[1-2]": 0.6057,
            "panda_joint[3-4]": 0.4625,
            "panda_joint[5-7]": 0.2055,
        },
    ),
    "panda_hand": ImplicitActuatorCfg(
        joint_names_expr=["panda_finger_joint1"],
        joint_effort_limit=70.0,
        actuator_velocity_limit=0.2,
        joint_velocity_limit=2.0,
        stiffness=350.0,
        damping=175.0,
        armature=0.1,
    ),
    "panda_finger2_passive": ImplicitActuatorCfg(
        joint_names_expr=["panda_finger_joint2"],
        joint_effort_limit=1.0,
        actuator_velocity_limit=0.2,
        joint_velocity_limit=2.0,
        stiffness=0.0,
        damping=0.0,
        armature=0.1,
    ),
}

"""Franka Panda configuration for the lift tasks."""

FINGERTIP_LIST = ["panda_rightfinger", "panda_leftfinger"]
"""Finger bodies that carry an object contact sensor."""

THUMB_SENSOR = "panda_leftfinger_object_s"
"""Contact sensor that plays the thumb in the finger-contact rewards."""

FINGER_SENSORS = [f"{name}_object_s" for name in FINGERTIP_LIST if name != "panda_leftfinger"]
"""Contact sensors of the remaining fingers."""


##
# Scene definition
##


@configclass
class FrankaSceneCfg(lift.SceneCfg):
    """Franka scene for the lift and reorient tasks."""

    robot: ArticulationCfg = FRANKA_PANDA_LIFT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.activate_contact_sensors = True
        # the base is rotated by 180 degrees about z so the workspace lies at positive x
        self.robot.init_state.rot = (0.0, 0.0, 1.0, 0.0)
        # one object contact sensor per finger
        for link_name in FINGERTIP_LIST:
            setattr(
                self,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Geometry/panda_link0/panda_link1/panda_link2/panda_link3/panda_link4/panda_link5/panda_link6/panda_link7/panda_hand/"
                    + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
        graspable_shape_assets_cfg = [
            MeshCuboidCfg(size=(0.05, 0.05, 0.05), **lift.OBJECT_PHYSICS),
            MeshCuboidCfg(size=(0.025, 0.05, 0.05), **lift.OBJECT_PHYSICS),
            MeshCuboidCfg(size=(0.025, 0.025, 0.05), **lift.OBJECT_PHYSICS),
            MeshCuboidCfg(size=(0.01, 0.05, 0.05), **lift.OBJECT_PHYSICS),
            MeshSphereCfg(radius=0.02, **lift.OBJECT_PHYSICS),
            MeshCapsuleCfg(radius=0.025, height=0.1, **lift.OBJECT_PHYSICS),
            MeshCapsuleCfg(radius=0.025, height=0.2, **lift.OBJECT_PHYSICS),
            MeshCapsuleCfg(radius=0.01, height=0.2, **lift.OBJECT_PHYSICS),
        ]
        self.object.spawn.shapes.assets_cfg = graspable_shape_assets_cfg
        self.object.spawn.default.assets_cfg = graspable_shape_assets_cfg


##
# MDP settings
##


@configclass
class StateObservationCfg(lift.ObservationsCfg):
    """State observations for the Franka lift tasks."""

    def __post_init__(self):
        super().__post_init__()
        self.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in FINGERTIP_LIST]},
            clip=(-20.0, 20.0),
        )
        self.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = FINGERTIP_LIST


@configclass
class FrankaRelJointPosActionCfg:
    """Relative joint position targets for all joints."""

    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class FrankaReorientRewardCfg(lift.RewardsCfg):
    """Reward terms for the MDP, with the Franka finger contact sensors filled in."""

    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.75,
        params={"threshold": 0.01, "thumb_name": THUMB_SENSOR, "finger_names": FINGER_SENSORS},
    )

    contact_count = RewTerm(
        func=mdp.contact_count,
        weight=0.1,
        params={"threshold": 0.01, "sensor_names": FINGER_SENSORS + [THUMB_SENSOR]},
    )

    def __post_init__(self):
        super().__post_init__()
        self.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=".*finger")
        self.fingers_to_object.params["thumb_name"] = THUMB_SENSOR
        self.fingers_to_object.params["finger_names"] = FINGER_SENSORS
        self.position_tracking.params["thumb_name"] = THUMB_SENSOR
        self.position_tracking.params["finger_names"] = FINGER_SENSORS
        if self.orientation_tracking:
            self.orientation_tracking.params["thumb_name"] = THUMB_SENSOR
            self.orientation_tracking.params["finger_names"] = FINGER_SENSORS
        self.success.params["thumb_name"] = THUMB_SENSOR
        self.success.params["finger_names"] = FINGER_SENSORS


@configclass
class FrankaEventCfg(lift.EventCfg):
    """Franka-specific event configuration."""

    # closing speed = kp * action_scale / kd; kd range derived for 0.2 .. 0.01 m/s so it
    # tracks kp/base-kd/scale changes. Driven finger only: damping on the passive mimic
    # joint drags the pair asymmetrically.
    gripper_closing_speed = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint1"),
            "damping_distribution_params": (
                0.0,
                FRANKA_PANDA_LIFT_CFG.actuators["panda_hand"].stiffness * 0.1 / 0.01
                - FRANKA_PANDA_LIFT_CFG.actuators["panda_hand"].damping,
            ),
            "operation": "add",
        },
    )

    def __post_init__(self):
        super().__post_init__()
        reset_terms = self.conditional_reset.params["terms"]
        criteria = self.conditional_reset.params["valid_criteria"]
        # the coupled finger pair is one mechanical DOF: independent per-joint draws write
        # equality-violating states the solver snaps shut at birth, so the pair shares one draw
        reset_terms["reset_robot_wrist_joint"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names="panda_joint7")
        reset_terms["reset_robot_joints"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names="panda_joint.*")
        fingers = SceneEntityCfg("robot", joint_names="panda_finger_joint.*")
        reset_terms["reset_gripper_width"] = EventTerm(
            func=mdp.reset_joints_shared_offset,
            mode="reset",
            params={"position_range": [-0.04, 0.0], "asset_cfg": fingers},
        )
        # spawn-in-hand: the grasp center sits ~0.10 m along the hand z-axis (fingertip plane)
        to_target = reset_terms["reset_object_to_target"].params
        to_target["target_cfg"] = SceneEntityCfg("robot", body_names="panda_hand")
        to_target["pose_range"] = {"x": [-0.02, 0.02], "y": [-0.02, 0.02], "z": [0.08, 0.12]}
        # every link but the ground-mounted base (a base-link ground check is unsatisfiable)
        criteria["robot_table_clearance"].body_names = ["panda_link[1-7]", "panda_hand", ".*finger"]
        # spread the reset bank over the grasp geometry, same bodies as fingers_to_object
        diversity_feature = self.conditional_reset.params.get("diversity_feature")
        if diversity_feature is not None:
            diversity_feature.body_names = ".*finger"
        # keep the generic gain randomization off the fingers: the closing-speed term owns
        # their damping, and stacking the x2 scale on top cancels the grip force entirely.
        self.joint_stiffness_and_damping.params["asset_cfg"] = SceneEntityCfg("robot", joint_names="panda_joint.*")


##
# Environment configuration
##


@configclass
class FrankaMixinCfg:
    """Franka-specific scene, observation, action, reward and event terms, mixed into the task configurations."""

    scene: FrankaSceneCfg = FrankaSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)
    rewards: FrankaReorientRewardCfg = FrankaReorientRewardCfg()
    observations: StateObservationCfg = StateObservationCfg()
    actions: FrankaRelJointPosActionCfg = FrankaRelJointPosActionCfg()
    events: FrankaEventCfg = FrankaEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.body_name = "panda_hand"
        # Franka base is rotated 180 deg about z, so the workspace mirrors to positive x.
        self.commands.object_pose.ranges.pos_x = (0.3, 0.7)
        self.terminations.abnormal_robot.params["asset_cfg"] = SceneEntityCfg("robot", joint_names="panda_joint.*")


@configclass
class FrankaReorientEnvCfg(FrankaMixinCfg, lift.ReorientEnvCfg):
    """Franka object reorientation environment."""

    def play_mode(self):
        super().play_mode()
        # evaluate at the datasheet gripper speed: without the closing-speed randomization the hand
        # damping caps closing at the real hand's jaw-speed limit of 0.2 m/s
        self.events.gripper_closing_speed = None


@configclass
class FrankaLiftEnvCfg(FrankaMixinCfg, lift.LiftEnvCfg):
    """Franka object lifting environment."""

    def play_mode(self):
        super().play_mode()
        # evaluate at the datasheet gripper speed: without the closing-speed randomization the hand
        # damping caps closing at the real hand's jaw-speed limit of 0.2 m/s
        self.events.gripper_closing_speed = None
