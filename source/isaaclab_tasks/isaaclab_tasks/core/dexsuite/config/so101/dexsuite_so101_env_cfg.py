# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import MeshCapsuleCfg, MeshCuboidCfg, MeshSphereCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.so101 import SO101_CFG

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp

# The single-jaw gripper opposes the moving jaw (thumb analog) against the fixed finger on
# the gripper body. Only the moving jaw carries a contact sensor: the gripper body bundles
# two collision shapes (servo + shell) and PhysX filtered contact reporting only supports
# single-shape pairings, so the jaw-object pair is the one observable contact. A free object
# cannot sustain jaw force without force closure against the fixed finger, so the jaw force
# alone is a faithful pinch signal for the contact-gated tracking rewards.
JAW_LIST = ["gripper", "moving_jaw_so101_v1"]
THUMB_SENSOR = "jaw_object_s"
FINGER_SENSORS = [THUMB_SENSOR]


@configclass
class SO101SceneCfg(dexsuite.SceneCfg):
    """SO-101 scene for the dexsuite lift task.

    The arm is seated on the dexsuite table (top surface at z=0.255) in its default state;
    at the identity base orientation it reaches along -y over the tabletop.
    """

    robot: ArticulationCfg = SO101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(-0.55, 0.2, 0.255)),
    )

    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.activate_contact_sensors = True
        # the object hosts the contact sensors (see below); note that ``default`` is a deep
        # copy of ``shapes`` in the preset config, so each preset is flagged separately
        for preset in ("shapes", "cube", "default"):
            getattr(self.object.spawn, preset).activate_contact_sensors = True
        # the jaw-vs-object contact pair, sensed from the object side (both bodies in the
        # pairing have a single collision shape, which PhysX filtered reporting requires)
        self.jaw_object_s = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1"],
        )
        # objects sized to the SO-101 jaw (~4 cm span at full open, ~0.2 N*m drive)
        graspable_shape_assets_cfg = [
            MeshCuboidCfg(size=(0.03, 0.03, 0.03), **dexsuite.OBJECT_PHYSICS),
            MeshCuboidCfg(size=(0.02, 0.03, 0.03), **dexsuite.OBJECT_PHYSICS),
            MeshCuboidCfg(size=(0.02, 0.02, 0.03), **dexsuite.OBJECT_PHYSICS),
            MeshCuboidCfg(size=(0.01, 0.03, 0.03), **dexsuite.OBJECT_PHYSICS),
            MeshSphereCfg(radius=0.015, **dexsuite.OBJECT_PHYSICS),
            MeshCapsuleCfg(radius=0.015, height=0.05, **dexsuite.OBJECT_PHYSICS),
            MeshCapsuleCfg(radius=0.01, height=0.08, **dexsuite.OBJECT_PHYSICS),
        ]
        self.object.spawn.shapes.assets_cfg = graspable_shape_assets_cfg
        self.object.spawn.default.assets_cfg = graspable_shape_assets_cfg
        # float the spawn region within the SO-101's compact reach envelope
        self.object.init_state.pos = (-0.55, -0.08, 0.38)


@configclass
class SO101RelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class SO101StateObservationCfg(dexsuite.ObservationsCfg):
    """State observations for the SO-101 dexsuite lift task."""

    def __post_init__(self):
        super().__post_init__()
        self.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [THUMB_SENSOR]},
            clip=(-20.0, 20.0),  # jaw contact force stays well under 20 N
        )
        self.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = JAW_LIST


@configclass
class SO101LiftRewardCfg(dexsuite.RewardsCfg):
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=1.0,
        params={"threshold": 0.01, "thumb_name": THUMB_SENSOR, "finger_names": FINGER_SENSORS},
    )

    contact_count = RewTerm(
        func=mdp.contact_count,
        weight=1.0,
        params={"threshold": 0.01, "sensor_names": [THUMB_SENSOR]},
    )

    def __post_init__(self):
        super().__post_init__()
        self.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=JAW_LIST)
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
class SO101MixinCfg:
    scene: SO101SceneCfg = SO101SceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)
    rewards: SO101LiftRewardCfg = SO101LiftRewardCfg()
    observations: SO101StateObservationCfg = SO101StateObservationCfg()
    actions: SO101RelJointPosActionCfg = SO101RelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "gripper"
        # goal workspace within the SO-101's compact reach (root frame: the arm works at -y)
        self.commands.object_pose.ranges.pos_x = (-0.1, 0.1)
        self.commands.object_pose.ranges.pos_y = (-0.36, -0.2)
        self.commands.object_pose.ranges.pos_z = (0.1, 0.22)
        events = self.events.conditional_reset.params["terms"]
        events["reset_robot_wrist_joint"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names="wrist_roll")
        events["reset_object_to_target"].params["target_cfg"] = SceneEntityCfg("robot", body_names="gripper")
        # grasp center between the jaws sits at ~(0.012, -0.003, -0.097) in the gripper frame
        events["reset_object_to_target"].params["pose_range"] = {
            "x": [-0.01, 0.03],
            "y": [-0.02, 0.02],
            "z": [-0.12, -0.08],
        }
        # spawn the free-floating object within the arm's reach envelope
        events["reset_object"].params["pose_range"] = {
            "x": [-0.1, 0.1],
            "y": [-0.08, 0.08],
            "z": [0.0, 0.08],
            "roll": [-3.14, 3.14],
            "pitch": [-3.14, 3.14],
            "yaw": [-3.14, 3.14],
        }
        # table/ground clearance: everything but the table-mounted base
        self.events.conditional_reset.params["valid_criteria"]["robot_table_clearance"].body_names = "(?!base$).*"
        # velocity-limit termination on the arm joints only: the jaw legitimately stalls on objects
        self.terminations.abnormal_robot.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names="(shoulder_pan|shoulder_lift|elbow_flex|wrist_flex|wrist_roll)"
        )


@configclass
class DexsuiteSO101LiftEnvCfg(SO101MixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteSO101LiftEnvCfg_PLAY(SO101MixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass
