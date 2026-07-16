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

    The arm is clamped at the +x side edge of the dexsuite table, yawed -90 deg so it
    reaches across the table's short axis along world -x.
    """

    robot: ArticulationCfg = SO101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        # the asset's root frame is authored 3.008 cm above the bottom of its clamp foot,
        # so z = 0.255 (tabletop) - 0.03008 plants the foot on the table surface
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.16, 0.2, 0.22492),
            rot=(0.0, 0.0, -0.70710678, 0.70710678),
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.activate_contact_sensors = True
        # soften the jaw drive to the SO-101 scale: the asset-default 10 N*m effort against
        # palm-sized objects ejects them from every pinch attempt, so a stable held placement
        # (and with it the success reward) never becomes learnable
        self.robot.actuators["gripper"].effort_limit_sim = 0.5
        # the object hosts the contact sensors (see below); note that ``default`` is a deep
        # copy of ``shapes`` in the preset config, so each preset is flagged separately.
        # Objects are lightened to the jaw's scale (the dexsuite default is 0.2 kg)
        for preset in ("shapes", "cube", "default"):
            getattr(self.object.spawn, preset).activate_contact_sensors = True
            getattr(self.object.spawn, preset).mass_props.mass = 0.05
        # the jaw-vs-object contact pair, sensed from the object side (both bodies in the
        # pairing have a single collision shape, which PhysX filtered reporting requires)
        self.jaw_object_s = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1"],
        )
        # objects sized to the SO-101 jaw (~4 cm span at full open)
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
        # float the spawn region within the SO-101's measured reach envelope: 0.2 m in
        # front of the base, which with the -90 deg base yaw is along world -x
        self.object.init_state.pos = (-0.36, 0.2, 0.38)


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
    # no ``contact_count`` term: with a single jaw sensor it duplicates ``good_finger_contact``
    # exactly, and the doubled touch payout teaches parking in contact instead of transporting
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.75,
        params={"threshold": 0.01, "thumb_name": THUMB_SENSOR, "finger_names": FINGER_SENSORS},
    )

    def __post_init__(self):
        super().__post_init__()
        self.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=JAW_LIST)
        self.fingers_to_object.params["thumb_name"] = THUMB_SENSOR
        self.fingers_to_object.params["finger_names"] = FINGER_SENSORS
        self.position_tracking.params["thumb_name"] = THUMB_SENSOR
        self.position_tracking.params["finger_names"] = FINGER_SENSORS
        # widen the tracking kernel so the transport gradient reaches typical spawn-to-goal
        # distances (~0.2-0.3 m); at std 0.1 the touch stream dominates and the policy parks
        self.position_tracking.params["std"] = 0.15
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
        # goal workspace inside the measured dense reachable band (root frame, arm works at
        # -y): random-joint FK sampling puts the gripper's 25th-75th percentile envelope at
        # x (-0.08, 0.12), y (-0.24, -0.13), z (0.11, 0.30); goals beyond y=-0.31 are
        # practically unreachable and cap the achievable success rate
        self.commands.object_pose.ranges.pos_x = (-0.08, 0.08)
        self.commands.object_pose.ranges.pos_y = (-0.30, -0.16)
        self.commands.object_pose.ranges.pos_z = (0.08, 0.22)
        events = self.events.conditional_reset.params["terms"]
        events["reset_robot_wrist_joint"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names="wrist_roll")
        events["reset_object_to_target"].params["target_cfg"] = SceneEntityCfg("robot", body_names="gripper")
        # grasp center between the jaws sits at ~(0.012, -0.003, -0.097) in the gripper frame
        events["reset_object_to_target"].params["pose_range"] = {
            "x": [-0.01, 0.03],
            "y": [-0.02, 0.02],
            "z": [-0.12, -0.08],
        }
        # spawn the free-floating object within the measured reach envelope
        # (x is along the arm's reach direction, y lateral, after the -90 deg base yaw)
        events["reset_object"].params["pose_range"] = {
            "x": [-0.07, 0.07],
            "y": [-0.08, 0.08],
            "z": [0.0, 0.08],
            "roll": [-3.14, 3.14],
            "pitch": [-3.14, 3.14],
            "yaw": [-3.14, 3.14],
        }
        # table/ground clearance: everything but the table-mounted base and the shoulder
        # yoke bolted to it — with the clamp foot planted on the tabletop both live at
        # table height by construction and would reject every reset draw
        self.events.conditional_reset.params["valid_criteria"][
            "robot_table_clearance"
        ].body_names = "(?!(base|shoulder)$).*"
        # velocity-limit termination on the arm joints only: the jaw legitimately stalls on objects
        self.terminations.abnormal_robot.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names="(shoulder_pan|shoulder_lift|elbow_flex|wrist_flex|wrist_roll)"
        )
        # keep the generic gain randomization off the jaw (cf. the Franka config): scaling the
        # soft jaw drive by x2 restores the object-ejecting grip the effort limit removes
        self.events.joint_stiffness_and_damping.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names="(shoulder_pan|shoulder_lift|elbow_flex|wrist_flex|wrist_roll)"
        )
        # size the inertia randomization to the palm-sized objects: the dexsuite default adds
        # 0.01 kg*m^2, three orders of magnitude above these objects' natural inertia, which
        # gyroscopically freezes their rotation and fights reorienting a held object
        self.events.object_physics_inertia.params["inertia_distribution_params"] = (0.0002, 0.0002)
        # give the 5-DOF arm time to transport a held object per goal: repositioning the
        # pinch across the workspace needs wrist reorientation a 6-DOF arm does not
        self.commands.object_pose.resampling_time_range = (4.0, 6.0)


@configclass
class DexsuiteSO101LiftEnvCfg(SO101MixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteSO101LiftEnvCfg_PLAY(SO101MixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass
