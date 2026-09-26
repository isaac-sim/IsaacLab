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
from isaaclab.utils import configclass

from isaaclab_tasks.utils import preset

from isaaclab_assets.robots.so101 import SO101_CFG

from ... import lift_env_cfg as lift
from ... import mdp

# The object-side sensor measures contact with the moving jaw. The fixed finger shares a body
# with multiple collision shapes, which prevents filtered PhysX pair reporting for that body.
JAW_LIST = ["gripper", "moving_jaw_so101_v1"]
THUMB_SENSOR = "jaw_object_s"
FINGER_SENSORS = [THUMB_SENSOR]


@configclass
class SO101SceneCfg(lift.SceneCfg):
    """SO-101 scene for the lift task.

    The arm is clamped at the +x side edge of the table, yawed -90 deg so it
    reaches across the table's short axis along world -x.
    """

    robot: ArticulationCfg = SO101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=SO101_CFG.spawn.replace(
            variants={
                "Robot": "robot",
                "Sensor": "sensors",
                "Physics": preset(
                    default="physics", isaacsim_physx="physx", physx="physx", ovphysx="physx", newton_mjwarp="physics"
                ),
            }
        ),
        # the asset's root frame is authored 3.008 cm above the bottom of its clamp foot,
        # so z = 0.255 (tabletop) - 0.03008 plants the foot on the table surface
        init_state=SO101_CFG.init_state.replace(
            pos=(-0.16, 0.2, 0.22492),
            rot=(0.0, 0.0, -0.70710678, 0.70710678),
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        # The shared SO101 asset already activates contact sensing and supplies SysID drives.
        # the object hosts the contact sensors (see below); note that ``default`` is a deep
        # copy of ``shapes`` in the preset config, so each preset is flagged separately.
        # Objects are lightened to the jaw's scale (the shared lift default is 0.2 kg)
        for name in ("shapes", "cube", "default", "ovphysx"):
            object_spawn = getattr(self.object.spawn, name)
            object_spawn.activate_contact_sensors = True
            object_spawn.mass_props.mass = 0.05
        self.object.spawn.cube.size = (0.03, 0.03, 0.03)
        self.object.spawn.ovphysx.size = (0.03, 0.03, 0.03)
        # Sense the jaw-object pair from the object side for filtered PhysX reporting.
        self.jaw_object_s = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1"],
        )
        # objects sized to the SO-101 jaw (~4 cm span at full open)
        graspable_shape_assets_cfg = [
            MeshCuboidCfg(size=(0.03, 0.03, 0.03), **lift.OBJECT_PHYSICS),
            MeshCuboidCfg(size=(0.02, 0.03, 0.03), **lift.OBJECT_PHYSICS),
            MeshCuboidCfg(size=(0.02, 0.02, 0.03), **lift.OBJECT_PHYSICS),
            MeshCuboidCfg(size=(0.01, 0.03, 0.03), **lift.OBJECT_PHYSICS),
            MeshSphereCfg(radius=0.015, **lift.OBJECT_PHYSICS),
            MeshCapsuleCfg(radius=0.015, height=0.05, **lift.OBJECT_PHYSICS),
            MeshCapsuleCfg(radius=0.01, height=0.08, **lift.OBJECT_PHYSICS),
        ]
        self.object.spawn.shapes.assets_cfg = graspable_shape_assets_cfg
        self.object.spawn.default.assets_cfg = graspable_shape_assets_cfg
        # spawn 40 mm above the tabletop (surface at z = 0.255), centered on the peak of the
        # pinch-feasibility map (the spot with the most valid approach orientations)
        self.object.init_state.pos = (-0.27, 0.2, 0.295)


@configclass
class SO101RelJointPosActionCfg:
    """Relative position actions for the SO-101 arm and gripper."""

    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class SO101StateObservationCfg(lift.ObservationsCfg):
    """State observations for the SO-101 lift task."""

    def __post_init__(self):
        super().__post_init__()
        self.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [THUMB_SENSOR]},
            clip=(-20.0, 20.0),  # jaw contact force stays well under 20 N
        )
        self.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = JAW_LIST


@configclass
class SO101LiftRewardCfg(lift.RewardsCfg):
    """Rewards for the SO-101 lift task."""

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
        if self.orientation_tracking:
            self.orientation_tracking.params["thumb_name"] = THUMB_SENSOR
            self.orientation_tracking.params["finger_names"] = FINGER_SENSORS
        self.success.params["thumb_name"] = THUMB_SENSOR
        self.success.params["finger_names"] = FINGER_SENSORS


@configclass
class SO101MixinCfg:
    """SO-101 scene and MDP settings for the lift task."""

    scene: SO101SceneCfg = SO101SceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)
    rewards: SO101LiftRewardCfg = SO101LiftRewardCfg()
    observations: SO101StateObservationCfg = SO101StateObservationCfg()
    actions: SO101RelJointPosActionCfg = SO101RelJointPosActionCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.body_name = "gripper"
        # goal workspace inside the dense reachable band (root frame, arm works at -y):
        # random-joint FK sampling puts the gripper's 25th-75th percentile envelope at
        # x (-0.08, 0.12), y (-0.24, -0.13), z (0.11, 0.30); goals beyond y=-0.31 are
        # practically unreachable and cap the achievable success rate
        self.commands.object_pose.ranges.pos_x = (-0.08, 0.08)
        self.commands.object_pose.ranges.pos_y = (-0.30, -0.16)
        self.commands.object_pose.ranges.pos_z = (0.08, 0.22)
        events = self.events.conditional_reset.params["terms"]
        events["reset_robot_wrist_joint"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names="wrist_roll")
        events["reset_robot_joints"].params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names="(shoulder_pan|shoulder_lift|elbow_flex|wrist_flex|gripper)"
        )
        events["reset_object_to_target"].params["target_cfg"] = SceneEntityCfg("robot", body_names="gripper")
        # in-gripper placement ~20 mm toward the wrist: mid-face contact is a necessary
        # condition for capture, tip-edge placements always slip
        events["reset_object_to_target"].params["pose_range"] = {
            "x": [-0.01, 0.03],
            "y": [-0.02, 0.02],
            "z": [-0.12, -0.08],
        }
        # tabletop spawn region (x along the arm's reach direction after the -90 deg base
        # yaw, y lateral): the band directly in front of the base where the 5-DOF arm can
        # present a horizontal pinch at table height; peripheral spots are reachable but
        # cannot pose a valid grasp orientation
        events["reset_object"].params["pose_range"] = {
            "x": [-0.03, 0.03],
            "y": [-0.03, 0.03],
            "z": [0.0, 0.005],
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
        self.events.conditional_reset.params["diversity_feature"].body_names = JAW_LIST
        # velocity-limit termination on the arm joints only: the jaw legitimately stalls on objects
        self.terminations.abnormal_robot.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names="(shoulder_pan|shoulder_lift|elbow_flex|wrist_flex|wrist_roll)"
        )
        # The tabletop starts at z=0.255 m, below the shared task's z=0.3 m cutoff.
        self.terminations.object_out_of_bound.params["in_bound_range"]["z"] = (0.20, 2.0)
        # Keep generic gain randomization off the USD-calibrated jaw drive.
        self.events.joint_stiffness_and_damping.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names="(shoulder_pan|shoulder_lift|elbow_flex|wrist_flex|wrist_roll)"
        )
        # size the inertia randomization to the palm-sized objects: the shared lift default adds
        # 0.01 kg*m^2, three orders of magnitude above these objects' natural inertia, which
        # gyroscopically freezes their rotation and fights reorienting a held object
        self.events.object_physics_inertia.params["inertia_distribution_params"] = (0.0002, 0.0002)


@configclass
class SO101LiftEnvCfg(SO101MixinCfg, lift.LiftEnvCfg):
    """SO-101 object lifting environment."""
