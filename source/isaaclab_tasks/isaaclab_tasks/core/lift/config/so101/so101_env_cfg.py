# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State-only SO-101 cube lifting with fresh tabletop resets and full gravity."""

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import MassCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.contrib.lift.mdp.observations import object_position_in_robot_root_frame
from isaaclab_tasks.contrib.stack.mdp.observations import ee_frame_quat
from isaaclab_tasks.utils import preset

from isaaclab_assets.robots.so101 import SO101_CFG

from ... import lift_env_cfg as lift
from ... import mdp as lift_mdp


@configclass
class SO101SceneCfg(lift.SceneCfg):
    """SO-101 mounted on the shared table with a 3 cm, 50 g cube."""

    robot: ArticulationCfg = SO101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=SO101_CFG.spawn.replace(
            activate_contact_sensors=True,
            variants={
                "Robot": "robot",
                "Sensor": "sensors",
                "Physics": preset(
                    default="physics", isaacsim_physx="physx", physx="physx", ovphysx="physx", newton_mjwarp="physics"
                ),
            },
        ),
        init_state=SO101_CFG.init_state.replace(
            # The asset root is 3.008 cm above the clamp foot; the tabletop is at z=0.255 m.
            pos=(-0.16, 0.2, 0.22492),
            rot=(0.0, 0.0, -0.70710678, 0.70710678),
            joint_pos={
                "shoulder_pan": 0.15,
                "shoulder_lift": -0.5,
                "elbow_flex": 0.6,
                "wrist_flex": 1.3,
                "wrist_roll": 0.0,
                "gripper": 0.8,
            },
        ),
    )
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=lift.ObjectCfg().cube.replace(
            size=(0.03, 0.03, 0.03), mass_props=MassCfg(mass=0.05), activate_contact_sensors=True
        ),
        # Start 2 mm above the resting center height to avoid penetration on reset.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.43, 0.2, 0.272)),
    )
    table: RigidObjectCfg = lift.SceneCfg().table.replace(spawn=lift.TABLE_SPAWN_CFG.replace(visible=True))
    # Object-side sensing supports filtered moving-jaw contact on PhysX as well as Newton.
    jaw_object_s: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1"],
    )
    grasp_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper",
                name="grasp",
                # Between the finger collision surfaces, near their tips.
                offset=OffsetCfg(pos=(0.006, 0.0, -0.095)),
            )
        ],
    )


@configclass
class SO101StateObservationCfg:
    """Current robot and object state, without cameras, point clouds, or history."""

    @configclass
    class PolicyCfg(ObsGroup):
        """The same 39 state and command values are available to actor and critic."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        object_pos_b = ObsTerm(func=object_position_in_robot_root_frame)
        object_quat_w = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        gripper_to_object_b = ObsTerm(
            func=lift_mdp.ee_to_object_b, params={"ee_frame_cfg": SceneEntityCfg("grasp_frame")}
        )
        gripper_quat_w = ObsTerm(func=ee_frame_quat, params={"ee_frame_cfg": SceneEntityCfg("grasp_frame")})
        target_object_pose_b = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SO101JointPosActionCfg:
    """Arm targets within joint limits and an analog jaw target in [0, 1] rad."""

    arm = mdp.JointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
    )
    gripper = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        scale=0.5,
        offset=0.5,
        use_default_offset=False,
        clip={".*": (0.0, 1.0)},
    )


@configclass
class SO101EventCfg:
    """Reset to a safe home pose and sample the cube on the tabletop."""

    reset = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "yaw": (-0.785398, 0.785398)},
            "velocity_range": {},
        },
    )


@configclass
class SO101LiftRewardCfg(lift.RewardsCfg):
    """Shared lift rewards with the SO-101 jaw contact binding."""

    action_l2 = None
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    orientation_tracking = None
    early_termination = None
    # One moving jaw provides the grasp-contact signal; a second contact-count term duplicates it.
    good_finger_contact = RewTerm(
        func=lift_mdp.contacts,
        weight=0.75,
        params={"threshold": 0.01, "thumb_name": "jaw_object_s", "finger_names": ["jaw_object_s"]},
    )

    def __post_init__(self):
        super().__post_init__()
        # Make transport competitive with maintaining jaw contact on the table.
        self.position_tracking.weight = 50.0
        self.success.weight = 50.0
        self.fingers_to_object.weight = 5.0
        self.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["gripper", "moving_jaw_so101_v1"]
        )
        for term in (self.fingers_to_object, self.position_tracking, self.success):
            term.params["thumb_name"] = "jaw_object_s"
            term.params["finger_names"] = ["jaw_object_s"]
        self.success.params["rot_std"] = None


@configclass
class SO101TerminationCfg:
    """End an episode at its time limit or when the cube falls off the table."""

    timeout = DoneTerm(func=mdp.time_out, time_out=True)
    dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.20, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class SO101LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Lift a tabletop cube to a commanded position under full gravity.

    Episodes last 12 s. Each reset samples the cube within a 5 cm square centered
    27 cm in front of the robot base, with yaw in
    [-45, 45] degrees and returns the arm to an open-gripper home pose. Actor and critic
    receive the same 39 state and command values; there are no cameras, point clouds, or history.

    Five actions command arm targets within joint limits, and one commands the jaw angle.
    The command generator and contact-gated position-progress and success rewards are
    shared with Franka and Kuka lift. Target positions are sampled in the robot root
    frame within its reach. Orientation tracking is disabled, with no hold-duration
    requirement, physics randomization, reset bank, or curriculum.

    A target is reached when object-to-target distance is below 5 cm while moving-jaw
    contact exceeds 0.01 N. The shared reward's ``succeeded`` flag means at least one
    target was reached during the episode; per-command success must be evaluated
    separately over each command's 4-6 s interval.
    """

    scene: SO101SceneCfg = SO101SceneCfg(num_envs=2048, env_spacing=2.0, replicate_physics=True)
    observations: SO101StateObservationCfg = SO101StateObservationCfg()
    actions: SO101JointPosActionCfg = SO101JointPosActionCfg()
    events: SO101EventCfg = SO101EventCfg()
    commands: lift.CommandsCfg = lift.CommandsCfg()
    rewards: SO101LiftRewardCfg = SO101LiftRewardCfg()
    terminations: SO101TerminationCfg = SO101TerminationCfg()

    def __post_init__(self):
        self.commands.object_pose.position_only = True
        self.commands.object_pose.ranges.pos_x = (-0.025, 0.025)
        # Keep both the cube resets and commanded lifts away from the base.
        self.commands.object_pose.ranges.pos_y = (-0.295, -0.245)
        self.commands.object_pose.ranges.pos_z = (0.15, 0.20)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)
        self.commands.object_pose.success_vis_asset_name = ""
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = lift.PhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(-0.8, -0.25, 0.6), lookat=(-0.29, 0.2, 0.33))
