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
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import MassCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.contrib.lift.mdp.observations import object_position_in_robot_root_frame
from isaaclab_tasks.contrib.lift.mdp.rewards import object_ee_distance
from isaaclab_tasks.contrib.stack.mdp.observations import ee_frame_quat
from isaaclab_tasks.utils import preset

from isaaclab_assets.robots.so101 import SO101_CFG

from ... import lift_env_cfg as lift
from . import mdp as so101_mdp


@configclass
class SO101SceneCfg(lift.SceneCfg):
    """SO-101 mounted on the shared table with a 3 cm, 50 g cube."""

    robot: ArticulationCfg = SO101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=SO101_CFG.spawn.replace(
            activate_contact_sensors=False,
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
        spawn=lift.ObjectCfg().cube.replace(size=(0.03, 0.03, 0.03), mass_props=MassCfg(mass=0.05)),
        # Start 2 mm above the resting center height to avoid penetration on reset.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.32, 0.2, 0.272)),
    )
    table: RigidObjectCfg = lift.SceneCfg().table.replace(spawn=lift.TABLE_SPAWN_CFG.replace(visible=True))
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
        """The same 32 state values are available to actor and critic."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        object_pos_b = ObsTerm(func=object_position_in_robot_root_frame)
        object_quat_w = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        gripper_to_object_b = ObsTerm(func=so101_mdp.gripper_to_object_b)
        gripper_quat_w = ObsTerm(func=ee_frame_quat, params={"ee_frame_cfg": SceneEntityCfg("grasp_frame")})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SO101JointPosActionCfg:
    """Absolute arm targets about home and an analog jaw target in [0, 1] rad."""

    arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        scale=0.7,
        use_default_offset=True,
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
class SO101LiftRewardCfg:
    """Reach, lift, and avoid unnecessary changes in joint targets."""

    reach = RewTerm(
        func=object_ee_distance, weight=1.0, params={"std": 0.06, "ee_frame_cfg": SceneEntityCfg("grasp_frame")}
    )
    lift = RewTerm(
        func=so101_mdp.LiftReward, weight=10.0, params={"resting_height": 0.27, "lift_height": 0.08, "speed_std": 0.5}
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class SO101TerminationCfg:
    """End an episode at six seconds or when the cube falls off the table."""

    timeout = DoneTerm(func=mdp.time_out, time_out=True)
    dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.20, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class SO101LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Lift a tabletop cube by 5 cm and hold it under full gravity.

    Episodes last 6 s. Each reset samples the cube within a 5 cm square with yaw in
    [-45, 45] degrees and returns the arm to an open-gripper home pose. Actor and critic
    receive the same 32 state values; there are no cameras, point clouds, or history.

    Five actions command arm joint offsets about home, and one commands the jaw angle.
    Rewards encourage reaching the grasp frame to the cube, lifting it by 8 cm while
    slowing its motion, and keeping successive actions smooth. Success is measured
    independently as at least 5 cm of lift held for 0.5 s with object speed below 0.2 m/s.
    No command generator, physics randomization, reset bank, or curriculum is used.
    """

    scene: SO101SceneCfg = SO101SceneCfg(num_envs=2048, env_spacing=2.0, replicate_physics=True)
    observations: SO101StateObservationCfg = SO101StateObservationCfg()
    actions: SO101JointPosActionCfg = SO101JointPosActionCfg()
    events: SO101EventCfg = SO101EventCfg()
    rewards: SO101LiftRewardCfg = SO101LiftRewardCfg()
    terminations: SO101TerminationCfg = SO101TerminationCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 6.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = lift.PhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(-0.8, -0.25, 0.6), lookat=(-0.29, 0.2, 0.33))
