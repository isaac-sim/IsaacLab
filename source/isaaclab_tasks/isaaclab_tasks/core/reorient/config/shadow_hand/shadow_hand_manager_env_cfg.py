# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based counterpart of the state-based Shadow Hand reorientation task."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.reorient.mdp as mdp
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_env_cfg import (
    GOAL_OBJECT_CFG,
    OBJECT_CFG,
    OPENAI_ACTION_NOISE_CFG,
    OPENAI_OBSERVATION_NOISE_CFG,
    ROBOT_CFG,
    NewtonEventCfg,
    ObjectCfg,
    PhysicsCfg,
    PhysxEventCfg,
)
from isaaclab_tasks.core.reorient.reorient_task_constants import (
    GOAL_MARKER_POSITION,
    OPENAI_ACT_MOVING_AVERAGE,
    OPENAI_ACTION_PENALTY_SCALE,
    OPENAI_AV_FACTOR,
    OPENAI_DECIMATION,
    OPENAI_DIST_REWARD_SCALE,
    OPENAI_EPISODE_LENGTH_S,
    OPENAI_FALL_DIST,
    OPENAI_FALL_PENALTY,
    OPENAI_FORCE_TORQUE_OBS_SCALE,
    OPENAI_MAX_CONSECUTIVE_SUCCESS,
    OPENAI_REACH_GOAL_BONUS,
    OPENAI_RESET_DOF_POS_NOISE,
    OPENAI_RESET_DOF_VEL_NOISE,
    OPENAI_RESET_POSITION_NOISE,
    OPENAI_ROT_EPS,
    OPENAI_ROT_REWARD_SCALE,
    OPENAI_SIM_DT,
    OPENAI_SUCCESS_COUNT_THRESHOLD,
    OPENAI_SUCCESS_TOLERANCE,
    SHADOW_ACT_MOVING_AVERAGE,
    SHADOW_ACTION_PENALTY_SCALE,
    SHADOW_ACTUATED_JOINT_NAMES,
    SHADOW_AV_FACTOR,
    SHADOW_DECIMATION,
    SHADOW_DIST_REWARD_SCALE,
    SHADOW_EPISODE_LENGTH_S,
    SHADOW_FALL_DIST,
    SHADOW_FALL_PENALTY,
    SHADOW_FINGERTIP_BODY_NAMES,
    SHADOW_REACH_GOAL_BONUS,
    SHADOW_RESET_DOF_POS_NOISE,
    SHADOW_RESET_DOF_VEL_NOISE,
    SHADOW_RESET_POSITION_NOISE,
    SHADOW_ROT_EPS,
    SHADOW_ROT_REWARD_SCALE,
    SHADOW_SUCCESS_COUNT_THRESHOLD,
    SHADOW_SUCCESS_TOLERANCE,
    SHADOW_VEL_OBS_SCALE,
)
from isaaclab_tasks.utils import PresetCfg


@configclass
class _ShadowHandManagerSceneCfg(InteractiveSceneCfg):
    """Scene shared by the Shadow Hand Manager backend alternatives."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: PresetCfg = ROBOT_CFG
    object: ObjectCfg = OBJECT_CFG
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class ShadowHandManagerSceneCfg(PresetCfg):
    """Backend-specific scene cloning settings matching the Direct task."""

    physx = _ShadowHandManagerSceneCfg(num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=True)
    newton_mjwarp = _ShadowHandManagerSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=False
    )
    ovphysx = physx
    newton_kamino = newton_mjwarp
    default = physx


@configclass
class CommandsCfg:
    """Object pose goal matching the Direct in-hand target."""

    object_pose = mdp.ReorientEpisodeCommandCfg(
        asset_name="object",
        init_pos_offset=(0.0, 0.0, -0.04),
        update_goal_on_success=True,
        orientation_success_threshold=SHADOW_SUCCESS_TOLERANCE,
        make_quat_unique=False,
        fixed_marker_pos=GOAL_MARKER_POSITION,
        goal_pose_visualizer_cfg=GOAL_OBJECT_CFG,
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Twenty actuated Shadow Hand joints."""

    joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=SHADOW_ACTUATED_JOINT_NAMES,
        alpha=SHADOW_ACT_MOVING_AVERAGE,
        rescale_to_limits=True,
    )


@configclass
class FullStateWithoutActionCfg(ObsGroup):
    """Shared first 137 dimensions of the full Shadow state."""

    joint_pos = ObsTerm(
        func=mdp.joint_pos_limit_normalized,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=False)},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        scale=SHADOW_VEL_OBS_SCALE,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=False)},
    )
    object_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
    object_quat = ObsTerm(
        func=mdp.root_quat_w,
        params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False},
    )
    object_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("object")})
    object_ang_vel = ObsTerm(
        func=mdp.root_ang_vel_w,
        scale=SHADOW_VEL_OBS_SCALE,
        params={"asset_cfg": SceneEntityCfg("object")},
    )
    goal_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
    goal_quat_diff = ObsTerm(
        func=mdp.goal_quat_diff,
        params={"asset_cfg": SceneEntityCfg("object"), "command_name": "object_pose", "make_quat_unique": False},
    )
    fingertip_pos = ObsTerm(
        func=mdp.fingertip_pos,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=SHADOW_FINGERTIP_BODY_NAMES, preserve_order=False)},
    )
    fingertip_quat = ObsTerm(
        func=mdp.fingertip_quat,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=SHADOW_FINGERTIP_BODY_NAMES, preserve_order=False)},
    )
    fingertip_vel = ObsTerm(
        func=mdp.fingertip_vel,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=SHADOW_FINGERTIP_BODY_NAMES, preserve_order=False)},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    """Full 157-dimensional state observation in Direct order."""

    @configclass
    class PolicyCfg(FullStateWithoutActionCfg):
        last_action = ObsTerm(func=mdp.reorient_last_action, params={"action_name": "joint_pos"})

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset distributions matching the Direct task."""

    reset_state = EventTerm(
        func=mdp.reset_reorient_state,
        mode="reset",
        params={
            "position_noise": SHADOW_RESET_POSITION_NOISE,
            "joint_position_noise": SHADOW_RESET_DOF_POS_NOISE,
            "joint_velocity_noise": SHADOW_RESET_DOF_VEL_NOISE,
            "action_name": "joint_pos",
        },
    )


@configclass
class RewardsCfg:
    """Direct-compatible reward and success accounting."""

    reorient = RewTerm(
        func=mdp.DirectReorientReward,
        weight=1.0,
        params={
            "command_name": "object_pose",
            "distance_scale": SHADOW_DIST_REWARD_SCALE,
            "rotation_scale": SHADOW_ROT_REWARD_SCALE,
            "rotation_epsilon": SHADOW_ROT_EPS,
            "action_penalty_scale": SHADOW_ACTION_PENALTY_SCALE,
            "success_tolerance": SHADOW_SUCCESS_TOLERANCE,
            "success_bonus": SHADOW_REACH_GOAL_BONUS,
            "fall_distance": SHADOW_FALL_DIST,
            "fall_penalty": SHADOW_FALL_PENALTY,
            "averaging_factor": SHADOW_AV_FACTOR,
            "success_count_threshold": SHADOW_SUCCESS_COUNT_THRESHOLD,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination conditions matching the Direct task."""

    object_out_of_reach = DoneTerm(
        func=mdp.object_reorientation_out_of_reach,
        params={
            "threshold": SHADOW_FALL_DIST,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class ShadowHandManagerEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based state Shadow Hand task with Direct-compatible semantics."""

    scene: ShadowHandManagerSceneCfg = ShadowHandManagerSceneCfg()
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = SHADOW_DECIMATION
        self.episode_length_s = SHADOW_EPISODE_LENGTH_S
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 2.0)


@configclass
class OpenAICommandsCfg:
    """OpenAI goal command with its wider success tolerance."""

    object_pose = mdp.ReorientEpisodeCommandCfg(
        asset_name="object",
        init_pos_offset=(0.0, 0.0, -0.04),
        update_goal_on_success=True,
        orientation_success_threshold=OPENAI_SUCCESS_TOLERANCE,
        make_quat_unique=False,
        fixed_marker_pos=GOAL_MARKER_POSITION,
        goal_pose_visualizer_cfg=GOAL_OBJECT_CFG,
        debug_vis=True,
    )


@configclass
class OpenAIActionsCfg:
    """OpenAI actions with Direct-compatible EMA and stateful noise."""

    joint_pos = mdp.NoisyEMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=SHADOW_ACTUATED_JOINT_NAMES,
        alpha=OPENAI_ACT_MOVING_AVERAGE,
        rescale_to_limits=True,
        noise_model=OPENAI_ACTION_NOISE_CFG,
    )


@configclass
class OpenAIObservationsCfg:
    """OpenAI 42-dimensional actor and 187-dimensional critic observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        openai = ObsTerm(
            func=mdp.OpenAIPolicyObservation,
            params={
                "command_name": "object_pose",
                "action_name": "joint_pos",
                "noise_model": OPENAI_OBSERVATION_NOISE_CFG,
                "robot_cfg": SceneEntityCfg("robot", body_names=SHADOW_FINGERTIP_BODY_NAMES, preserve_order=False),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(FullStateWithoutActionCfg):
        fingertip_wrench = ObsTerm(
            func=mdp.fingertip_wrench,
            scale=OPENAI_FORCE_TORQUE_OBS_SCALE,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "joint_wrench", body_names=SHADOW_FINGERTIP_BODY_NAMES, preserve_order=False
                )
            },
        )
        last_action = ObsTerm(func=mdp.reorient_last_action, params={"action_name": "joint_pos"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class _ShadowHandOpenAIManagerSceneCfg(_ShadowHandManagerSceneCfg):
    """Shadow Hand scene with fingertip joint-wrench sensing."""

    joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ShadowHandOpenAIManagerSceneCfg(PresetCfg):
    """Backend-specific OpenAI scene alternatives."""

    physx = _ShadowHandOpenAIManagerSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=True
    )
    newton_mjwarp = _ShadowHandOpenAIManagerSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=False
    )
    ovphysx = physx
    newton_kamino = newton_mjwarp
    default = physx


_OPENAI_RESET_PARAMS = {
    "position_noise": OPENAI_RESET_POSITION_NOISE,
    "joint_position_noise": OPENAI_RESET_DOF_POS_NOISE,
    "joint_velocity_noise": OPENAI_RESET_DOF_VEL_NOISE,
    "action_name": "joint_pos",
}


@configclass
class OpenAIPhysxEventCfg(PhysxEventCfg):
    """PhysX OpenAI randomization and state reset events."""

    reset_state = EventTerm(func=mdp.reset_reorient_state, mode="reset", params=_OPENAI_RESET_PARAMS)


@configclass
class OpenAINewtonEventCfg(NewtonEventCfg):
    """Newton OpenAI randomization and state reset events."""

    reset_state = EventTerm(func=mdp.reset_reorient_state, mode="reset", params=_OPENAI_RESET_PARAMS)


@configclass
class OpenAIEventCfg(PresetCfg):
    """Backend-specific OpenAI event alternatives."""

    physx = OpenAIPhysxEventCfg()
    newton_mjwarp = OpenAINewtonEventCfg()
    ovphysx = physx
    newton_kamino = newton_mjwarp
    default = physx


@configclass
class OpenAIRewardsCfg:
    """Direct-compatible OpenAI reward and success accounting."""

    reorient = RewTerm(
        func=mdp.DirectReorientReward,
        weight=1.0,
        params={
            "command_name": "object_pose",
            "distance_scale": OPENAI_DIST_REWARD_SCALE,
            "rotation_scale": OPENAI_ROT_REWARD_SCALE,
            "rotation_epsilon": OPENAI_ROT_EPS,
            "action_penalty_scale": OPENAI_ACTION_PENALTY_SCALE,
            "success_tolerance": OPENAI_SUCCESS_TOLERANCE,
            "success_bonus": OPENAI_REACH_GOAL_BONUS,
            "fall_distance": OPENAI_FALL_DIST,
            "fall_penalty": OPENAI_FALL_PENALTY,
            "averaging_factor": OPENAI_AV_FACTOR,
            "success_count_threshold": OPENAI_SUCCESS_COUNT_THRESHOLD,
            "action_name": "joint_pos",
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class OpenAITerminationsCfg:
    """Direct-compatible OpenAI termination conditions."""

    object_out_of_reach = DoneTerm(
        func=mdp.object_reorientation_out_of_reach,
        params={
            "threshold": OPENAI_FALL_DIST,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    time_out = DoneTerm(
        func=mdp.DirectReorientTimeout,
        time_out=True,
        params={
            "command_name": "object_pose",
            "reward_name": "reorient",
            "success_tolerance": OPENAI_SUCCESS_TOLERANCE,
            "max_successes": OPENAI_MAX_CONSECUTIVE_SUCCESS,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class ShadowHandOpenAIManagerEnvCfg(ShadowHandManagerEnvCfg):
    """Manager counterpart shared by the OpenAI FF and LSTM variants."""

    scene: ShadowHandOpenAIManagerSceneCfg = ShadowHandOpenAIManagerSceneCfg()
    observations: OpenAIObservationsCfg = OpenAIObservationsCfg()
    actions: OpenAIActionsCfg = OpenAIActionsCfg()
    commands: OpenAICommandsCfg = OpenAICommandsCfg()
    rewards: OpenAIRewardsCfg = OpenAIRewardsCfg()
    terminations: OpenAITerminationsCfg = OpenAITerminationsCfg()
    events: OpenAIEventCfg = OpenAIEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = OPENAI_DECIMATION
        self.episode_length_s = OPENAI_EPISODE_LENGTH_S
        self.sim.dt = OPENAI_SIM_DT
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
