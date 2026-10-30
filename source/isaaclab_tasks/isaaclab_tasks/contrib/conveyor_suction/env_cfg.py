# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One shared Franka policy sorts variable cuboids into five physical size bins."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import mdp as common_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import ResetStateSamplerCfg
from isaaclab_tasks.utils.sampling import SamplerCfg, UniformSamplingStrategyCfg

from ..conveyor_franka.conveyor_franka_env_cfg import ConveyorForceCfg, ConveyorFrankaEnvCfg
from ..conveyor_franka.mdp import action_term_l2, finite_action_rate_l2, finite_joint_velocity_l2
from ..conveyor_franka.mdp.rewards import transfer_success_reward
from ..conveyor_franka.mdp.terminations import invalid_action as invalid_policy_action
from . import mdp
from .actions_cfg import TOOL_OFFSET, CartesianSuctionArmCfg, VacuumActionCfg
from .scene import SuctionSortingSceneCfg, spawn_suction_franka

_ARM = [f"panda_joint{index}" for index in range(1, 8)]


@configclass
class ActionsCfg:
    arm_action = CartesianSuctionArmCfg(
        asset_name="robot",
        joint_names=_ARM,
        body_name="panda_hand",
        body_offset=CartesianSuctionArmCfg.OffsetCfg(pos=TOOL_OFFSET),
        scale=0.025,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    )
    gripper_action = VacuumActionCfg()


@configclass
class CommandsCfg:
    sort = mdp.SizeSortCommandCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joints = ObservationTermCfg(
            func=common_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM)}
        )
        joint_velocities = ObservationTermCfg(
            func=common_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM)}
        )
        parcel = ObservationTermCfg(func=mdp.parcel_observation)
        tool = ObservationTermCfg(func=mdp.tool_observation)
        suction = ObservationTermCfg(func=mdp.suction_observation)
        actions = ObservationTermCfg(func=common_mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_parcels = EventTerm(
        func=mdp.ResetParcels, mode="reset", params={"sampling": ResetStateSamplerCfg(), "full_starts": False}
    )


@configclass
class RewardsCfg:
    delivery = RewardTermCfg(func=transfer_success_reward, params={"command_name": "sort"}, weight=600.0)
    failure = RewardTermCfg(func=mdp.failed_delivery, weight=-60.0)
    effort = RewardTermCfg(func=action_term_l2, params={"action_name": "arm_action"}, weight=-0.001)
    action_rate = RewardTermCfg(
        func=finite_action_rate_l2, params={"action_names": ("arm_action", "gripper_action")}, weight=-0.001
    )
    joint_velocity = RewardTermCfg(
        func=finite_joint_velocity_l2, params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM)}, weight=-0.0001
    )


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=common_mdp.time_out, time_out=True)
    batch_complete = TerminationTermCfg(func=mdp.batch_complete)
    failure = TerminationTermCfg(func=mdp.failed_delivery)
    invalid_action = TerminationTermCfg(
        func=invalid_policy_action, params={"action_names": ("arm_action", "gripper_action")}
    )


@configclass
class SuctionSortingEnvCfg(ConveyorFrankaEnvCfg):
    class_type: type | str = "{DIR}.env:SuctionSortingEnv"
    scene: SuctionSortingSceneCfg = SuctionSortingSceneCfg(num_envs=512, env_spacing=8.0)
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum = None
    episode_length_s: float = 240.0
    conveyor_force: ConveyorForceCfg = ConveyorForceCfg(
        speed=0.25,
        friction=0.6,
        transported_body_pattern=r"(?:^|/)Cube[0-9]+(?:/|$)",
        transported_body_count_per_env=12,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot.spawn.func = spawn_suction_franka
        self.sim.physics.solver_cfg.nconmax = 400
        self.sim.physics.solver_cfg.njmax = 700
        self.sim.physics.collision_cfg.include_static_kinematic_pairs = False
        self.sim.visualizer_cfgs = VisualizerCfg(
            eye=(-2.6, -4.2, 2.8),
            lookat=(0.8, 0.2, -0.25),
            focal_length=32.0,
            max_visible_envs=1,
            randomly_sample_visible_envs=False,
        )

    def play_mode(self) -> None:
        self.scene.num_envs = min(self.scene.num_envs, 4)
        self.events.reset_parcels.params["full_starts"] = True
        self.events.reset_parcels.params["sampling"].sampling = SamplerCfg(strategies=[UniformSamplingStrategyCfg()])

    def validate_config(self) -> None:
        """Require the force-based suction and conveyor contact path."""
        physics = self.sim.physics
        if not isinstance(physics, NewtonCfg) or not isinstance(physics.solver_cfg, MJWarpSolverCfg):
            raise ValueError("Suction sorting requires Newton MJWarp physics.")
        if physics.solver_cfg.use_mujoco_contacts or physics.collision_cfg is None:
            raise ValueError("Suction sorting requires Newton's external collision pipeline.")
