# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import dataclass
from typing import Any

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import config_field
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.contrib.navigation.mdp as mdp
from isaaclab_tasks.contrib.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg

LOW_LEVEL_ENV_CFG = AnymalCFlatEnvCfg()


@dataclass
class EventCfg:
    """Configuration for events."""

    reset_base: Any = config_field(
        EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (-0.0, 0.0),
                    "y": (-0.0, 0.0),
                    "z": (-0.0, 0.0),
                    "roll": (-0.0, 0.0),
                    "pitch": (-0.0, 0.0),
                    "yaw": (-0.0, 0.0),
                },
            },
        )
    )


@dataclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = config_field(
        mdp.PreTrainedPolicyActionCfg(
            asset_name="robot",
            policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt",
            low_level_decimation=4,
            low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
            low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
        )
    )


@dataclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @dataclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel: Any = config_field(ObsTerm(func=mdp.base_lin_vel))
        projected_gravity: Any = config_field(ObsTerm(func=mdp.projected_gravity))
        pose_command: Any = config_field(ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"}))

    # observation groups
    policy: PolicyCfg = config_field(PolicyCfg())


@dataclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty: Any = config_field(RewTerm(func=mdp.is_terminated, weight=-400.0))
    position_tracking: Any = config_field(
        RewTerm(
            func=mdp.position_command_error_tanh,
            weight=0.5,
            params={"std": 2.0, "command_name": "pose_command"},
        )
    )
    position_tracking_fine_grained: Any = config_field(
        RewTerm(
            func=mdp.position_command_error_tanh,
            weight=0.5,
            params={"std": 0.2, "command_name": "pose_command"},
        )
    )
    orientation_tracking: Any = config_field(
        RewTerm(
            func=mdp.heading_command_error_abs,
            weight=-0.2,
            params={"command_name": "pose_command"},
        )
    )


@dataclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command: Any = config_field(
        mdp.UniformPose2dCommandCfg(
            asset_name="robot",
            simple_heading=False,
            resampling_time_range=(8.0, 8.0),
            debug_vis=True,
            position_success_threshold=0.5,
            ranges=mdp.UniformPose2dCommandCfg.Ranges(
                pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)
            ),
        )
    )


@dataclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: Any = config_field(DoneTerm(func=mdp.time_out, time_out=True))
    base_contact: Any = config_field(
        DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
        )
    )


@dataclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # environment settings
    scene: SceneEntityCfg = config_field(LOW_LEVEL_ENV_CFG.scene)
    actions: ActionsCfg = config_field(ActionsCfg())
    observations: ObservationsCfg = config_field(ObservationsCfg())
    events: EventCfg = config_field(EventCfg())
    # mdp settings
    commands: CommandsCfg = config_field(CommandsCfg())
    rewards: RewardsCfg = config_field(RewardsCfg())
    terminations: TerminationsCfg = config_field(TerminationsCfg())

    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
