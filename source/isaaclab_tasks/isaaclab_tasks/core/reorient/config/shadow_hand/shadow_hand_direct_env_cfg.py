# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_common import (
    CUBE_CFG,
    GOAL_OBJECT_CFG,
    SHADOW_HAND_ROBOT_CFG,
    PhysicsCfg,
    ShadowHandEventCfg,
    ShadowHandRobotCfg,
)
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.shadow_hand import SHADOW_ACTUATED_JOINT_NAMES, SHADOW_FINGERTIP_BODY_NAMES

FULL_OBSERVATION_SPACE = 157
"""Size of the full state observation the actor reads by default."""
REDUCED_OBSERVATION_SPACE = 42
"""Size of the actor observation under ``presets=openai``."""
ASYMMETRIC_STATE_SPACE = 187
"""Size of the privileged critic observation when :attr:`asymmetric_obs` is set."""


@configclass
class ShadowHandDomainRandomizationCfg(PresetCfg):
    """``presets=randomized`` enables the same randomization the manager tasks apply."""

    randomized = ShadowHandEventCfg()
    default = None


@configclass
class ShadowHandEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = 20
    observation_space = FULL_OBSERVATION_SPACE
    state_space = 0
    asymmetric_obs = False
    reduced_obs = False
    """Narrow the actor to the quantities a physical hand can estimate, leaving the critic unchanged."""

    # ``presets=randomized`` enables domain randomization
    events: ShadowHandDomainRandomizationCfg = ShadowHandDomainRandomizationCfg()

    # simulation — values mirrored by the manager cfg (guarded by the value-parity test)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
    )

    # robot
    robot_cfg: ShadowHandRobotCfg = SHADOW_HAND_ROBOT_CFG
    actuated_joint_names = SHADOW_ACTUATED_JOINT_NAMES
    fingertip_body_names = SHADOW_FINGERTIP_BODY_NAMES

    # in-hand object
    object_cfg: RigidObjectCfg = CUBE_CFG
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = GOAL_OBJECT_CFG
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=0.75, replicate_physics=True)

    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250.0
    fall_penalty = 0.0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.1
    max_consecutive_success = 0
    success_count_threshold: int = 1
    """Minimum number of goals reached in an episode to count it as a successful episode."""
    in_hand_pos_offset: tuple[float, float, float] = (0.0, 0.0, -0.04)
    """In-hand goal anchor, relative to the object's default position [m]."""
    goal_marker_position: tuple[float, float, float] = (-0.2, -0.45, 0.68)
    """Fixed goal-marker display position [m], environment frame."""
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0

    def validate_config(self):
        """Check that the declared observation sizes match the selected observation terms.

        A mismatch otherwise surfaces as a tensor-shape error inside the policy, far from
        the configuration that caused it.
        """
        expected_obs = REDUCED_OBSERVATION_SPACE if self.reduced_obs else FULL_OBSERVATION_SPACE
        if self.observation_space != expected_obs:
            raise ValueError(
                f"'observation_space' is {self.observation_space}, but 'reduced_obs={self.reduced_obs}'"
                f" produces {expected_obs} values. Select 'presets=openai' rather than setting the"
                " observation flags individually."
            )
        expected_state = ASYMMETRIC_STATE_SPACE if self.asymmetric_obs else 0
        if self.state_space != expected_state:
            raise ValueError(
                f"'state_space' is {self.state_space}, but 'asymmetric_obs={self.asymmetric_obs}'"
                f" produces {expected_state} values. Select 'presets=openai' rather than setting the"
                " observation flags individually."
            )


@configclass
class ShadowHandOpenAIObsEnvCfg(ShadowHandEnvCfg):
    """The observation architecture of `Learning Dexterous In-Hand Manipulation`_.

    The actor sees only what a physical hand can measure, while a privileged critic reads
    the full simulator state. The training regime that paper pairs this with lives in
    ``IsaacContrib-Reorient-Cube-Shadow-OpenAI``.

    .. _Learning Dexterous In-Hand Manipulation: https://arxiv.org/pdf/1808.00177.pdf
    """

    observation_space = REDUCED_OBSERVATION_SPACE
    state_space = ASYMMETRIC_STATE_SPACE
    asymmetric_obs = True
    reduced_obs = True


@configclass
class ShadowHandDirectEnvCfg(PresetCfg):
    """``presets=openai`` swaps in the reduced actor and its privileged critic."""

    openai = ShadowHandOpenAIObsEnvCfg()
    default = ShadowHandEnvCfg()
