# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One policy for flat walking and recovery, trained from scratch with body collisions."""

from isaaclab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.utils import configclass

from isaaclab_assets import MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG

from isaaclab_tasks.utils import resolve_presets

from .flat_env_cfg import MICRODUCK_JOINT_NAMES, MicroDuckVelocityFlatEnvCfg
from .mdp import recovery as recovery_mdp


@configclass
class MicroDuckRecoveryVelocityEnvCfg(MicroDuckVelocityFlatEnvCfg):
    """Flat velocity tracking with a performance-gated fall-recovery curriculum."""

    recovery: recovery_mdp.MicroDuckRecoveryCfg = recovery_mdp.MicroDuckRecoveryCfg()

    def __post_init__(self):
        super().__post_init__()
        resolve_presets(self, selected=("backlash",))
        self.scene.robot = MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 16384
        self.sim.physics.solver_cfg.nconmax = 128
        self.sim.physics.solver_cfg.njmax = 512
        self.observations.critic.recovery = ObservationTermCfg(func=recovery_mdp.recovery_observation)
        self.terminations.fell_over = TerminationTermCfg(func=recovery_mdp.recovery_failed)
        self.events.reset_base = EventTermCfg(
            func=recovery_mdp.reset_recovery,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)},
        )
        self.events.reset_robot_joints = None
        # Performance, rather than iteration count, controls this task's difficulty.
        self.curriculum = {"recovery": CurriculumTermCfg(func=recovery_mdp.recovery_curriculum)}
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.head_pose.ranges = ((0.0, 0.0),) * 4
        self.commands.body_pose.ranges = ((0.0, 0.0),) * 6
        self.rewards.upright = RewardTermCfg(func=recovery_mdp.recovery_upright, weight=2.0)
        self.rewards.fall = RewardTermCfg(func=recovery_mdp.recovery_fall_cost, weight=-1.0)
        # The head-bias integrator would retain errors from using the head as ground support.
        self.rewards.head_pose_bias = None
        self.rewards.angular_momentum.weight = -0.002
        for name in (
            "track_lin_vel",
            "track_ang_vel",
            "pose",
            "body_ang_vel",
            "air_time",
            "foot_clearance",
            "foot_swing_height",
            "foot_slip",
            "head_pose_tracking",
        ):
            term = getattr(self.rewards, name)
            setattr(
                self.rewards,
                name,
                RewardTermCfg(
                    func=recovery_mdp.recovery_walking_reward,
                    weight=term.weight,
                    params={"term_func": term.func, "term_params": term.params},
                ),
            )

    def play_mode(self):
        """Exercise the full pose distribution without changing difficulty during evaluation."""
        super().play_mode()
        self.recovery.initial_level = self.recovery.max_level
        self.recovery.curriculum_enabled = False
