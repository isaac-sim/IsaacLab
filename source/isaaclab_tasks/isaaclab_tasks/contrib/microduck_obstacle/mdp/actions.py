# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen MicroDuck walking actor driven by bounded velocity commands."""

import torch

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.microduck.mdp.observations import joint_pos_rel_biased
from isaaclab_tasks.contrib.navigation.mdp.pre_trained_policy_action import (
    PreTrainedPolicyAction,
    PreTrainedPolicyActionCfg,
)


class MicroDuckWalkingAction(PreTrainedPolicyAction):
    """Preserve encoder bias and low-level history while PPO learns three commands."""

    def __init__(self, cfg, env):
        # The nested observation manager is built before env.action_manager exists.
        # Read bias directly from its owning low-level term during that construction.
        def encoder_positions(env, asset_cfg, biased=True, backlash_cfg=None):
            positions = joint_pos_rel_biased(env, asset_cfg=asset_cfg, biased=False, backlash_cfg=backlash_cfg)
            return positions + self.encoder_bias if biased else positions

        cfg.low_level_observations.joint_pos.func = encoder_positions
        # Newton may execute all physics substeps inside one action application.
        steps_per_apply = env.cfg.decimation if env.sim.backend_manager.handles_decimation() else 1
        if cfg.low_level_decimation % steps_per_apply:
            raise ValueError("Walking policy period must be divisible by the backend's batched physics steps.")
        cfg.low_level_decimation //= steps_per_apply
        super().__init__(cfg, env)
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        self._velocity_limits = torch.tensor(cfg.velocity_limits, device=self.device)
        self._command_step = torch.tensor(cfg.acceleration_limits, device=self.device) * env.step_dt
        self.previous_velocity_commands = torch.zeros_like(self._raw_actions)

    @property
    def encoder_bias(self):
        """Encoder offsets owned by the native biased joint action."""
        return self._low_level_action_term.encoder_bias

    @property
    def velocity_commands(self):
        """Velocity commands actually delivered to the frozen walking actor [m/s, m/s, rad/s]."""
        return self._raw_actions

    def process_actions(self, actions):
        requested = self._env.command_manager.get_command("base_velocity")
        self.previous_velocity_commands.copy_(self._raw_actions)
        commands = requested + actions.clamp(-1.0, 1.0) * self._velocity_limits
        # Front-facing sensors support braking and steering without blind reverse motion.
        commands[:, 0].clamp_(0.0, self.cfg.velocity_limits[0])
        commands[:, 1].clamp_(-self.cfg.velocity_limits[1], self.cfg.velocity_limits[1])
        commands[:, 2].clamp_(-self.cfg.velocity_limits[2], self.cfg.velocity_limits[2])
        # A residual may redirect the request, but cannot increase its translation speed.
        requested_speed = requested[:, :2].norm(dim=-1, keepdim=True)
        adjusted_speed = commands[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        commands[:, :2] *= (requested_speed / adjusted_speed).clamp(max=1.0)

        # Rate-limit the command, then constrain lateral motion to the forward cone.
        command_change = (commands - self._raw_actions).clamp(-self._command_step, self._command_step)
        commands = self._raw_actions + command_change
        lateral_limit = commands[:, 0] * self.cfg.max_lateral_ratio
        commands[:, 1] = commands[:, 1].clamp(-lateral_limit, lateral_limit)

        # An explicit stop or missing/stale frame bypasses acceleration limits.
        sensor = self._env.scene["head_tof"]
        stop_requested = requested.abs().sum(dim=-1) == 0
        sensor_unavailable = (sensor.delivered_sample_time < 0.0) | (sensor.sample_age > self.cfg.maximum_sensor_age)
        commands = torch.where((stop_requested | sensor_unavailable).unsqueeze(-1), 0.0, commands)
        super().process_actions(commands)

    @torch.no_grad()
    def apply_actions(self):
        super().apply_actions()

    def reset(self, env_ids=None):
        rows = slice(None) if env_ids is None else env_ids
        self._raw_actions[rows] = 0.0
        self.previous_velocity_commands[rows] = 0.0
        self.low_level_actions[rows] = 0.0
        self._low_level_action_term.reset(env_ids)
        self._low_level_obs_manager.reset(env_ids)


@configclass
class MicroDuckWalkingActionCfg(PreTrainedPolicyActionCfg):
    """Requires an explicitly selected exported official actor, including normalization."""

    class_type: type = MicroDuckWalkingAction
    velocity_limits: tuple[float, float, float] = (0.35, 0.12, 0.75)
    maximum_sensor_age: float = 0.25
    """Stop until a ToF frame arrives, or when its age exceeds this bound [s]."""
    max_lateral_ratio: float = 0.35
    """Keep translation within approximately +/-19 degrees of the head's forward direction."""
    acceleration_limits: tuple[float, float, float] = (1.0, 1.0, 3.0)
    """Maximum command rate of change [m/s², m/s², rad/s²], except an explicit stop."""
    debug_vis: bool = False
