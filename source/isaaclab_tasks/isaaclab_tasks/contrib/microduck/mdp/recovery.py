# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recovery deadlines, reset curriculum, and rewards for one walking/get-up policy."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from .rewards import track_angular_velocity, track_linear_velocity

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class MicroDuckRecoveryCfg:
    """Performance gates and recovery timing; level zero learns upright walking."""

    initial_level: int = 0
    max_level: int = 5
    curriculum_enabled: bool = True
    episodes_per_level: int = 4096
    walking_survival: float = 0.8
    walking_tracking: float = 0.65
    recovery_success: float = 0.6
    recovery_time_s: float = 6.0
    """Deadline for each recovery attempt [s]."""
    total_recovery_time_s: float = 8.0
    """Maximum accumulated recovery time in an episode [s]."""
    stable_time_s: float = 0.5
    """Continuous stable standing required to end a recovery [s]."""
    tracking_time_s: float = 2.0
    """Continuous commanded motion required after standing to count success [s]."""


class RecoveryState:
    """Per-episode state, updated exactly once by the termination manager."""

    def __init__(self, num_envs: int, device: str, dt: float, cfg: MicroDuckRecoveryCfg):
        self.cfg, self.dt, self.level = cfg, dt, cfg.initial_level
        if not 0 <= self.level <= cfg.max_level or cfg.max_level < 1:
            raise ValueError("Recovery level must lie between zero and max_level (at least one).")
        for name in ("elapsed", "total", "stable", "tracking_time", "tracking_sum", "steps", "budget"):
            setattr(self, name, torch.zeros(num_envs, device=device))
        for name in ("active", "selected", "had_fall", "fall", "success"):
            setattr(self, name, torch.zeros(num_envs, dtype=torch.bool, device=device))
        self.episode_level = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.seen = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.totals = torch.zeros(6, device=device)
        self.metrics = {"level": float(self.level), "survival": 0.0, "tracking": 0.0, "recovery_success": 0.0}

    def reset(self, env_ids: torch.Tensor, selected: torch.Tensor) -> None:
        """Start episodes, without penalizing deliberately randomized starts."""
        for name in ("elapsed", "total", "stable", "tracking_time", "tracking_sum", "steps"):
            getattr(self, name)[env_ids] = 0
        for name in ("had_fall", "fall", "success"):
            getattr(self, name)[env_ids] = False
        self.active[env_ids] = selected
        self.selected[env_ids] = selected
        self.episode_level[env_ids] = self.level
        # Freeze each episode's rule even if other environments advance the curriculum.
        self.budget[env_ids] = self.cfg.recovery_time_s if self.level else 0.0

    def update(self, height: torch.Tensor, alignment: torch.Tensor, speed: torch.Tensor, tracking: torch.Tensor):
        """Return failures using height [m], signed up alignment, and angular speed [rad/s]."""
        fallen = (height < 0.055) | (alignment < math.cos(math.radians(70.0)))
        standing = (height > 0.095) & (alignment > math.cos(math.radians(30.0))) & (speed < 2.0)
        self.fall.copy_(fallen & ~self.active)
        self.had_fall |= self.fall
        self.active |= fallen
        self.elapsed += self.active * self.dt
        self.total += self.active * self.dt
        self.stable.copy_(torch.where(self.active & standing, self.stable + self.dt, 0.0))
        # Expiry wins over a stand-up that finishes on the deadline.
        failed = (self.active & (self.elapsed >= self.budget)) | (self.total >= self.cfg.total_recovery_time_s)
        recovered = self.active & (self.stable >= self.cfg.stable_time_s) & ~failed
        self.active &= ~recovered
        self.elapsed[recovered] = 0.0
        following = standing & ~self.active & (tracking >= self.cfg.walking_tracking)
        self.tracking_time.copy_(torch.where(following, self.tracking_time + self.dt, 0.0))
        self.success |= self.selected & (self.tracking_time >= self.cfg.tracking_time_s)
        self.tracking_sum += tracking * standing
        self.steps += 1
        return failed

    def curriculum(self, env_ids: torch.Tensor, timed_out: torch.Tensor) -> dict[str, float]:
        """Advance only on completed episodes from the current level and both cohorts."""
        valid = (self.steps[env_ids] > 0) & (self.episode_level[env_ids] == self.level) & ~self.seen[env_ids]
        self.seen[env_ids[valid]] = True
        walking = valid & ~self.selected[env_ids]
        recovery = valid & self.selected[env_ids]
        self.totals += torch.stack(
            (
                valid.sum(),
                walking.sum(),
                (walking & ~self.had_fall[env_ids] & timed_out).sum(),
                (walking * self.tracking_sum[env_ids] / self.steps[env_ids].clamp_min(1)).sum(),
                recovery.sum(),
                (recovery & self.success[env_ids] & ~self.active[env_ids] & timed_out).sum(),
            )
        )
        # Every environment contributes once: early failures must not be forgotten before survivors finish.
        if not self.seen.all().item():
            return self.metrics.copy()
        self.seen.zero_()
        if self.totals[0].item() >= self.cfg.episodes_per_level:
            _, walks, survivors, tracking, recoveries, successes = self.totals.tolist()
            self.metrics.update(
                survival=survivors / max(walks, 1),
                tracking=tracking / max(walks, 1),
                recovery_success=successes / max(recoveries, 1),
            )
            ready = (
                walks > 0
                and self.metrics["survival"] >= self.cfg.walking_survival
                and self.metrics["tracking"] >= self.cfg.walking_tracking
                and (
                    self.level == 0
                    or (recoveries > 0 and self.metrics["recovery_success"] >= self.cfg.recovery_success)
                )
            )
            if ready and self.cfg.curriculum_enabled:
                self.level = min(self.level + 1, self.cfg.max_level)
            self.metrics["level"] = float(self.level)
            self.totals.zero_()
        return self.metrics.copy()


def recovery_state(env: ManagerBasedRLEnv) -> RecoveryState:
    """Get the state shared by this task's reset, termination, and curriculum terms."""
    if not hasattr(env, "_microduck_recovery"):
        env._microduck_recovery = RecoveryState(env.num_envs, env.device, env.step_dt, env.cfg.recovery)
    return env._microduck_recovery


def recovery_posture(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return root height over the flat floor [m] and signed vertical alignment."""
    data = env.scene["robot"].data
    return data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2], -data.projected_gravity_b.torch[:, 2]


def recovery_failed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Allow all falls the episode's recovery budget; terminate expired attempts."""
    height, alignment = recovery_posture(env)
    speed = env.scene["robot"].data.root_link_ang_vel_b.torch.norm(dim=-1)
    return recovery_state(env).update(height, alignment, speed, recovery_tracking(env))


def recovery_tracking(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Require both velocity scores and motion in each nontrivial commanded direction."""
    command = env.command_manager.get_command("base_velocity")
    data = env.scene["robot"].data
    linear = data.root_link_lin_vel_b.torch[:, :2]
    yaw = data.root_link_ang_vel_b.torch[:, 2]
    magnitude_squared = command[:, :2].square().sum(dim=-1)
    translating = (magnitude_squared <= 0.05**2) | ((linear * command[:, :2]).sum(-1) >= 0.5 * magnitude_squared)
    turning = (command[:, 2].abs() <= 0.2) | (yaw * command[:, 2] >= 0.5 * command[:, 2].square())
    score = torch.minimum(
        track_linear_velocity(env, math.sqrt(0.1), "base_velocity"),
        track_angular_velocity(env, math.sqrt(0.5), "base_velocity"),
    )
    return torch.nan_to_num(score * translating * turning, nan=0.0)


def recovery_curriculum(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice) -> dict[str, float]:
    """Measure fall-free walking separately from recovery followed by commanded motion."""
    state = recovery_state(env)
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    # RSL-RL randomizes initial episode clocks; omit these shortened episodes from the gate.
    env_ids = env_ids[state.steps[env_ids] == env.episode_length_buf[env_ids]]
    timed_out = env.episode_length_buf[env_ids] >= env.max_episode_length
    return state.curriculum(env_ids, timed_out)


def reset_recovery(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice, asset_cfg: SceneEntityCfg) -> None:
    """Mix upright starts with bounded joint offsets [rad] and progressively tilted drops."""
    state = recovery_state(env)
    robot = env.scene[asset_cfg.name]
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    count = len(env_ids)
    difficulty = state.level / state.cfg.max_level
    selected = torch.rand(count, device=env.device) < 0.5 * difficulty
    state.reset(env_ids, selected)
    pose = robot.data.default_root_pose.torch[env_ids].clone()
    pose[:, :3] += env.scene.env_origins[env_ids]
    angles = (2 * torch.rand(count, 3, device=env.device) - 1) * math.pi
    angles[:, :2] *= difficulty * selected[:, None]
    pose[:, 3:7] = quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2])
    # Start above the robot's reach: gravity produces a landing without teleporting colliders through the floor.
    pose[selected, 2] = env.scene.env_origins[env_ids[selected], 2] + 0.4
    joints = robot.data.default_joint_pos.torch[env_ids].clone()
    joint_ids = asset_cfg.joint_ids
    offsets = (2 * torch.rand(count, len(joint_ids), device=env.device) - 1) * difficulty
    limits = robot.data.soft_joint_pos_limits.torch[env_ids][:, joint_ids]
    joints[:, joint_ids] = (joints[:, joint_ids] + offsets * selected[:, None]).clamp(limits[..., 0], limits[..., 1])
    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(count, 6, device=env.device), env_ids=env_ids)
    robot.write_joint_state_to_sim_index(position=joints, velocity=torch.zeros_like(joints), env_ids=env_ids)


def recovery_readiness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Smoothly enable walking rewards near standing height and upright orientation."""
    height, alignment = recovery_posture(env)
    return ((height - 0.055) / 0.04).clamp(0, 1) * ((alignment - 0.35) / 0.5).clamp(0, 1)


class recovery_walking_reward(ManagerTermBase):
    """Keep the existing walking reward while allowing other movements on the ground."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.term = cfg.params["term_func"]
        if isinstance(self.term, type):
            self.term = self.term(RewardTermCfg(func=self.term, params=cfg.params["term_params"], weight=1.0), env)

    def reset(self, env_ids=None) -> None:
        if isinstance(self.term, ManagerTermBase):
            self.term.reset(env_ids)

    def __call__(self, env: ManagerBasedRLEnv, term_func: Callable, term_params: dict) -> torch.Tensor:
        return recovery_readiness(env) * self.term(env, **term_params)


def recovery_upright(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Give broad signed orientation and height shaping, including while lying down."""
    height, alignment = recovery_posture(env)
    return 0.5 * (alignment + 1).clamp(0, 2) * (height / 0.115).clamp(0, 1)


def recovery_fall_cost(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Charge once per spontaneous fall, independent of the control timestep [s]."""
    return recovery_state(env).fall.float() / env.step_dt


def recovery_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Expose episode timing and height [m] to the privileged value function."""
    state = recovery_state(env)
    height, _ = recovery_posture(env)
    return torch.stack(
        (
            state.active.float(),
            (state.budget - state.elapsed).clamp_min(0) / state.cfg.recovery_time_s,
            (state.cfg.total_recovery_time_s - state.total).clamp_min(0) / state.cfg.total_recovery_time_s,
            height,
        ),
        dim=-1,
    )
