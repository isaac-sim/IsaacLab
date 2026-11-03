# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity, pose, posture and phase commands for MicroDuck."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import math


class UniformPoseDeltaCommand(CommandTerm):
    """Generic N-dimensional uniform pose-delta command."""

    cfg: UniformPoseDeltaCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: UniformPoseDeltaCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term."""
        super().__init__(cfg, env)
        self.dim = len(cfg.ranges)
        self._command = torch.zeros(self.num_envs, self.dim, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Pose deltas [m or rad, depending on dimension], shape (num_envs, dim)."""
        return self._command

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass

    def _resample_command(self, env_ids: Sequence[int] | slice):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        num_envs = len(env_ids)
        if num_envs == 0:
            return
        # Read the bounds on every resample: the curriculum widens the head-pose ranges during training.
        low, high = torch.tensor(self.cfg.ranges, device=self.device).unbind(-1)
        self._command[env_ids] = low + (high - low) * torch.rand(num_envs, self.dim, device=self.device)


@configclass
class UniformPoseDeltaCommandCfg(CommandTermCfg):
    """Configuration for the N-dimensional uniform pose-delta command term."""

    class_type: type[UniformPoseDeltaCommand] = UniformPoseDeltaCommand
    ranges: tuple[tuple[float, float], ...] = ()
    """Per-dimension sampling bounds [m or rad, depending on dimension]. Length sets the command width."""


class MicroDuckVelocityCommand(UniformVelocityCommand):
    """Velocity command with MicroDuck's forward-only and turn-in-place buckets."""

    cfg: MicroDuckVelocityCommandCfg
    """Configuration for the command term."""

    def _resample_command(self, env_ids: Sequence[int] | slice):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)
        forward_ids = env_ids[r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs]
        if len(forward_ids) > 0:
            self.vel_command_b[forward_ids, 0] = (
                self.vel_command_b[forward_ids, 0].abs().clamp(min=self.cfg.forward_min_speed)
            )
            self.vel_command_b[forward_ids, 1] = 0.0
            self.vel_command_b[forward_ids, 2] = 0.0
        if self.cfg.rel_turn_in_place_envs <= 0.0:
            return
        turn_ids = env_ids[r.uniform_(0.0, 1.0) < self.cfg.rel_turn_in_place_envs]
        if len(turn_ids) == 0:
            return
        self.vel_command_b[turn_ids, :2] = 0.0
        low, high = self.cfg.ranges.ang_vel_z
        max_rate = max(abs(low), abs(high))
        turn_r = torch.empty(len(turn_ids), device=self.device)
        sign = torch.where(turn_r.uniform_(0.0, 1.0) < 0.5, -1.0, 1.0)
        magnitude = turn_r.uniform_(self.cfg.turn_in_place_min_fraction * max_rate, max_rate)
        self.vel_command_b[turn_ids, 2] = sign * magnitude
        self.is_standing_env[turn_ids] = False


@configclass
class MicroDuckVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the MicroDuck velocity command term."""

    class_type: type[MicroDuckVelocityCommand] = MicroDuckVelocityCommand
    rel_forward_envs: float = 0.0
    """Probability that an environment is drawn into the forward-only bucket. Defaults to 0.0.

    The turn-in-place bucket is drawn independently afterwards and takes precedence, so the effective forward-only
    share is ``rel_forward_envs * (1 - rel_turn_in_place_envs)``.
    """
    rel_turn_in_place_envs: float = 0.0
    """Probability that an environment is commanded to turn on the spot. Defaults to 0.0."""
    forward_min_speed: float = 0.3
    """Minimum forward speed of forward-only commands [m/s]."""
    turn_in_place_min_fraction: float = 0.4
    """Minimum turn-in-place yaw rate, as a fraction of the largest ``ang_vel_z`` bound [-]."""


class MicroDuckRoughVelocityCommand(MicroDuckVelocityCommand):
    """Accumulate walking and tracking before command resampling for terrain progression."""

    def __init__(self, cfg: MicroDuckVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.terrain_progress = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in ("commanded_distance", "walked_distance", "error_distance", "walking_time", "yaw_error", "time")
        }
        self._last_progress_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def record_terrain_progress(self, env_ids: Sequence[int] | slice = slice(None)) -> None:
        """Record each control sample once, including the terminal sample before reset.

        Distances and integrated XY error are in [m], times in [s], and integrated yaw-rate error in [rad].
        Translation commands of at least 0.05 [m/s] contribute to walking statistics.
        """
        steps = self._env.episode_length_buf[env_ids]
        dt = (steps > self._last_progress_step[env_ids]) * self._env.step_dt
        command = self.command[env_ids]
        velocity = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.robot.data.root_link_quat_w.torch[env_ids]),
            self.robot.data.root_link_lin_vel_w.torch[env_ids],
        )[:, :2]
        speed = command[:, :2].norm(dim=-1)
        walking_dt = dt * (speed >= 0.05)
        values = {
            "commanded_distance": speed * walking_dt,
            "walked_distance": velocity.norm(dim=-1) * walking_dt,
            "error_distance": (command[:, :2] - velocity).norm(dim=-1) * walking_dt,
            "walking_time": walking_dt,
            "yaw_error": (command[:, 2] - self.robot.data.root_link_ang_vel_w.torch[env_ids, 2]).abs() * dt,
            "time": dt,
        }
        for name, value in values.items():
            self.terrain_progress[name][env_ids] += value
        self._last_progress_step[env_ids] = steps

    def _update_metrics(self):
        super()._update_metrics()
        self.record_terrain_progress()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Clear progress after the curriculum has consumed the completed episodes."""
        extras = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        for value in self.terrain_progress.values():
            value[ids] = 0.0
        self._last_progress_step[ids] = 0
        return extras


class ZeroablePoseDeltaCommand(CommandTerm):
    """Generic N-dimensional uniform pose-delta command.

    Each dimension is drawn independently and uniformly from its own range and then held until the
    next resample. The width of the command is the length of :attr:`ZeroablePoseDeltaCommandCfg.ranges`,
    so one term serves both MicroDuck pose commands:

    The ranges are read on every resample rather than cached, so a curriculum can widen them by
    reassigning :attr:`ZeroablePoseDeltaCommandCfg.ranges` on the live term configuration. Widening
    only: the command width is fixed at construction, and a resample asserts that the replacement
    tuple still has one range per dimension.
    """

    cfg: ZeroablePoseDeltaCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: ZeroablePoseDeltaCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        super().__init__(cfg, env)

        self.dim = len(cfg.ranges)
        """Number of commanded dimensions."""

        self._command = torch.zeros(self.num_envs, self.dim, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPoseDeltaCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The commanded pose delta. Shape is (num_envs, dim)."""
        return self._command

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass

    def _resample_command(self, env_ids: Sequence[int] | slice):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        num_envs = len(env_ids)
        if num_envs == 0:
            return
        # a curriculum that shortened the tuple would silently leave the dropped dimensions holding
        # their last sampled value rather than fail
        assert len(self.cfg.ranges) == self.dim, (
            f"Command width is fixed at {self.dim} dimensions but the configuration now lists"
            f" {len(self.cfg.ranges)} ranges."
        )
        r = torch.empty(num_envs, device=self.device)
        for dim, (low, high) in enumerate(self.cfg.ranges):
            self._command[env_ids, dim] = r.uniform_(low, high)
        if self.cfg.zero_command_prob > 0.0:
            zeroed = torch.as_tensor(env_ids, device=self.device)[
                torch.rand(num_envs, device=self.device) < self.cfg.zero_command_prob
            ]
            self._command[zeroed] = 0.0


@configclass
class ZeroablePoseDeltaCommandCfg(CommandTermCfg):
    """Configuration for the N-dimensional uniform pose-delta command term.

    Please refer to the :class:`ZeroablePoseDeltaCommand` class for more details.
    """

    class_type: type[ZeroablePoseDeltaCommand] = ZeroablePoseDeltaCommand

    ranges: tuple[tuple[float, float], ...] = ()
    """Per-dimension ``(low, high)`` sampling range. Its length sets the command width.

    Left as a plain tuple rather than a structured range class because a curriculum reassigns it
    wholesale as the training progresses.
    """

    zero_command_prob: float = 0.0
    """Probability that a resample yields the exact all-zero command. Defaults to 0.0.

    Independent uniform sampling of several dimensions essentially never produces the all-zero
    command, so without this bucket the deployment idle case -- "hold the nominal pose, no request"
    -- is absent from training and the policy only holds still when it is asked to.
    """


class ForwardMaskVelocityCommand(UniformVelocityCommand):
    """Velocity command with MicroDuck's forward-only and turn-in-place buckets.

    On top of the stock uniform sampling and the standing bucket, a resample assigns two further
    buckets:

    Both fractions and :attr:`~UniformVelocityCommandCfg.rel_standing_envs` are read on every
    resample, so a curriculum can ramp them on the live term configuration.
    """

    cfg: ForwardMaskVelocityCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: ForwardMaskVelocityCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        super().__init__(cfg, env)

        self.is_forward_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """Whether each environment is currently in the forward-only bucket."""

    def __str__(self) -> str:
        msg = super().__str__().replace("UniformVelocityCommand:", "MicroDuckVelocityCommand:", 1)
        msg += f"\n\tForward-only probability: {self.cfg.rel_forward_envs}"
        msg += f"\n\tTurn-in-place probability: {self.cfg.rel_turn_in_place_envs}"
        return msg

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int] | slice):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        # the bucket masks index into ``env_ids``, which the command manager may hand over as a
        # plain sequence
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._resample_command(env_ids)

        r = torch.empty(len(env_ids), device=self.device)
        self.is_forward_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs
        forward_ids = env_ids[self.is_forward_env[env_ids]]
        if len(forward_ids) > 0:
            self.vel_command_b[forward_ids, 0] = self.vel_command_b[forward_ids, 0].abs().clamp(min=0.3)
            self.vel_command_b[forward_ids, 1] = 0.0
            self.vel_command_b[forward_ids, 2] = 0.0

        if self.cfg.rel_turn_in_place_envs <= 0.0:
            return
        turn_ids = env_ids[r.uniform_(0.0, 1.0) < self.cfg.rel_turn_in_place_envs]
        if len(turn_ids) == 0:
            return
        self.vel_command_b[turn_ids, :2] = 0.0
        low, high = self.cfg.ranges.ang_vel_z
        max_rate = max(abs(low), abs(high))
        turn_r = torch.empty(len(turn_ids), device=self.device)
        sign = torch.where(turn_r.uniform_(0.0, 1.0) < 0.5, -1.0, 1.0)
        magnitude = turn_r.uniform_(0.4 * max_rate, max_rate)
        self.vel_command_b[turn_ids, 2] = sign * magnitude
        # these environments must actually turn, so un-mark them as standing -- otherwise
        # ``_update_command`` would zero the command they were just given
        self.is_standing_env[turn_ids] = False


@configclass
class ForwardMaskVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the MicroDuck velocity command term.

    Please refer to the :class:`ForwardMaskVelocityCommand` class for more details.
    """

    class_type: type[ForwardMaskVelocityCommand] = ForwardMaskVelocityCommand

    rel_forward_envs: float = 0.0
    """Probability that an environment is commanded to walk straight forward. Defaults to 0.0.

    Such an environment gets ``vx = |vx|.clamp(min=0.3)`` [m/s] and zero sway and yaw rate.
    """

    rel_turn_in_place_envs: float = 0.0
    """Probability that an environment is commanded to turn on the spot. Defaults to 0.0.

    Such an environment gets a zero linear command and a yaw rate of ``±U(0.4 * max, max)``
    [rad/s], where ``max`` is the larger magnitude of :attr:`Ranges.ang_vel_z`.
    """


class RelativeHeadingVelocityCommand(ForwardMaskVelocityCommand):
    """Velocity command whose yaw slot carries a *heading error* rather than a yaw rate.

    * a **target heading** in the world frame is drawn uniformly on every resample, and the yaw slot
      of the command is filled every step with the wrapped error between it and the robot's current
      heading, clamped to the configured yaw range. The stock heading controller cannot express this:
      it multiplies the error by a stiffness to produce a yaw *rate*, and it only drives the fraction
      of environments in its heading bucket;
    * the surge slot is a **throttle**, not a velocity target -- ``0`` coasts, positive pushes,
      negative brakes -- which is what the roller reward set reads it as.
    """

    cfg: RelativeHeadingVelocityCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: RelativeHeadingVelocityCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        super().__init__(cfg, env)

        self.target_heading_w = torch.zeros(self.num_envs, device=self.device)
        """Heading [rad] each environment is steered toward, in the world frame."""

        self._heading_max = float(cfg.ranges.ang_vel_z[1]) if cfg.ranges.ang_vel_z else 1.0
        # Clearing them drops the metrics rather than logging a perfect score for a comparison that was never made.
        self.metrics.clear()

    def __str__(self) -> str:
        msg = super().__str__().replace("MicroDuckVelocityCommand:", "RelativeHeadingVelocityCommand:", 1)
        msg += f"\n\tHeading error clamp: {self._heading_max}"
        return msg

    """
    Implementation specific functions.
    """

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> dict[str, float]:
        # skips ``UniformVelocityCommand.reset``, whose only extra work is finalizing the tracking
        # metrics cleared in ``__init__``
        return CommandTerm.reset(self, env_ids)

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int] | slice):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        super()._resample_command(env_ids)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        heading = torch.empty(len(env_ids), device=self.device)
        self.target_heading_w[env_ids] = heading.uniform_(-math.pi, math.pi)
        # the sampled yaw *rate* is meaningless here; ``_update_command`` overwrites the slot anyway
        self.vel_command_b[env_ids, 2] = 0.0

    def _update_command(self):
        super()._update_command()
        error = math_utils.wrap_to_pi(self.target_heading_w - self.robot.data.heading_w.torch)
        self.vel_command_b[:, 2] = error.clamp(-self._heading_max, self._heading_max)


@configclass
class RelativeHeadingVelocityCommandCfg(ForwardMaskVelocityCommandCfg):
    """Configuration for the relative-heading velocity command term.

    Please refer to the :class:`RelativeHeadingVelocityCommand` class for more details.
    """

    class_type: type[RelativeHeadingVelocityCommand] = RelativeHeadingVelocityCommand


class SitStandCommand(UniformVelocityCommand):
    """Posture flag riding in the twist slot: ``[sit_flag, 0, 0]``, ``sit_flag`` in ``{0, 1}``.

    Two quantities and the distinction between them is the whole design:

    The resample draws a fresh flag with probability :attr:`SitStandCommandCfg.sit_prob`, on the
    configured dwell time. Combined with a reset that spawns seated or standing with equal
    probability, the four (start state x request) combinations get equal coverage, which is what
    trains "hold what you are already doing" alongside the two transitions.
    """

    cfg: SitStandCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: SitStandCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        super().__init__(cfg, env)

        self._alpha = torch.zeros(self.num_envs, device=self.device)
        # Clearing them drops the metrics rather than logging a perfect score, as
        # :class:`RelativeHeadingVelocityCommand` does for the same reason.
        self.metrics.clear()

    def __str__(self) -> str:
        msg = super().__str__().replace("UniformVelocityCommand:", "SitStandCommand:", 1)
        msg += f"\n\tSit probability: {self.cfg.sit_prob}"
        msg += f"\n\tTarget ramp: {self.cfg.ramp_s} s"
        return msg

    """
    Properties
    """

    @property
    def alpha(self) -> torch.Tensor:
        """Slewed posture blend, 0 at the stand target and 1 at the sit target. Shape is (num_envs,).

        The posture rewards read this rather than :attr:`command`; see the class documentation for
        why the two differ.
        """
        return self._alpha

    """
    Implementation specific functions.
    """

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> dict[str, float]:
        # Skips ``UniformVelocityCommand.reset``, whose only extra work is finalizing the tracking
        # metrics cleared in ``__init__`` -- and which would raise on their absence -- exactly as
        # :class:`RelativeHeadingVelocityCommand` does for the same reason.
        #
        # The cost of the skip is that any *future* behaviour added to that parent's reset is
        # forfeited here silently, so a change to it has to be mirrored into these two subclasses by
        # hand. That is the price of not registering metrics for a comparison this term never makes.
        extras = CommandTerm.reset(self, env_ids)
        # Seed the blend from the height the robot actually spawned at, not from the flag it was
        # just handed: a seated spawn under a stand request must start its ramp at the sit end.
        if env_ids is None:
            env_ids = slice(None)
        self._alpha[env_ids] = self._alpha_from_height()[env_ids]
        return extras

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass  # no heading controller and no standing-environment machinery on a posture flag

    def _resample_command(self, env_ids: Sequence[int] | slice):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        num_envs = len(env_ids)
        if num_envs == 0:
            return
        sit = (torch.rand(num_envs, device=self.device) < self.cfg.sit_prob).float()
        # the whole slot is rewritten, so the sway and yaw columns stay identically zero
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_b[env_ids, 0] = sit

    def compute(self, dt: float):
        """Resample the flag on the dwell timer, then slew the blend toward it.

        Args:
            dt: Time [s] since the last call.
        """
        super().compute(dt)
        # Constant-rate slew: a full stand-to-sit traverse takes exactly ``ramp_s`` seconds whatever
        # the control rate, and the clamp is what keeps the blend from jumping when the flag flips.
        step = dt / max(self.cfg.ramp_s, 1e-6)
        delta = self.vel_command_b[:, 0] - self._alpha
        self._alpha += torch.clamp(delta, -step, step)

    """
    Helper functions.
    """

    def _alpha_from_height(self) -> torch.Tensor:
        """The blend the current trunk height corresponds to, clamped to ``[0, 1]``."""
        height = torch.nan_to_num(
            self.robot.data.root_link_pos_w.torch[:, 2] - self._env.scene.env_origins[:, 2],
            nan=self.cfg.stand_height,
        )
        span = max(self.cfg.stand_height - self.cfg.sit_height, 1e-6)
        return torch.clamp((self.cfg.stand_height - height) / span, 0.0, 1.0)


@configclass
class SitStandCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the sit-stand posture command term.

    Please refer to the :class:`SitStandCommand` class for more details.

    The inherited velocity ranges are never sampled -- :meth:`SitStandCommand._resample_command`
    writes the flag directly -- and neither are the heading and standing-environment fractions, which
    the overridden ``_update_command`` ignores.
    """

    class_type: type[SitStandCommand] = SitStandCommand

    sit_prob: float = 0.5
    """Probability that a resample requests the sitting posture. Defaults to 0.5."""

    ramp_s: float = 2.0
    """Time [s] for the slewed blend to traverse the full stand-to-sit range. Defaults to 2.0."""

    sit_height: float = 0.060
    """Trunk height [m] of the seated rest, used to seed the blend from a spawn pose."""

    stand_height: float = 0.115
    """Trunk height [m] of the standing rest, used to seed the blend from a spawn pose."""


class GroundPickPhaseCommand(UniformVelocityCommand):
    """Open-loop cycle clock riding in the twist slot: ``[cos(2*pi*phi), sin(2*pi*phi), 0]``.

    The clock is **open loop**. It advances by ``dt / period`` every control step and neither the
    robot's state nor the resampling timer can move it, so a policy trained here follows a schedule
    rather than a goal; on the robot the runtime plays the same clock from a button press.

    The starting phase is drawn uniformly at every reset by default
    (:attr:`GroundPickPhaseCommandCfg.randomize_phase`), which decorrelates the environments. With it
    off, every episode starts standing at phase 0, which is what a task whose deployed cycle begins
    on a button press wants instead.
    """

    cfg: GroundPickPhaseCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: GroundPickPhaseCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        super().__init__(cfg, env)

        self._phase = torch.zeros(self.num_envs, device=self.device)

        self.metrics.clear()
        self._write_command()

    def __str__(self) -> str:
        msg = super().__str__().replace("UniformVelocityCommand:", "GroundPickPhaseCommand:", 1)
        msg += f"\n\tCycle period: {self.cfg.period} s"
        msg += f"\n\tRandomized start phase: {self.cfg.randomize_phase}"
        return msg

    """
    Properties
    """

    @property
    def phase(self) -> torch.Tensor:
        """Position in the cycle, in ``[0, 1)``. Shape is (num_envs,)."""
        return self._phase

    """
    Implementation specific functions.
    """

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> dict[str, float]:
        # Skips ``UniformVelocityCommand.reset``, which would finalize the tracking metrics cleared
        # in ``__init__`` and raise on their absence; see :meth:`SitStandCommand.reset` for the cost
        # of that skip.
        extras = CommandTerm.reset(self, env_ids)
        if env_ids is None:
            env_ids = slice(None)
        if self.cfg.randomize_phase:
            self._phase[env_ids] = torch.rand_like(self._phase[env_ids])
        else:
            self._phase[env_ids] = 0.0
        self._write_command()
        return extras

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass  # no heading controller and no standing-environment machinery on a phase clock

    def _resample_command(self, env_ids: Sequence[int] | slice):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        pass  # the phase is continuous, so there is nothing to resample

    def compute(self, dt: float):
        """Advance the clock by one control step and re-encode it into the command slot.

        Args:
            dt: Time [s] since the last call.
        """
        self._phase = (self._phase + dt / max(self.cfg.period, 1e-6)) % 1.0
        self._write_command()

    """
    Helper functions.
    """

    def _write_command(self) -> None:
        """Encode the current phase into the three-wide twist slot."""
        angle = 2.0 * math.pi * self._phase
        self.vel_command_b[:, 0] = torch.cos(angle)
        self.vel_command_b[:, 1] = torch.sin(angle)
        self.vel_command_b[:, 2] = 0.0


@configclass
class GroundPickPhaseCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the ground-pick phase command term.

    Please refer to the :class:`GroundPickPhaseCommand` class for more details.

    The inherited velocity ranges, the resampling time and the heading and standing-environment
    fractions are all inert: the clock writes the slot directly and the overridden ``compute`` never
    reaches the machinery that would read them.
    """

    class_type: type[GroundPickPhaseCommand] = GroundPickPhaseCommand

    period: float = 4.0
    """Length [s] of one bend-and-return cycle. Defaults to 4.0, ``GP_PERIOD``."""

    randomize_phase: bool = True
    """Whether each episode starts at a uniformly drawn phase. Defaults to True."""
