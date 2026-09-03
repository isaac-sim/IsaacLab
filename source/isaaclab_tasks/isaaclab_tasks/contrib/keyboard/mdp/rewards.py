# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the SO101 keyboard letter-typing task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import LetterTypingCommand


def mechanical_power(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Compute total absolute mechanical power across the robot joints [W].

    Non-finite values that can occur briefly during a reset are replaced with zero.

    Args:
        env: The environment instance.
        robot_cfg: Scene entity for the robot articulation.

    Returns:
        Total absolute joint power [W], shape ``(num_envs,)``.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    power = torch.sum((robot.data.applied_torque.torch * robot.data.joint_vel.torch).abs(), dim=1)
    return torch.where(torch.isfinite(power), power, torch.zeros_like(power))


def letter_typing_progress(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    r"""High-water-mark typing-progress reward: a per-episode ratchet on the correct-prefix length.

    Fetches the :class:`LetterTypingCommand` term and returns a sparse per-step signal driven only by
    *records* in the length of the contiguous correct prefix (``prefix_len``):

    * ``+1`` on any step that sets a new episode maximum prefix length (genuine forward progress),
    * ``-1`` on any step that sets a new episode minimum prefix length (destroying correct progress),
    * ``0`` otherwise - staying put, or re-reaching a prefix length already seen this episode.

    Since one control step registers at most one keystroke, ``prefix_len`` moves by at most one, so each
    integer level is crossed cleanly. Rewarding only *new* maxima means the total positive reward per
    episode is bounded by ``target_len - prefix_0`` and the total penalty by ``prefix_0`` (with
    ``prefix_0`` the correct prefix of the reset buffer); consequently ``type-wrong -> backspace`` and
    ``backspace -> retype`` loops cannot farm reward - a re-reached level is neither a new max nor a new
    min. The marks are episode state seeded at reset, so they live on the command (see
    :meth:`~...typing_commands.LetterTypingCommand._update_metrics`); this term only reads the resulting
    per-step flags. With single-letter targets ``prefix_0`` is always ``0``, so the new-min penalty is
    inert and only the new-max bonus fires; it starts mattering for multi-letter targets.

    Args:
        env: The environment instance.
        command_name: Name of the :class:`LetterTypingCommand` term to read progress from.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    return command.new_high.float() - command.new_low.float()


def typing_success(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    r"""Sparse success bonus: ``1`` when the whole target word is typed correctly.

    Fires when the :class:`LetterTypingCommand` edit distance is zero, i.e. the typed buffer exactly
    matches the target with no missing and no extra/wrong keys. It is returned on every step the env
    stays in that completed state (until the command resamples a new word).

    Args:
        env: The environment instance.
        command_name: Name of the :class:`LetterTypingCommand` term to read completion from.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    return (command.distance == 0).float()


class reach_key(ManagerTermBase):
    r"""Reach reward toward the next target key using a tanh kernel on either jaw tip.

    Each gripper jaw has a tip at a fixed offset in its link frame. The reward uses the *closer* of
    the two tips to the next target key, so reaching with either jaw counts:

    .. math::

        r = 1 - \tanh\!\left(\frac{\min(d_\text{moving}, d_\text{fixed})}{\text{std}}\right)

    The value is ``1`` when a tip is on the (lowered) target and decays toward ``0`` with distance
    (``std`` [m] sets the falloff scale), so it stays in ``(0, 1]``. The target is placed
    ``press_depth`` [m] below the key (along world ``-Z``) so the optimum sits past the actuation point,
    encouraging the policy to push down rather than just hover on the cap. Body indices, tip offsets, and
    the press offset are resolved once at construction so each step only does the gather and kernel.

    Args (from ``cfg.params``):
        command_name: Name of the :class:`LetterTypingCommand` term providing the next target key.
        moving_jaw_body: Body name of the moving jaw link.
        moving_jaw_offset: Tip offset [m] in the moving jaw link frame.
        fixed_jaw_body: Body name of the fixed jaw link.
        fixed_jaw_offset: Tip offset [m] in the fixed jaw link frame.
        std: Distance [m] falloff scale of the tanh kernel. Defaults to ``0.2``.
        press_depth: Distance [m] to lower the reach target below the key along world ``-Z`` so reaching
            it requires pressing the key down. Defaults to ``0.0`` (reach the key itself).
        asset_cfg: Scene entity for the robot whose jaw links are read. Defaults to ``SceneEntityCfg("robot")``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.robot = env.scene[asset_cfg.name]
        self.std: float = cfg.params.get("std", 0.2)  # type: ignore
        # resolve body indices and broadcast tip offsets once (they never change at runtime).
        self._moving = self.robot.body_names.index(cfg.params["moving_jaw_body"])
        self._fixed = self.robot.body_names.index(cfg.params["fixed_jaw_body"])
        self._off_moving = torch.tensor(cfg.params["moving_jaw_offset"], device=env.device).expand(env.num_envs, 3)
        self._off_fixed = torch.tensor(cfg.params["fixed_jaw_offset"], device=env.device).expand(env.num_envs, 3)
        # world -Z offset that lowers the reach target below the key to encourage pressing.
        self._press_offset = torch.tensor((0.0, 0.0, cfg.params.get("press_depth", 0.0)), device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        moving_jaw_body: str,
        moving_jaw_offset: tuple[float, float, float],
        fixed_jaw_body: str,
        fixed_jaw_offset: tuple[float, float, float],
        std: float = 0.2,
        press_depth: float = 0.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
        # aim slightly below the key (world -Z) so the tip is rewarded for pressing, not just touching.
        target_pos_w = command.target_key_pos_w() - self._press_offset  # (N, 3)

        body_pos_w = self.robot.data.body_pos_w.torch
        body_quat_w = self.robot.data.body_quat_w.torch
        tip_moving = body_pos_w[:, self._moving] + quat_apply(body_quat_w[:, self._moving], self._off_moving)
        tip_fixed = body_pos_w[:, self._fixed] + quat_apply(body_quat_w[:, self._fixed], self._off_fixed)

        distance = torch.minimum(
            torch.linalg.norm(tip_moving - target_pos_w, dim=-1),
            torch.linalg.norm(tip_fixed - target_pos_w, dim=-1),
        )
        return 1.0 - torch.tanh(distance / self.std)
