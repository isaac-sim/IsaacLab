# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the SO101 keyboard letter-typing task."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import LetterTypingCommand


def typing_mistake(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate the episode when a wrong (or extra) key has been typed.

    Fires as soon as the typed buffer extends past the correct prefix (``typed_len > prefix_len``),
    i.e. the agent typed a key that does not continue the target word. Note this makes the backspace
    recovery path unreachable - a single mistyped key ends the episode.

    Args:
        env: The environment instance.
        command_name: Name of the :class:`LetterTypingCommand` term to read typing state from.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    return command.typed_len > command.prefix_len


def typing_complete(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate the episode once the whole target word has been typed correctly.

    Fires when the :class:`LetterTypingCommand` edit distance reaches zero (the typed buffer exactly
    matches the target), mirroring the :func:`~...mdp.rewards.typing_success` bonus. This is a genuine
    goal-reached terminal state, so it should be registered without ``time_out=True``.

    Args:
        env: The environment instance.
        command_name: Name of the :class:`LetterTypingCommand` term to read typing state from.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    return command.distance == 0
