# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional action sources for interactive playback of ordinary RL tasks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from isaaclab.utils import configclass

if TYPE_CHECKING:
    import torch

    from .teleop import TeleopSource


@configclass
class TeleopSourceCfg:
    """Construction of an input source, activated only by an interactive runner."""

    class_type: type[TeleopSource] | str = MISSING
    """Source constructor, called with ``(cfg, env, *, env_ids)``.

    Use a lazy ``"package.module:Class"`` reference to keep device dependencies
    out of task configuration and training imports.
    """


@configclass
class TeleopCfg:
    """Route an optional input source into an RL environment's existing actions.

    Merely declaring this configuration does not initialize a device or change
    training. Playback activates it through :class:`~isaaclab.envs.TeleopEnvWrapper`.
    Selected actions replace the baseline policy actions while the source is active;
    all other actions keep their policy values. No action terms are replaced or added.
    """

    source: TeleopSourceCfg = MISSING
    """Capture input source configured for the task's action conventions."""

    env_ids: list[int] = [0]
    """Environment rows controlled by the source, in source-output order."""

    action_terms: list[str] = []
    """Manager action terms to control, in source-output order. Empty selects all actions.

    Mutually exclusive with :attr:`action_indices`. Only manager-based tasks support
    named terms. The source must match each term's existing units, scaling, and frames.
    """

    action_indices: list[int] = []
    """Explicit flattened action columns, in source-output order. Empty selects all actions."""

    agent: str | None = None
    """Agent to control in a direct multi-agent task. Required for multi-agent tasks."""

    action_adapter: Callable[[torch.Tensor, Any, torch.Tensor], torch.Tensor] | None = None
    """Optional conversion called with ``(source_actions, env, env_ids)``.

    Return a tensor of shape ``(number of selected rows, number of selected columns)``.
    A vector is also accepted for a single selected environment. Conversion may read
    scene state (for example to transform frames), but must not step or reset the
    environment. Without a conversion, the source already uses task action conventions.
    """

    def validate_config(self) -> None:
        """Reject ambiguous ownership before starting a device."""
        if not self.env_ids or any(type(index) is not int or index < 0 for index in self.env_ids):
            raise ValueError("teleop.env_ids must contain non-negative integer environment indices.")
        if len(set(self.env_ids)) != len(self.env_ids):
            raise ValueError("teleop.env_ids must not contain duplicates.")
        if self.action_terms and self.action_indices:
            raise ValueError("Select teleop.action_terms or teleop.action_indices, not both.")
        if len(set(self.action_terms)) != len(self.action_terms):
            raise ValueError("teleop.action_terms must not contain duplicates.")
        if any(type(index) is not int or index < 0 for index in self.action_indices):
            raise ValueError("teleop.action_indices must contain non-negative integers.")
        if len(set(self.action_indices)) != len(self.action_indices):
            raise ValueError("teleop.action_indices must not contain duplicates.")
