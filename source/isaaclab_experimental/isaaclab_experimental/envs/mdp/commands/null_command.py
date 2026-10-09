# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first command generator that does nothing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab_experimental.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs.mdp.commands.commands_cfg import NullCommandCfg


class NullCommand(CommandTerm):
    """Command generator that does nothing.

    Warp-first twin of :class:`isaaclab.envs.mdp.commands.NullCommand`. It does not generate any commands and
    is used for environments that do not require any commands. Like the stable term, it never resamples, so
    its infinite resampling time is never drawn.
    """

    cfg: NullCommandCfg
    """Configuration for the command generator."""

    def __str__(self) -> str:
        msg = "NullCommand:\n"
        msg += "\tCommand dimension: N/A\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self):
        """Null command.

        Raises:
            RuntimeError: No command is generated. Always raises this error.
        """
        raise RuntimeError("NullCommandTerm does not generate any commands.")

    """
    Operations.
    """

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        env_mask: wp.array | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.reset_extras

    def compute(self, dt: float):
        pass

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        pass

    def _resample_command(self, env_mask: wp.array):
        pass

    def _update_command(self):
        pass
