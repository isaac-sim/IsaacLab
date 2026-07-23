# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native command term that does nothing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from isaaclab_experimental.managers import CommandTerm

if TYPE_CHECKING:
    from .commands_cfg import NullCommandCfg


class NullCommand(CommandTerm):
    """Command generator that does nothing.

    Warp-native twin of :class:`isaaclab.envs.mdp.commands.NullCommand`. It does not
    generate any commands and is used for environments that do not require one. All
    inherited bookkeeping runs as device-side kernels on empty metrics with an
    infinite resampling time, so the term is trivially graph-capturable.
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
    Implementation specific functions.
    """

    def _update_metrics(self):
        pass

    def _resample_command(self, env_mask: wp.array):
        pass

    def _update_command(self):
        pass
