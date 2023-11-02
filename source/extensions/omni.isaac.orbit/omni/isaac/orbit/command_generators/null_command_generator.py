# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generator that does nothing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from omni.isaac.orbit.command_generators.command_generator_base import CommandGeneratorBase

if TYPE_CHECKING:
    from .command_generator_cfg import NullCommandGeneratorCfg


class NullCommandGenerator(CommandGeneratorBase):
    """Command generator that does nothing.

    This command generator does not generate any commands. It is used for environments that do not
    require any commands.
    """

    cfg: NullCommandGeneratorCfg
    """Configuration for the command generator."""

    def __str__(self) -> str:
        msg = "NullCommandGenerator:\n"
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
        raise RuntimeError("NullCommandGenerator does not generate any commands.")

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        return {}

    def compute(self, dt: float):
        pass

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass
