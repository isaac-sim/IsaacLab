# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interface for the process runtime a simulation config requires."""

from __future__ import annotations

import argparse


class SimulationLauncher:
    """Starts the process runtime a resolved simulation config requires, and stops it.

    A backend package subclasses this and names the subclass in its config's ``launcher_type``.
    :func:`~isaaclab.app.launch_simulation` constructs it, which starts the runtime, and calls
    :meth:`close` when the simulation ends.
    """

    device: str | None = None
    """Simulation device chosen by the runtime, or None to keep the config's device."""

    def __init__(self, launcher_args: argparse.Namespace | dict | None = None):
        """Start the runtime.

        Args:
            launcher_args: Parsed launcher arguments.
        """

    def close(self, exit_code: int = 0) -> None:
        """Stop the runtime.

        Args:
            exit_code: Exit status of the simulation, for runtimes that end the process.
        """
