# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for different data types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class MultiRotorActions:
    """Data container to store articulation's thruster actions.

    This class is used to store the actions of the thrusters of a multirotor.
    It is used to store the thrust values and indices.

    If the actions are not provided, the values are set to None.
    """

    thrusts: torch.Tensor | None = None
    """The thrusts of the multirotor. Defaults to None."""

    thruster_indices: torch.Tensor | Sequence[int] | slice | None = None
    """The thruster indices of the multirotor. Defaults to None.

    If the thruster indices are a slice, this indicates that the indices are continuous and correspond
    to all the thrusters of the multirotor. We use a slice to make the indexing more efficient.
    """
