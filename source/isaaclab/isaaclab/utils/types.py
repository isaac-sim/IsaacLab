# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for different data types."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class ArticulationActions:
    """Data container to store articulation's joints actions.

    This class is used to store the actions of the joints of an articulation.
    It is used to store the joint positions, velocities, efforts, and indices.

    If the actions are not provided, the values are set to None.
    """

    joint_positions: torch.Tensor | None = None
    """The joint positions of the articulation. Defaults to None."""

    joint_velocities: torch.Tensor | None = None
    """The joint velocities of the articulation. Defaults to None."""

    joint_efforts: torch.Tensor | None = None
    """The joint efforts of the articulation. Defaults to None."""

    joint_indices: torch.Tensor | Sequence[int] | slice | None = None
    """The joint indices of the articulation. Defaults to None.

    If the joint indices are a slice, this indicates that the indices are continuous and correspond
    to all the joints of the articulation. We use a slice to make the indexing more efficient.
    """
