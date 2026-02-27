# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for multirotor-specific data types.

This module defines data container classes used for passing multirotor-specific
information between components (e.g., between action terms and actuator models).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class MultiRotorActions:
    """Data container to store multirotor thruster actions.

    This dataclass is used to pass thrust commands and thruster indices between
    components in the multirotor control pipeline. It is primarily used internally
    by the :class:`~isaaclab_contrib.assets.Multirotor` class to communicate with
    :class:`~isaaclab_contrib.actuators.Thruster` actuator models.

    The container supports partial actions by allowing specification of which
    thrusters the actions apply to through the :attr:`thruster_indices` field.

    Attributes:
        thrusts: Thrust values for the specified thrusters. Shape is typically
            ``(num_envs, num_selected_thrusters)``.
        thruster_indices: Indices of thrusters that the thrust values apply to.
            Can be a tensor of indices, a sequence, a slice, or None for all thrusters.

    Example:
        .. code-block:: python

            # Create actions for all thrusters
            actions = MultiRotorActions(
                thrusts=torch.ones(num_envs, 4) * 5.0,
                thruster_indices=slice(None),  # All thrusters
            )

            # Create actions for specific thrusters
            actions = MultiRotorActions(
                thrusts=torch.tensor([[6.0, 7.0]]),
                thruster_indices=[0, 2],  # Only thrusters 0 and 2
            )

    Note:
        If both fields are ``None``, no action is taken. This is useful for
        conditional action application.

    .. seealso::
        - :class:`~isaaclab.utils.types.ArticulationActions`: Similar container for joint actions
        - :class:`~isaaclab_contrib.actuators.Thruster`: Thruster actuator that consumes these actions
    """

    thrusts: torch.Tensor | None = None
    """Thrust values for the multirotor thrusters.

    Shape: ``(num_envs, num_thrusters)`` or ``(num_envs, num_selected_thrusters)``

    The units depend on the actuator model configuration:
        - For force-based control: Newtons (N)
        - For RPS-based control: Revolutions per second (1/s)

    If ``None``, no thrust commands are specified.
    """

    thruster_indices: torch.Tensor | Sequence[int] | slice | None = None
    """Indices of thrusters that the thrust values apply to.

    This field specifies which thrusters the :attr:`thrusts` values correspond to.
    It can be:
        - A torch.Tensor of integer indices: ``torch.tensor([0, 2, 3])``
        - A sequence of integers: ``[0, 2, 3]``
        - A slice: ``slice(None)`` for all thrusters, ``slice(0, 2)`` for first two
        - ``None``: Defaults to all thrusters

    Using a slice is more efficient for contiguous thruster ranges as it avoids
    creating intermediate index tensors.

    Example:
        .. code-block:: python

            # All thrusters (most efficient)
            thruster_indices = slice(None)

            # First two thrusters
            thruster_indices = slice(0, 2)

            # Specific thrusters
            thruster_indices = [0, 2, 3]
    """
