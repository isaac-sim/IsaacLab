# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock implementation of PhysX shared metatype."""

from __future__ import annotations


class MockSharedMetatype:
    """Mock implementation of the shared metatype for articulation views.

    The shared metatype contains metadata about the articulation structure that is
    shared across all instances, such as DOF count, link count, and names.
    """

    def __init__(
        self,
        dof_count: int = 1,
        link_count: int = 2,
        dof_names: list[str] | None = None,
        link_names: list[str] | None = None,
        fixed_base: bool = False,
    ):
        """Initialize the mock shared metatype.

        Args:
            dof_count: Number of degrees of freedom (joints).
            link_count: Number of links (bodies).
            dof_names: Names of the DOFs. Defaults to ["dof_0", "dof_1", ...].
            link_names: Names of the links. Defaults to ["link_0", "link_1", ...].
            fixed_base: Whether the articulation has a fixed base.
        """
        self._dof_count = dof_count
        self._link_count = link_count
        self._dof_names = dof_names or [f"dof_{i}" for i in range(dof_count)]
        self._link_names = link_names or [f"link_{i}" for i in range(link_count)]
        self._fixed_base = fixed_base

    @property
    def dof_count(self) -> int:
        """Number of degrees of freedom."""
        return self._dof_count

    @property
    def link_count(self) -> int:
        """Number of links."""
        return self._link_count

    @property
    def dof_names(self) -> list[str]:
        """Names of the degrees of freedom."""
        return self._dof_names

    @property
    def link_names(self) -> list[str]:
        """Names of the links."""
        return self._link_names

    @property
    def fixed_base(self) -> bool:
        """Whether the articulation has a fixed base."""
        return self._fixed_base
