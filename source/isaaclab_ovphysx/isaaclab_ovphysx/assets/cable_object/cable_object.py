# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import CableObjectCfg


class CableObject:
    """Unsupported OpenUSD PhysX cable object implementation."""

    __backend_name__: str = "ovphysx"
    """The name of the backend for the cable object."""

    def __init__(self, cfg: CableObjectCfg):
        """Raise because OpenUSD PhysX does not support cable objects.

        Args:
            cfg: A configuration instance.

        Raises:
            NotImplementedError: Always raised.
        """
        del cfg
        raise NotImplementedError("CableObject is not supported by the OpenUSD PhysX backend.")
