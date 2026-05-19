# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from .cloner_strategies import random


@configclass
class CloneCfg:
    """Configuration for environment replication.

    The scene builds a :class:`~isaaclab.cloner.ClonePlan` directly from asset
    configuration, spawns the representative source prims, and then uses this
    configuration to dispatch USD and physics replication for that plan.
    """

    clone_regex: str = "/World/envs/env_.*"
    """Destination template for per-environment paths.

    The substring ``".*"`` is replaced with ``"{}"`` internally and formatted with the
    environment index (e.g., ``/World/envs/env_0``, ``/World/envs/env_1``).
    """

    clone_usd: bool = True
    """Enable USD-spec replication to author cloned prims and optional transforms."""

    clone_physics: bool = True
    """Enable PhysX replication for the same mapping to speed up physics setup."""

    physics_clone_fn: callable | None = None
    """Function used to perform physics replication."""

    clone_strategy: callable = random
    """Function used to build prototype-to-environment mapping. Default is :func:`random`."""

    device: str = "cpu"
    """Torch device on which mapping buffers are allocated."""

    clone_in_fabric: bool = False
    """Enable/disable cloning in fabric for PhysX replication. Default is False."""
