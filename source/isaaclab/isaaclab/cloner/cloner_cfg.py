# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable

from isaaclab.utils.configclass import configclass

from .cloner_strategies import random


@configclass
class CloneCfg:
    """Configuration for environment replication.

    Holds the knobs :class:`~isaaclab.scene.InteractiveScene` forwards to
    :func:`~isaaclab.cloner.make_clone_plan` when building per-env layouts.
    """

    clone_strategy: Callable[..., object] = random
    """Function used to build prototype-to-environment mapping. Default is :func:`random`."""

    device: str = "cpu"
    """Torch device on which mapping buffers are allocated."""

    clone_regex: str = "/World/envs/env_.*"
    """Regex matching every replicated env prim. Used to expand ``{ENV_REGEX_NS}`` cfg macros."""
