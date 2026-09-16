# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for procedural SO-101 keyboards."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils import config_field

from .keyboard_schema import (
    DEFAULT_BUCKET_SIZES,
    DEFAULT_FAMILIES,
    DEFAULT_MAX_SLOTS,
    DEFAULT_PARTITION_DOF,
    KeyboardFamily,
    PartitionMode,
    PartitionTailPolicy,
    TopologyMode,
)


@dataclass
class KeyboardSpawnerCfg(SpawnerCfg):
    """Configuration for one deterministic procedural keyboard.

    The spawner supports a single articulation rooted at the requested prim or fixed-DOF
    articulations under ``parts/part_*``.
    """

    func: Callable | str = config_field("{DIR}.keyboard_usd:spawn_keyboard")

    family: KeyboardFamily = config_field("random")
    """Keyboard layout family. ``"random"`` samples from :attr:`families`."""

    families: tuple[str, ...] = config_field(DEFAULT_FAMILIES)
    """Candidate families when :attr:`family` is ``"random"``."""

    style: str | None = config_field(None)
    """Optional style archetype name. ``None`` samples a compatible style."""

    seed: int = config_field(0)
    """Seed for deterministic generation."""

    topology_mode: TopologyMode = config_field("exact")
    """Key-slot topology. Padded modes are intended for ablation benchmarks."""

    max_slots: int = config_field(DEFAULT_MAX_SLOTS)
    """Maximum key slots for ``global_padded`` topology."""

    bucket_sizes: tuple[int, ...] = config_field(DEFAULT_BUCKET_SIZES)
    """Canonical slot counts for ``bucketed`` topology."""

    partition_mode: PartitionMode = config_field("single")
    """Articulation partitioning mode."""

    partition_dof: int = config_field(DEFAULT_PARTITION_DOF)
    """DOF per fixed-DOF partition."""

    partition_tail_policy: PartitionTailPolicy = config_field("pad_tail")
    """Tail policy for fixed-DOF partitions."""

    use_tapered_keycaps: bool = config_field(True)
    """Whether keycaps use tapered visual meshes."""

    include_case_collision: bool = config_field(True)
    """Whether the generated case collides with the robot."""

    key_collision_margin: float = config_field(0.0015)
    """Inset applied to key collision boxes [m]."""

    case_collision_margin: float = config_field(0.001)
    """Inset applied to the case collision box [m]."""

    strict_validation: bool = config_field(True)
    """Whether structural validation errors raise an exception."""
