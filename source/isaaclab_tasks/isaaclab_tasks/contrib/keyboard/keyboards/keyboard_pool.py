# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedural keyboard variants and slot metadata used by the SO-101 task."""

from __future__ import annotations

from dataclasses import dataclass

from .keyboard_gen_cfg import KeyboardSpawnerCfg
from .keyboard_geometry import base_profile_visuals, generate_keyboard

_KEYBOARD_DOF = 108
_KEYBOARD_VARIANTS = 32
_PARTITION_DOF = 6


@dataclass(frozen=True)
class _TypingKeyboardPool:
    """Spawner configurations and key metadata consumed by the task."""

    spawners_partitioned: tuple[KeyboardSpawnerCfg, ...]
    spawners_single: tuple[KeyboardSpawnerCfg, ...]
    backspace_slot: int
    active_slots: tuple[int, ...]
    slot_labels: tuple[str, ...]


_BASE_SPAWNER = KeyboardSpawnerCfg(family="ansi_full", topology_mode="exact", partition_dof=_PARTITION_DOF)
_SPAWNERS_SINGLE = tuple(_BASE_SPAWNER.replace(seed=seed) for seed in range(_KEYBOARD_VARIANTS))
_SPAWNERS_PARTITIONED = tuple(cfg.replace(partition_mode="fixed_dof") for cfg in _SPAWNERS_SINGLE)
_KEYBOARDS = tuple(generate_keyboard(cfg) for cfg in _SPAWNERS_SINGLE)

for keyboard in _KEYBOARDS:
    if keyboard.active_key_count != _KEYBOARD_DOF or keyboard.slot_count != _KEYBOARD_DOF:
        raise ValueError(
            f"Keyboard variant seed {keyboard.seed} produced {keyboard.active_key_count} active keys and "
            f"{keyboard.slot_count} slots; expected {_KEYBOARD_DOF}."
        )

_trim_counts = {len(base_profile_visuals(keyboard)) for keyboard in _KEYBOARDS}
if len(_trim_counts) != 1:
    raise ValueError(f"Keyboard variants have non-uniform case-trim counts: {sorted(_trim_counts)}.")

_REFERENCE_KEYBOARD = _KEYBOARDS[0]
TYPING_KEYBOARD_POOL = _TypingKeyboardPool(
    spawners_partitioned=_SPAWNERS_PARTITIONED,
    spawners_single=_SPAWNERS_SINGLE,
    backspace_slot=next(key.slot for key in _REFERENCE_KEYBOARD.keys if key.label.lower() == "backspace"),
    active_slots=tuple(key.slot for key in _REFERENCE_KEYBOARD.keys if key.active),
    slot_labels=tuple(key.label for key in _REFERENCE_KEYBOARD.keys),
)
