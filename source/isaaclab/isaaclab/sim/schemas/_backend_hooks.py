# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend registration hooks for schema writers.

This module holds the inversion-of-control registries that let physics backends (e.g.
``isaaclab_physx``, ``isaaclab_newton``) inject backend-specific behaviour into the core schema
writers without core importing any backend. It is deliberately kept free of ``pxr``/``omni`` imports
so a backend can register its hook at package-import time without eagerly pulling USD libraries into
an otherwise USD-free import path.
"""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.utils.string import string_to_callable

# Backend-registered predicates that exclude a joint prim from joint-drive authoring. Backends (e.g.
# PhysX tendons) register here via :func:`register_joint_drive_skip_predicate` so the core joint-drive
# writers can skip backend-controlled joints without core carrying any backend-specific schema name.
_JOINT_DRIVE_SKIP_PREDICATES: list[Callable] = []

# Backend modifiers for legacy fixed-tendon cfgs. The compatibility writer calls these when a prim
# does not carry a PhysX fixed-tendon schema, allowing backends to preserve legacy behavior without
# placing their prim types or attribute namespaces in core.
_FIXED_TENDON_MODIFIERS: list[Callable | str] = []


def register_joint_drive_skip_predicate(predicate: Callable) -> None:
    """Register a predicate that excludes a joint prim from joint-drive authoring.

    The joint-drive writers (:func:`~isaaclab.sim.schemas.apply_drive`,
    :func:`~isaaclab.sim.schemas.apply_joint_drive_properties`) skip any joint for which a registered
    predicate returns ``True``. This is the backend hook for cases like PhysX fixed tendons, where the
    controlling backend owns certain joints and no drive should be authored on them -- the backend
    registers its own detector so core needs no backend-specific knowledge.

    Args:
        predicate: A callable ``predicate(prim) -> bool`` returning True to exclude the prim.
    """
    if predicate not in _JOINT_DRIVE_SKIP_PREDICATES:
        _JOINT_DRIVE_SKIP_PREDICATES.append(predicate)


def register_fixed_tendon_modifier(modifier: Callable | str) -> None:
    """Register a backend modifier for legacy fixed-tendon configurations.

    The compatibility writer invokes registered modifiers when a prim does not carry a PhysX
    fixed-tendon schema. A modifier returns ``True`` when it recognizes and updates the prim, or
    ``False`` to let another backend try.

    Args:
        modifier: A callable or resolvable callable string with the signature
            ``modifier(cfg, prim_path, stage) -> bool``.
    """
    if modifier not in _FIXED_TENDON_MODIFIERS:
        _FIXED_TENDON_MODIFIERS.append(modifier)


def _skip_joint_drive(prim) -> bool:
    """Return whether any backend-registered predicate excludes ``prim`` from joint-drive authoring."""
    return any(predicate(prim) for predicate in _JOINT_DRIVE_SKIP_PREDICATES)


def _modify_fixed_tendon_with_backend(cfg, prim_path: str, stage) -> bool:
    """Try backend-registered legacy fixed-tendon modifiers for a prim."""
    for modifier in _FIXED_TENDON_MODIFIERS:
        func = modifier if callable(modifier) else string_to_callable(modifier)
        if func(cfg, prim_path, stage):
            return True
    return False
