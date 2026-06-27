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

# Backend-registered creators that fix an articulation base by authoring a world<->root fixed joint.
# Creating it is backend-specific (PhysX requires relocating the articulation root to the parent prim
# to work around its parser; Newton reads the fixed joint directly), so each backend registers its own
# creator plus a predicate that reports when that backend is the active simulation. This keeps core
# free of backend knowledge while letting each backend run on its own.
_FIXED_ROOT_JOINT_CREATORS: list[tuple[Callable, Callable]] = []


def register_fixed_root_joint_creator(creator: Callable, is_active: Callable) -> None:
    """Register a backend creator that fixes an articulation base via a world<->root fixed joint.

    Called by :func:`~isaaclab.sim.schemas.apply_articulation_root_properties` when ``fix_root_link``
    is True and no fixed joint yet exists. Multiple backends may register; the creator whose
    ``is_active()`` predicate returns True for the running simulation is selected, so each physics
    backend (e.g. ``isaaclab_physx``, ``isaaclab_newton``) provides and is matched to its own
    implementation without core knowing any of them.

    Args:
        creator: A callable ``creator(articulation_prim, stage)`` that authors the fixed joint.
        is_active: A callable ``is_active() -> bool`` returning True when ``creator``'s backend is the
            active simulation backend.
    """
    _FIXED_ROOT_JOINT_CREATORS.append((is_active, creator))


def _resolve_fixed_root_joint_creator():
    """Return the creator whose backend is active, or None if no registered backend matches."""
    for is_active, creator in _FIXED_ROOT_JOINT_CREATORS:
        try:
            if is_active():
                return creator
        except Exception:  # noqa: BLE001 -- a backend's probe must never break creator resolution
            continue
    return None
