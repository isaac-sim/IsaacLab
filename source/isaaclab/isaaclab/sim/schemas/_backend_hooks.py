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

# Backend-registered creators that fix an articulation base by authoring a world<->root fixed joint,
# keyed by the backend's physics-cfg class (the type of ``SimulationContext.cfg.physics``). Creating it
# is backend-specific (PhysX relocates the articulation root to the parent prim to work around its
# parser; Newton reads the fixed joint directly), so each backend registers its own creator under its
# cfg type. The writer looks up the creator for the active simulation's ``cfg.physics`` type -- a plain
# dict lookup that never imports or executes a non-active backend, and "active" comes solely from the
# live ``cfg.physics`` rather than from which backends happen to be imported.
_FIXED_ROOT_JOINT_CREATORS: dict[type, Callable] = {}


def register_fixed_root_joint_creator(physics_cfg: type, creator: Callable) -> None:
    """Register the creator that fixes an articulation base for a given physics-backend cfg type.

    Called by :func:`~isaaclab.sim.schemas.apply_articulation_root_properties` when ``fix_root_link``
    is True and no fixed joint yet exists. Each physics backend (e.g. ``isaaclab_physx``,
    ``isaaclab_newton``) registers its own creator keyed by its physics-cfg class; the writer looks up
    the creator for the active simulation's ``cfg.physics`` type, so core selects the right backend
    without importing or probing any of them.

    Args:
        physics_cfg: The backend's physics-cfg class (the type of ``SimulationContext.cfg.physics`` for
            that backend), used as the lookup key.
        creator: A callable ``creator(articulation_prim, stage)`` that authors the fixed joint.
    """
    _FIXED_ROOT_JOINT_CREATORS[physics_cfg] = creator


def _resolve_fixed_root_joint_creator(physics_cfg: type | None) -> Callable | None:
    """Return the creator registered for the active backend's physics-cfg type, or None.

    Args:
        physics_cfg: The type of the active simulation's ``cfg.physics`` (or None if no simulation).
    """
    if physics_cfg is None:
        return None
    return _FIXED_ROOT_JOINT_CREATORS.get(physics_cfg)
