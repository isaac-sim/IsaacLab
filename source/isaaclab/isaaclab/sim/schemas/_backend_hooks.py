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

# Backend-registered predicates that exclude a joint prim from joint-drive authoring. Backends (e.g.
# PhysX tendons) register here via :func:`register_joint_drive_skip_predicate` so the core joint-drive
# writers can skip backend-controlled joints without core carrying any backend-specific schema name.
_JOINT_DRIVE_SKIP_PREDICATES: list[Callable] = []


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


def _skip_joint_drive(prim) -> bool:
    """Return whether any backend-registered predicate excludes ``prim`` from joint-drive authoring."""
    return any(predicate(prim) for predicate in _JOINT_DRIVE_SKIP_PREDICATES)


# Backend applied schemas that travel with an articulation root when a backend relocates it.
# Newton ships some of these as unregistered token schemas, which the USD schema registry cannot
# resolve, so the relocation helper has no way to discover their namespace on its own. Backends
# register the pairing here via :func:`register_articulation_root_companion`.
_ARTICULATION_ROOT_COMPANIONS: dict[str, str] = {}


def register_articulation_root_companion(schema_name: str, namespace: str) -> None:
    """Register a backend applied schema that belongs with an articulation root.

    When a backend relocates an articulation root to another prim, every schema that describes the
    root must move with it, together with the attributes it owns. Registered applied schemas are
    resolved through the USD schema registry, but a backend may ship a schema as an unregistered
    token, which the registry cannot describe. Registering it here supplies the namespace the
    relocation helper needs to carry the schema's authored attributes across.

    Args:
        schema_name: The applied schema name, e.g. ``"NewtonArticulationRootAPI"``.
        namespace: The attribute namespace the schema owns, e.g. ``"newton"``.
    """
    _ARTICULATION_ROOT_COMPANIONS[schema_name] = namespace


def _articulation_root_companion_namespace(schema_name: str) -> str | None:
    """Return the registered attribute namespace for an articulation-root companion schema."""
    return _ARTICULATION_ROOT_COMPANIONS.get(schema_name)
