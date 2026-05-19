# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the cable asset class."""

from __future__ import annotations
from dataclasses import MISSING
from typing import Literal

from isaaclab.actuators import ActuatorBaseCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.utils.configclass import configclass


@configclass
class CableAttachmentCfg:
    """Weld a cable endpoint to a body on another spawned asset.

    The attachment creates a Newton fixed joint between one of the cable's end
    rod-segment bodies and a body on a separately spawned rigid asset. The joint
    is realized at Newton model-build time, after both assets are registered
    with the builder. Newton's rigid solver then enforces the constraint
    natively each step; no per-step Python synchronization is required.
    """

    target_prim_path: str = MISSING
    """Prim path of the rigid body to weld the cable endpoint to.

    Must resolve to a prim that has been registered with Newton as a rigid body
    (e.g., spawned via :class:`~isaaclab.assets.RigidObject`) prior to the cable
    being realized. Regex patterns are not supported here; the path must be the
    same concrete spawn path the rigid asset's :class:`RigidObjectCfg` uses
    (the per-world resolver matches it against ``builder.body_label`` at
    builder-hook time).
    """

    cable_anchor: Literal["head", "tail"] = "tail"
    """Which end of the cable to anchor.

    ``"head"`` is the first rod-segment body (corresponding to the BasisCurves
    point at index 0). ``"tail"`` is the last rod-segment body. The internal
    resolver maps this symbolic name to the Newton body index recorded on the
    cable's registry entry at :meth:`newton.ModelBuilder.add_rod_graph` time.
    """

    local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Joint anchor position [m] in the target body's local frame."""

    local_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Joint anchor orientation as quaternion ``(w, x, y, z)`` in the target body's local frame."""


@configclass
class CableObjectCfg(ArticulationCfg):
    """Configuration for a cable / 1D-rod asset (Newton backend).

    Inherits all of :class:`ArticulationCfg` and overrides two defaults so the
    base :meth:`Articulation._initialize_impl` runs unchanged on cables. See
    :attr:`articulation_root_prim_path` and :attr:`actuators` for the rationale
    behind each override.
    """

    class_type: type | str = "{DIR}.cable_object:CableObject"

    articulation_root_prim_path: str | None = "/cable_articulation"
    """Sub-label produced by :meth:`newton.ModelBuilder.add_rod_graph` under the
    cable's source prim (``f"{label}_articulation"`` with ``label =
    "{prim_path}/cable"``). Overrides the base default (``None``, which would
    trigger a ``UsdPhysics.ArticulationRootAPI`` stage search) because Newton
    rod-graph cables don't author that schema. The base ``_initialize_impl``
    composes this with :attr:`prim_path` to build the
    :class:`newton.selection.ArticulationView` selector."""

    actuators: dict[str, ActuatorBaseCfg] = {}
    """Empty by design: cables have no user-defined actuators (joint stiffness
    is material-like, applied internally by the solver). Overrides the base
    ``MISSING`` default so the inherited ``_process_actuators_cfg`` iterates an
    empty dict instead of crashing on ``MISSING``; a harmless
    ``logger.warning("Not all actuators are configured!")`` is expected."""

    attachments: list[CableAttachmentCfg] = []
