# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the cable asset class."""

from __future__ import annotations

from isaaclab.actuators import ActuatorBaseCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.utils.configclass import configclass

from .attachment_cfg import CableAttachmentCfg


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
