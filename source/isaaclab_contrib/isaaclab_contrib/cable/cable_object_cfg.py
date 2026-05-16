# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the cable asset class."""

from __future__ import annotations

from isaaclab.actuators import ActuatorBaseCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.utils.configclass import configclass


@configclass
class CableObjectCfg(ArticulationCfg):
    """Configuration for a cable / 1D-rod asset (Newton backend).

    Inherits all of :class:`ArticulationCfg` and overrides two defaults so the
    base :meth:`Articulation._initialize_impl` runs unchanged on cables:

    - ``articulation_root_prim_path = "/cable_articulation"`` — the sub-label
      that :meth:`newton.ModelBuilder.add_rod_graph` produces under the cable's
      source prim path (``f"{label}_articulation"`` where ``label`` is
      ``"{prim_path}/cable"``). The base method composes this with
      ``cfg.prim_path`` and uses the result as the label pattern for
      :class:`newton.selection.ArticulationView`.
    - ``actuators = {}`` — cables have no user-defined actuators (cable joint
      stiffness is material-like, applied internally by the solver). The
      inherited ``_process_actuators_cfg`` iterates an empty dict safely and
      emits a harmless ``logger.warning("Not all actuators are configured!")``
      — expected and not suppressed in Phase 1.
    """

    class_type: type = "{DIR}.cable_object:CableObject"
    articulation_root_prim_path: str | None = "/cable_articulation"
    actuators: dict[str, ActuatorBaseCfg] = {}
