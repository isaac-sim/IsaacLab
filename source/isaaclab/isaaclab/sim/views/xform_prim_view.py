# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-dispatching XformPrimView.

``XformPrimView(path, device=...)`` automatically selects the right backend:
- PhysX: :class:`~isaaclab_physx.sim.views.FabricXformPrimView`
- Newton: :class:`~isaaclab_newton.sim.views.NewtonSiteXformPrimView`
"""

from __future__ import annotations

from isaaclab.utils.backend_utils import FactoryBase

from .base_xform_prim_view import BaseXformPrimView


class XformPrimView(FactoryBase, BaseXformPrimView):
    """XformPrimView that dispatches to the active physics backend.

    Callers use ``XformPrimView(prim_path, device=device)`` and get the
    correct implementation automatically:

    - **PhysX / no backend**: :class:`~isaaclab_physx.sim.views.FabricXformPrimView`
      (Fabric GPU acceleration with USD fallback).
    - **Newton**: :class:`~isaaclab_newton.sim.views.NewtonSiteXformPrimView`
      (GPU-resident site-based transforms).
    """

    _backend_class_names = {"physx": "FabricXformPrimView", "newton": "NewtonSiteXformPrimView"}

    @classmethod
    def _get_backend(cls, *args, **kwargs) -> str:
        from isaaclab.sim.simulation_context import SimulationContext  # noqa: PLC0415

        ctx = SimulationContext.instance()
        if ctx is None:
            return "physx"
        manager_name = ctx.physics_manager.__name__.lower()
        if "newton" in manager_name:
            return "newton"
        return "physx"

    def __new__(cls, *args, **kwargs) -> BaseXformPrimView:
        """Create a new XformPrimView for the active physics backend."""
        return super().__new__(cls, *args, **kwargs)
