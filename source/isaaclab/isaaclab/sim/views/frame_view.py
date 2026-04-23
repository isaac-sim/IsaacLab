# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-dispatching FrameView.

``FrameView(path, device=...)`` automatically selects the right backend:
- PhysX: :class:`~isaaclab_physx.sim.views.FabricFrameView`
- Newton: :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView`
"""

from __future__ import annotations

from isaaclab.utils.backend_utils import FactoryBase

from .base_frame_view import BaseFrameView


class FrameView(FactoryBase, BaseFrameView):
    """FrameView that dispatches to the active physics backend.

    Callers use ``FrameView(prim_path, device=device)`` and get the
    correct implementation automatically:

    - **PhysX / no backend**: :class:`~isaaclab_physx.sim.views.FabricFrameView`
      (Fabric GPU acceleration with USD fallback).
    - **Newton**: :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView`
      (GPU-resident site-based transforms).
    """

    _backend_class_names = {"physx": "FabricFrameView", "newton": "NewtonSiteFrameView"}

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

    def __new__(cls, *args, **kwargs) -> BaseFrameView:
        """Create a new FrameView for the active physics backend."""
        return super().__new__(cls, *args, **kwargs)
