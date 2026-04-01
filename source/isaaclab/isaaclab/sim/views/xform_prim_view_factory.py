# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory for creating backend-specific XformPrimView instances."""

from __future__ import annotations

from isaaclab.utils.backend_utils import FactoryBase

from .base_xform_prim_view import BaseXformPrimView


class XformPrimViewFactory(FactoryBase, BaseXformPrimView):
    """Factory that selects a :class:`BaseXformPrimView` implementation based on the active physics backend.

    - **PhysX / no backend**: returns the USD/Fabric-based
      :class:`~isaaclab.sim.views.XformPrimView`.
    - **Newton**: returns the GPU-state-backed
      :class:`~isaaclab_newton.sim.views.XformPrimView`.

    Use this in place of constructing an ``XformPrimView`` directly when the
    caller is physics-aware (e.g. ray-cast sensors) and wants the fastest
    available implementation for the active backend.
    """

    _backend_class_names = {"physx": "XformPrimView", "newton": "XformPrimView"}

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
