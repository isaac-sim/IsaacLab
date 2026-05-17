# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-dispatching FrameView.

``FrameView(path, device=...)`` automatically selects the right backend:
- PhysX + Fabric enabled + supported device: :class:`~isaaclab_physx.sim.views.FabricFrameView`
- PhysX without Fabric (or unsupported device): :class:`~isaaclab.sim.views.UsdFrameView`
- OVPhysX: :class:`~isaaclab_ovphysx.sim.views.OvPhysxFrameView`
- Newton: :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView`
"""

from __future__ import annotations

import logging

from isaaclab.utils.backend_utils import FactoryBase

from .base_frame_view import BaseFrameView
from .usd_frame_view import UsdFrameView

logger = logging.getLogger(__name__)

# Devices supported by Fabric GPU acceleration (USDRT SelectPrims + Warp).
_FABRIC_SUPPORTED_DEVICES = ("cpu", "cuda", "cuda:0")


class FrameView(FactoryBase, BaseFrameView):
    """FrameView that dispatches to the active physics backend.

    Callers use ``FrameView(prim_path, device=device)`` and get the
    correct implementation automatically:

    - **PhysX + Fabric**: :class:`~isaaclab_physx.sim.views.FabricFrameView`
      (GPU-accelerated transforms via Warp + USDRT).
    - **PhysX without Fabric**: :class:`~isaaclab.sim.views.UsdFrameView`
      (standard USD operations).
    - **OVPhysX**: :class:`~isaaclab_ovphysx.sim.views.OvPhysxFrameView`
      (Warp-native, reads body poses via an OVPhysX ``RIGID_BODY_POSE``
      tensor binding).
    - **Newton**: :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView`
      (Warp-native, reads ``body_q`` from the Newton state).
    """

    _backend_class_names = {
        "physx": "FabricFrameView",
        "ovphysx": "OvPhysxFrameView",
        "newton": "NewtonSiteFrameView",
        # "usd" is registered eagerly below — no dynamic import needed.
    }

    @classmethod
    def _get_backend(cls, *args, **kwargs) -> str:
        from isaaclab.app.settings_manager import SettingsManager  # noqa: PLC0415
        from isaaclab.sim.simulation_context import SimulationContext  # noqa: PLC0415

        ctx = SimulationContext.instance()
        if ctx is None:
            return "usd"

        manager_name = ctx.physics_manager.__name__.lower()
        if "newton" in manager_name:
            return "newton"
        if "ovphysx" in manager_name:
            return "ovphysx"

        # PhysX path — check if Fabric is enabled and the device is supported.
        settings = SettingsManager.instance()
        fabric_enabled = bool(settings.get("/physics/fabricEnabled", False))

        device = kwargs.get("device", "cpu")
        if len(args) >= 2:
            device = args[1]

        if fabric_enabled and device in _FABRIC_SUPPORTED_DEVICES:
            return "physx"

        if fabric_enabled and device not in _FABRIC_SUPPORTED_DEVICES:
            logger.warning(
                f"Fabric mode is not supported on device '{device}'. "
                "USDRT SelectPrims and Warp fabric arrays are currently "
                f"only supported on {', '.join(_FABRIC_SUPPORTED_DEVICES)}. "
                "Falling back to UsdFrameView."
            )

        return "usd"

    def __new__(cls, *args, **kwargs) -> BaseFrameView:
        """Create a new FrameView for the active physics backend."""
        return super().__new__(cls, *args, **kwargs)


# Eagerly register UsdFrameView — it lives in isaaclab, not a backend package,
# so FactoryBase's dynamic import (isaaclab_{backend}.sim.views) can't find it.
FrameView.register("usd", UsdFrameView)
