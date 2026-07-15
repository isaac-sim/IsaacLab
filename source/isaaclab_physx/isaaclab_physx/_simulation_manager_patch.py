# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lazy lifecycle patch for Isaac Sim's simulation manager."""

from __future__ import annotations

import sys
from contextlib import suppress
from types import ModuleType
from typing import Any


class _SimulationManagerPatch:
    """Own the late-loading Isaac Sim simulation-manager patch lifecycle.

    Isaac Sim's default STOP callback invalidates the tensor simulation view
    shared with :class:`~isaaclab_physx.physics.PhysxManager`. This owner disables
    those callbacks before replacing the module-level manager symbols and retains
    the extension-enable hook needed when Isaac Sim's manager loads later.
    """

    _DEFAULT_CALLBACK_ATTRIBUTES = (
        "_default_callback_warm_start",
        "_default_callback_on_stop",
        "_default_callback_stage_open",
        "_default_callback_stage_close",
    )

    def __init__(self):
        self._extension_enable_hook: object | None = None

    def claim_physics_lifecycle(self) -> None:
        """Patch an already-loaded manager and subscribe for late enablement."""
        self._patch_if_loaded()
        self._subscribe_to_enable()

    def _patch_if_loaded(self) -> None:
        """Replace Isaac Sim's manager after disabling its lifecycle callbacks.

        The patch is intentionally lazy. Config-only imports can occur before Kit
        starts, and optional extensions can load Isaac Sim's manager after
        :mod:`isaaclab_physx`.
        """
        original_module = sys.modules.get("isaacsim.core.simulation_manager")
        if original_module is None:
            return

        from .physics.physx_manager import IsaacEvents, PhysxManager

        original_class = self._get_original_class(original_module, PhysxManager)
        if original_class is not None and original_class is not PhysxManager:
            self._disable_default_callbacks(original_class)

        original_module.SimulationManager = PhysxManager
        original_module.IsaacEvents = IsaacEvents

    def _subscribe_to_enable(self) -> None:
        """Subscribe once to late enablement of Isaac Sim's manager extension."""
        if self._extension_enable_hook is not None:
            return

        app = self._get_kit_app()
        if app is None:
            return

        extension_manager = app.get_extension_manager()
        self._extension_enable_hook = extension_manager.subscribe_to_extension_enable(
            on_enable_fn=lambda _: self._patch_if_loaded(),
            on_disable_fn=lambda _: None,
            ext_name="isaacsim.core.simulation_manager",
            hook_name="isaaclab_physx simulation manager lifecycle patch",
        )

    @staticmethod
    def _get_kit_app() -> Any | None:
        """Return Kit's app interface when the runtime is ready."""
        kit_app = sys.modules.get("omni.kit.app")
        if kit_app is not None:
            # The Python module can exist before the Kit IApp interface, such as
            # during config-only pytest collection.
            with suppress(RuntimeError):
                return kit_app.get_app()
        return None

    @staticmethod
    def _get_original_class(original_module: ModuleType, physx_manager: type) -> type | None:
        """Return the Isaac Sim implementation class, including after a reload."""
        original_class = getattr(original_module, "SimulationManager", None)
        if original_class is not physx_manager:
            return original_class

        implementation_module = sys.modules.get("isaacsim.core.simulation_manager.impl.simulation_manager")
        return getattr(implementation_module, "SimulationManager", None)

    @classmethod
    def _disable_default_callbacks(cls, original_class: type):
        """Disable callbacks without assuming a specific Isaac Sim API version."""
        disable_callbacks = getattr(original_class, "enable_all_default_callbacks", None)
        if callable(disable_callbacks):
            disable_callbacks(False)
            return

        for attribute_name in cls._DEFAULT_CALLBACK_ATTRIBUTES:
            if hasattr(original_class, attribute_name):
                setattr(original_class, attribute_name, None)
