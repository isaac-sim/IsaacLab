# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing the PhysX simulation interfaces for IsaacLab core package."""

import importlib.metadata
import sys
from contextlib import suppress
from types import ModuleType
from typing import Any

try:
    __version__ = importlib.metadata.version("isaaclab_physx")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

_SIMULATION_MANAGER_ENABLE_HOOK: Any | None = None


def _patch_isaacsim_simulation_manager() -> None:
    """Patch legacy Isaac Sim's ``SimulationManager`` to use :class:`PhysxManager`.

    New Isaac Sim versions support disabling their default lifecycle callbacks
    before extension startup. In that case, Isaac Lab leaves the original class
    and module exports untouched. Older versions register timeline (PLAY/STOP)
    and stage (OPENED/CLOSED) subscriptions unconditionally, so this function
    disables those callbacks and redirects future imports to
    :class:`isaaclab_physx.physics.PhysxManager`.

    Starting with Isaac Sim 6.0.0-alpha.180 (commit ``8df6beeb0`` on
    ``develop``, "hmazhar/autofix_bugs"), the original
    ``SimulationManager._on_stop``/``_on_play``/``_on_stage_*`` methods were
    decorated with ``@staticmethod`` so they finally fire correctly from the
    Carb event subscriptions. Before that fix they were silently broken (the
    subscriptions invoked them as bound methods, so the ``event`` argument was
    being passed as ``self``/``cls`` and the bodies never executed).

    The newly-working ``_on_stop`` calls
    ``SimulationManager.invalidate_physics()``, which calls
    ``view.invalidate()`` on its ``omni.physics.tensors`` simulation view.
    Because ``omni.physics.tensors.create_simulation_view("warp", stage_id=...)``
    returns the same underlying SimulationView per stage_id, that invalidation
    also wrecks the view that :class:`PhysxManager` (and any articulation
    ``_root_view`` derived from it) relies on. The result is the runtime error
    ``Simulation view object is invalidated and cannot be used again to call
    getDofVelocities`` on the very first ``scene.update()`` after
    ``sim.reset()``.

    To prevent this on older versions, we disable the original class's default
    callbacks here *before* swapping the module attribute, so
    :class:`PhysxManager` becomes the single owner of the simulation lifecycle.

    This function is intentionally lazy. Config loading may import
    :mod:`isaaclab_physx` before Kit has launched; in that case, there is no app
    interface and no Isaac Sim callbacks to patch yet.

    :meth:`PhysxManager.initialize` subscribes to extension enablement so the
    patch is re-applied if optional Isaac Sim code loads the manager later.
    """
    original_module = sys.modules.get("isaacsim.core.simulation_manager")
    if original_module is None:
        return

    from .physics.physx_manager import IsaacEvents, PhysxManager

    # Tear down the original Isaac Sim SimulationManager's default timeline /
    # stage subscriptions so they cannot invalidate the omni.physics.tensors
    # view that PhysxManager owns. ``enable_all_default_callbacks(False)``
    # covers warm_start (PLAY), on_stop (STOP), stage_open (OPENED) and
    # stage_close (CLOSED). Older Isaac Sim builds may not expose this API, so
    # fall back gracefully.
    original_class = _get_original_simulation_manager_class(original_module, PhysxManager)
    if original_class is not None and _default_callbacks_disabled_at_startup(original_class):
        return

    if original_class is not None and original_class is not PhysxManager:
        try:
            original_class.enable_all_default_callbacks(False)
        except Exception:
            # Defensive: API changed or original class never finished startup.
            # Manually clear the subscription handles if they exist so any
            # remaining references go through the dead-callback path.
            for attr in (
                "_default_callback_warm_start",
                "_default_callback_on_stop",
                "_default_callback_stage_open",
                "_default_callback_stage_close",
            ):
                if hasattr(original_class, attr):
                    setattr(original_class, attr, None)

    original_module.SimulationManager = PhysxManager
    original_module.IsaacEvents = IsaacEvents


def _default_callbacks_disabled_at_startup(original_class: type) -> bool:
    """Return whether Isaac Sim supports and honored its startup callback setting."""
    implementation_module = sys.modules.get("isaacsim.core.simulation_manager.impl.simulation_manager")
    if implementation_module is None or not hasattr(
        implementation_module, "_SETTING_ENABLE_DEFAULT_CALLBACKS"
    ):
        return False

    try:
        return not any(original_class.get_default_callback_status().values())
    except (AttributeError, TypeError):
        return False


def _get_original_simulation_manager_class(original_module: ModuleType, physx_manager: type) -> type | None:
    """Return Isaac Sim's implementation class, including after a module-level patch."""
    original_class = getattr(original_module, "SimulationManager", None)
    if original_class is not physx_manager:
        return original_class

    implementation_module = sys.modules.get("isaacsim.core.simulation_manager.impl.simulation_manager")
    return getattr(implementation_module, "SimulationManager", None)


def _get_kit_extension_manager() -> Any | None:
    """Return Kit's extension manager when Kit is running."""
    try:
        import omni.kit.app  # noqa: PLC0415
    except Exception:
        return None

    with suppress(RuntimeError):
        app = omni.kit.app.get_app()
        return app.get_extension_manager()
    return None


def _subscribe_to_simulation_manager_enable() -> None:
    """Patch Isaac Sim's manager if it is enabled after PhysxManager starts."""
    global _SIMULATION_MANAGER_ENABLE_HOOK
    if _SIMULATION_MANAGER_ENABLE_HOOK is not None:
        return

    extension_manager = _get_kit_extension_manager()
    if extension_manager is None:
        return

    _SIMULATION_MANAGER_ENABLE_HOOK = extension_manager.subscribe_to_extension_enable(
        on_enable_fn=lambda _: _patch_isaacsim_simulation_manager(),
        on_disable_fn=lambda _: None,
        ext_name="isaacsim.core.simulation_manager",
        hook_name="isaaclab_physx simulation manager lifecycle patch",
    )


_patch_isaacsim_simulation_manager()
