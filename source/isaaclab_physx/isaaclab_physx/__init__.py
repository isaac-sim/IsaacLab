# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing the PhysX simulation interfaces for IsaacLab core package."""

import importlib.metadata
import sys


_simulation_manager_hook: object | None = None


try:
    __version__ = importlib.metadata.version("isaaclab_physx")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"


def _patch_isaacsim_simulation_manager():
    """Patch Isaac Sim's ``SimulationManager`` to use :class:`PhysxManager`.

    This redirects future ``from isaacsim.core.simulation_manager import SimulationManager``
    consumers to :class:`isaaclab_physx.physics.PhysxManager`, but the original
    Isaac Sim ``SimulationManager`` class has *already* registered timeline
    (PLAY/STOP) and stage (OPENED/CLOSED) subscriptions during its extension
    startup. Those subscriptions live on the original class, not the module
    attribute, so swapping the attribute alone is not enough.

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

    To prevent this, we disable the original class's default callbacks here
    *before* swapping the module attribute, so :class:`PhysxManager` becomes
    the single owner of the simulation lifecycle.

    This function is intentionally lazy: it only patches if
    ``isaacsim.core.simulation_manager`` is already present in ``sys.modules``.
    If :mod:`isaaclab_physx` is imported before Kit has launched or before an
    optional extension loads Isaac Sim's manager, this function is a no-op. The
    extension-enable subscription installed by
    :func:`_subscribe_to_simulation_manager_enable` applies the patch when the
    manager is loaded later.
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
    original_class = getattr(original_module, "SimulationManager", None)
    if original_class is PhysxManager:
        # Extension reloads run the original implementation's startup again, but
        # the package-level symbol remains patched. Recover the implementation
        # class so its newly registered callbacks can be disabled again.
        implementation_module = sys.modules.get("isaacsim.core.simulation_manager.impl.simulation_manager")
        original_class = getattr(implementation_module, "SimulationManager", None)
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


def _subscribe_to_simulation_manager_enable():
    """Patch Isaac Sim's simulation manager whenever its extension is enabled.

    The extension may be loaded after :mod:`isaaclab_physx` by an optional
    dependency, such as the surface gripper. Keeping this subscription alive
    prevents that late load from registering a second owner of the PhysX
    simulation lifecycle without loading the extension for other workflows.
    """
    global _simulation_manager_hook

    if _simulation_manager_hook is not None:
        return

    # Do not import Kit here. Config-only imports must remain usable before the
    # app launches; PhysxManager.initialize() retries this once Kit is available.
    kit_app = sys.modules.get("omni.kit.app")
    if kit_app is None:
        return

    app = kit_app.get_app()
    if app is None:
        return

    extension_manager = app.get_extension_manager()
    _simulation_manager_hook = extension_manager.subscribe_to_extension_enable(
        on_enable_fn=lambda _: _patch_isaacsim_simulation_manager(),
        on_disable_fn=lambda _: None,
        ext_name="isaacsim.core.simulation_manager",
        hook_name="isaaclab_physx simulation manager lifecycle patch",
    )


_patch_isaacsim_simulation_manager()
_subscribe_to_simulation_manager_enable()
