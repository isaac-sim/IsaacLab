# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing the PhysX simulation interfaces for IsaacLab core package."""

import os
import sys
import toml

# Conveniences to other module directories via relative paths
ISAACLAB_PHYSX_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_PHYSX_METADATA = toml.load(os.path.join(ISAACLAB_PHYSX_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_PHYSX_METADATA["package"]["version"]


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
    In the normal production flow Kit loads that module during extension startup,
    before any user script imports :mod:`isaaclab_physx`, so the condition is
    true and the patch fires on time. If :mod:`isaaclab_physx` happens to be
    imported for pure config loading before Kit has launched (e.g. in
    ``test_env_cfg_no_forbidden_imports``), the module is absent and this
    function is a no-op — which is correct, because no callbacks have been
    registered yet.
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


_patch_isaacsim_simulation_manager()
