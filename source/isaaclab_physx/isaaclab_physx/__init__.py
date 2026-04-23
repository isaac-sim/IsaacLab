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

    To prevent this, we unsubscribe only the original class's ``_on_stop``
    callback *before* swapping the module attribute. Other callbacks
    (warm_start/PLAY, stage_open, stage_close) are left intact — in particular
    warm_start must fire so the rendering pipeline initialises correctly on
    ``sim.reset()``. Disabling it causes tiled-camera RGB output to stay black.
    """
    # Force-import Isaac Sim's SimulationManager before patching so that the
    # subscriptions registered during its module/extension startup are taken
    # down deterministically here, regardless of the order in which Kit
    # extensions or user code happen to import the module.
    try:
        import isaacsim.core.simulation_manager  # noqa: F401
    except ImportError:
        # Isaac Sim is not installed (e.g. during ``./isaaclab.sh --install``
        # bootstrap or in pure unit-test environments). Nothing to patch.
        return

    original_module = sys.modules["isaacsim.core.simulation_manager"]
    from .physics.physx_manager import PhysxManager, IsaacEvents

    # Only unsubscribe _on_stop — that is the sole callback that calls
    # ``invalidate_physics()`` and wrecks the shared omni.physics.tensors view.
    # Leaving warm_start (PLAY) intact ensures the rendering pipeline initialises
    # correctly when ``sim.reset()`` fires the play event; disabling it causes
    # tiled-camera RGB to stay black (see isaaclab_visualizers CI failure).
    original_class = getattr(original_module, "SimulationManager", None)
    if original_class is not None and original_class is not PhysxManager:
        if hasattr(original_class, "_default_callback_on_stop"):
            # Carb subscription objects unsubscribe on destruction — setting to
            # None drops the reference and silently cancels the subscription.
            original_class._default_callback_on_stop = None
        else:
            # _default_callback_on_stop not found (API change). Fall back to
            # disabling all callbacks. Note: this may cause tiled-camera black
            # frames on newer Isaac Sim builds; the targeted fix above is preferred.
            try:
                original_class.enable_all_default_callbacks(False)
            except Exception:
                pass

    original_module.SimulationManager = PhysxManager
    original_module.IsaacEvents = IsaacEvents


_patch_isaacsim_simulation_manager()
