# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental Newton deformable module.

Importing this module makes deformable config classes available immediately.
Heavy runtime imports (NewtonManager, DeformableObject, etc.) and hook
registration are deferred until first access so that ``pxr`` is not loaded
at config-resolution time (before Kit/SimulationApp is started).
"""

# Lightweight config-only imports -- no pxr dependency
from .newton_manager_cfg import CoupledSolverCfg, NewtonModelCfg, VBDSolverCfg

# ---------------------------------------------------------------------------
# Deferred heavy imports & hook registration
# ---------------------------------------------------------------------------
_hooks_registered = False

# Names that are deferred and resolved via __getattr__
_DEFERRED_NAMES = {
    "NewtonManager",
    "DeformableObject",
    "DeformableRegistryEntry",
    "DeformableObjectData",
    "CoupledSolver",
    "per_world_deformable_hook",
    "post_replicate_deformable_hook",
    "apply_model_cfg",
    "sync_particles_to_usd",
    "create_coupled_solver",
    "create_vbd_solver",
    "register_hooks",
}


def _do_deferred_imports():
    """Import all heavy symbols and register hooks.

    Called once on first access to any deferred name.
    """
    global _hooks_registered

    from isaaclab_newton.physics import NewtonManager

    from .cloner_hooks import per_world_deformable_hook, post_replicate_deformable_hook
    from .coupled_solver import CoupledSolver
    from .deformable_object import DeformableObject, DeformableRegistryEntry
    from .deformable_object_data import DeformableObjectData
    from .model_cfg_hook import apply_model_cfg
    from .particle_sync import setup_fabric_particle_sync, sync_particles_to_usd
    from .solver_factories import create_coupled_solver, create_vbd_solver

    # Inject into module namespace so subsequent accesses are direct
    g = globals()
    g["NewtonManager"] = NewtonManager
    g["DeformableObject"] = DeformableObject
    g["DeformableRegistryEntry"] = DeformableRegistryEntry
    g["DeformableObjectData"] = DeformableObjectData
    g["CoupledSolver"] = CoupledSolver
    g["per_world_deformable_hook"] = per_world_deformable_hook
    g["post_replicate_deformable_hook"] = post_replicate_deformable_hook
    g["apply_model_cfg"] = apply_model_cfg
    g["sync_particles_to_usd"] = sync_particles_to_usd
    g["setup_fabric_particle_sync"] = setup_fabric_particle_sync
    g["create_coupled_solver"] = create_coupled_solver
    g["create_vbd_solver"] = create_vbd_solver

    # Register hooks with NewtonManager
    if not _hooks_registered:
        _hooks_registered = True
        _register_hooks_impl(NewtonManager, create_vbd_solver, create_coupled_solver,
                             sync_particles_to_usd, setup_fabric_particle_sync,
                             per_world_deformable_hook,
                             post_replicate_deformable_hook, apply_model_cfg)

    # Register the Newton DeformableObject backend with the factory
    from isaaclab.assets.deformable_object.deformable_object import DeformableObject as DeformableObjectFactory
    DeformableObjectFactory.register("newton", DeformableObject)


def _register_hooks_impl(NewtonManager, create_vbd_solver, create_coupled_solver,
                          sync_particles_to_usd, setup_fabric_particle_sync,
                          per_world_deformable_hook,
                          post_replicate_deformable_hook, apply_model_cfg):
    """Register all deformable hooks with NewtonManager."""
    NewtonManager.register_solver_factory("vbd", create_vbd_solver)
    NewtonManager.register_solver_factory("coupled", create_coupled_solver)
    NewtonManager._particle_sync_fn = sync_particles_to_usd
    NewtonManager._post_start_simulation_fn = setup_fabric_particle_sync
    if per_world_deformable_hook not in NewtonManager._per_world_builder_hooks:
        NewtonManager._per_world_builder_hooks.append(per_world_deformable_hook)
    if post_replicate_deformable_hook not in NewtonManager._post_replicate_hooks:
        NewtonManager._post_replicate_hooks.append(post_replicate_deformable_hook)
    NewtonManager._post_finalize_model_fn = apply_model_cfg


def register_hooks() -> None:
    """Register all deformable hooks with :class:`NewtonManager`.

    This is called automatically on first access to any deferred symbol
    and can be called again after :meth:`NewtonManager.clear` to
    re-register hooks that were wiped.
    """
    global _hooks_registered
    # Force re-registration so hooks survive NewtonManager.clear()
    _hooks_registered = False
    _do_deferred_imports()


def __getattr__(name: str):
    if name in _DEFERRED_NAMES:
        _do_deferred_imports()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
