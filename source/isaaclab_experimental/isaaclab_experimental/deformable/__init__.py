# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental Newton deformable module.

Importing this module registers solver factories, particle sync, cloner hooks,
and model configuration hooks with :class:`NewtonManager`.
"""

from isaaclab_newton.physics import NewtonManager

from .cloner_hooks import per_world_deformable_hook, post_replicate_deformable_hook
from .coupled_solver import CoupledSolver
from .deformable_object import DeformableObject, DeformableRegistryEntry
from .deformable_object_data import DeformableObjectData
from .model_cfg_hook import apply_model_cfg
from .newton_manager_cfg import CoupledSolverCfg, NewtonModelCfg, VBDSolverCfg
from .particle_sync import sync_particles_to_usd
from .solver_factories import create_coupled_solver, create_vbd_solver

def register_hooks() -> None:
    """Register all deformable hooks with :class:`NewtonManager`.

    This is called automatically on first import and can be called again
    after :meth:`NewtonManager.clear` to re-register hooks that were wiped.
    """
    NewtonManager.register_solver_factory("vbd", create_vbd_solver)
    NewtonManager.register_solver_factory("coupled", create_coupled_solver)
    NewtonManager._particle_sync_fn = sync_particles_to_usd
    if per_world_deformable_hook not in NewtonManager._per_world_builder_hooks:
        NewtonManager._per_world_builder_hooks.append(per_world_deformable_hook)
    if post_replicate_deformable_hook not in NewtonManager._post_replicate_hooks:
        NewtonManager._post_replicate_hooks.append(post_replicate_deformable_hook)
    NewtonManager._post_finalize_model_fn = apply_model_cfg


# Perform initial registration
register_hooks()

# Register the Newton DeformableObject backend with the factory
from isaaclab.assets.deformable_object.deformable_object import DeformableObject as DeformableObjectFactory

DeformableObjectFactory.register("newton", DeformableObject)
