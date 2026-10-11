# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton's VBD on a tetrahedral mesh of each berry: damped elastic tissue, gripped by friction.

The alternative to the MPM tissue solvers. Each berry is a tetrahedral mesh from its asset
(:func:`~..assets.berry_asset.load_berry_tetmesh`), simulated by Newton's stock :class:`~newton.solvers.SolverVBD`.
The tissue is elastic, without damage or tearing: a squeezed berry recovers its shape. The berries collide with each
other through VBD's surface contacts, filtered to keep a berry from colliding with itself.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from isaaclab_newton.physics import NewtonVBDManager, VBDSolverCfg
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg
from newton import Model
from newton.solvers import SolverVBD
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from isaaclab.utils import configclass

YOUNG_MODULUS = 24000.0
"""Young's modulus [Pa] of the tetrahedral tissue: stiffer than the MPM tissue's, which also resists by plasticity."""
POISSON_RATIO = 0.4
"""Poisson's ratio of the tetrahedral tissue."""
DAMPING = 200.0
"""Strain-rate damping of the tetrahedral tissue, which dissipates its elastic recoil."""


def tissue_material(profile: dict) -> NewtonDeformableBodyMaterialCfg:
    """Return the tetrahedral material of a berry from its asset profile: its density, and the VBD stiffness.

    Args:
        profile: The berry asset's ``/Berry/TaskData/Profile`` metadata.
    """
    return NewtonDeformableBodyMaterialCfg(
        density=float(profile["simulation"]["density"]),
        k_mu=YOUNG_MODULUS / (2 * (1 + POISSON_RATIO)),
        k_lambda=YOUNG_MODULUS * POISSON_RATIO / ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO)),
        k_damp=DAMPING,
        particle_radius=0.0005,
    )


def inter_berry_contact_filters(model: Model) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Exclude each berry's own surface from its contact candidates, keeping the contacts between berries.

    VBD detects self-contact and contact between separate meshes alike. The tetrahedra's connectivity tells the
    berries apart, so that a berry's nodes never collide with its own surface.
    """
    tets = model.tet_indices.numpy()
    rows = np.repeat(tets[:, 0], 3)
    graph = coo_matrix((np.ones(len(rows)), (rows, tets[:, 1:].reshape(-1))), shape=(model.particle_count,) * 2)
    count, components = connected_components(graph, directed=False)
    triangle_components = components[model.tri_indices.numpy()[:, 0]]
    # Newton's edge rows are (opposite 0, opposite 1, end 0, end 1).
    edge_components = components[model.edge_indices.numpy()[:, 2]]
    vertex_filters, edge_filters = {}, {}
    for component in range(count):
        own_triangles = set(np.flatnonzero(triangle_components == component).tolist())
        own_edges = set(np.flatnonzero(edge_components == component).tolist())
        vertex_filters.update({int(vertex): own_triangles for vertex in np.flatnonzero(components == component)})
        edge_filters.update({edge: own_edges for edge in own_edges})
    return vertex_filters, edge_filters


class GraspVBDManager(NewtonVBDManager):
    """:class:`~isaaclab_newton.physics.NewtonVBDManager` whose solver keeps only the contacts between berries."""

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: GraspVBDSolverCfg) -> SolverVBD:
        # The coupler passes this entry's model view: the damping applies to the berries' contacts only.
        model.shape_material_kd = wp.full(model.shape_count, solver_cfg.contact_damping, device=model.device)
        vertex_filters, edge_filters = inter_berry_contact_filters(model)
        return SolverVBD(
            model,
            **cls._filter_solver_kwargs(SolverVBD, solver_cfg),
            particle_external_vertex_contact_filtering_map=vertex_filters,
            particle_external_edge_contact_filtering_map=edge_filters,
        )


@configclass
class GraspVBDSolverCfg(VBDSolverCfg):
    """Configuration of the berries' VBD solver, with defaults tuned for berry tissue."""

    class_type: type = GraspVBDManager

    iterations: int = 20
    """VBD iterations per substep; fewer leave the damped tissue visibly softer."""
    friction_epsilon: float = 1.0e-4
    """Relative velocity [m/s] below which friction is smoothed; small, so that a held berry does not creep."""
    rigid_compliant_alm: bool | None = True
    rigid_body_particle_contact_buffer_size: int = 2048
    contact_damping: float = 0.5
    """Damping [N·s/m] of the shapes' contacts with the berries, sized for a berry of a few grams. Newton's default,
    sized for heavier objects, stalls the closing fingers and lets the berry slip out of them."""
    # Contacts between berries only: detected every iteration, with the berries' own surfaces filtered out instead of
    # topologically.
    particle_enable_self_contact: bool = True
    particle_self_contact_radius: float | None = None
    particle_self_contact_margin: float = 0.001
    particle_self_contact_gap: float = 0.001
    """Contact thickness [m] between berries."""
    particle_collision_detection_interval: int = 0
    particle_topological_contact_filter_threshold: int = 0
