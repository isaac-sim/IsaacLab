# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rigid contact profiles and material coupling for the coffee task."""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import TYPE_CHECKING

import newton
import numpy as np
import warp as wp
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonMPMManager,
)
from isaaclab_newton.physics.mpm_manager import _make_solver_config
from newton import StateFlags
from newton.solvers import SolverImplicitMPM
from newton.solvers.experimental.coupled import SolverCoupled
from warp.fem import TemporaryStore

from isaaclab.utils import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
from isaaclab_contrib.coupling.coupler import NewtonCouplerManager

from .assets import SPOON_BOWL_LENGTH, SPOON_BOWL_WIDTH

if TYPE_CHECKING:
    from newton import Model, State

    from .dissolution import CoffeeDissolution


@configclass
class CoffeeMPMSolverCfg(MPMSolverCfg):
    """Native MPM with optional passive rendering grains, disabled for training."""

    class_type: str = "isaaclab_tasks.contrib.rizon_sharpa_coffee.physics:CoffeeMPMManager"
    grains_per_particle: int = 0
    """Number of visual samples per physical particle; zero disables their allocation."""


class CoffeeMPMManager(NewtonMPMManager):
    """Construct the task's solver through the existing coupled-solver factory."""

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: CoffeeMPMSolverCfg) -> SolverImplicitMPM:
        if type(solver_cfg.grains_per_particle) is not int or solver_cfg.grains_per_particle < 0:
            raise ValueError("grains_per_particle must be a nonnegative integer.")
        return CoffeeMPMSolver(
            model,
            _make_solver_config(solver_cfg),
            grains_per_particle=solver_cfg.grains_per_particle,
        )


class CoffeeMPMSolver(SolverImplicitMPM):
    """Advect Newton's native rendering grains at the actual solver substep rate."""

    def __init__(self, model: Model, config: SolverImplicitMPM.Config, *, grains_per_particle: int):
        self.grains = None
        self.dissolution: CoffeeDissolution | None = None
        super().__init__(model, config=config, temporary_store=TemporaryStore())
        self._grain_previous = None
        if grains_per_particle:
            initial_state = model.state()
            self.grains = self.sample_render_grains(initial_state, grains_per_particle)
            # The native rendering routines read only these two previous-state fields.
            self._grain_previous = SimpleNamespace(
                particle_q=wp.clone(initial_state.particle_q),
                mpm=SimpleNamespace(particle_transform=wp.clone(initial_state.mpm.particle_transform)),
            )

    def step(self, state_in, state_out, control, contacts, dt):
        if self.dissolution is not None:
            self.dissolution.step(state_in, dt)
        if self.grains is not None:
            wp.copy(self._grain_previous.particle_q, state_in.particle_q)
            wp.copy(self._grain_previous.mpm.particle_transform, state_in.mpm.particle_transform)
        super().step(state_in, state_out, control, contacts, dt)
        if self.grains is not None:
            self.update_particle_frames(self._grain_previous, state_out, dt)
            self.update_render_grains(self._grain_previous, state_out, self.grains, dt)

    def reset(self, state: State, world_mask=None, flags=None):
        if self.dissolution is not None and (flags is None or flags & (StateFlags.PARTICLE_Q | StateFlags.PARTICLE_QD)):
            self.dissolution.reset(world_mask)
        super().reset(state, world_mask=world_mask, flags=flags)
        if self.grains is None or (flags is not None and not flags & (StateFlags.PARTICLE_Q | StateFlags.PARTICLE_QD)):
            return
        samples = self.sample_render_grains(state, self.grains.shape[1])
        if world_mask is None:
            wp.copy(self.grains, samples)
        else:
            wp.launch(
                _reset_grains,
                dim=self.grains.shape,
                inputs=[samples, self.grains, self.model.particle_world, world_mask],
                device=self.model.device,
            )


class NewtonCoffeeCouplerManager(NewtonCouplerManager):
    """Use independent stepping for a mount, or native proxy coupling for a grasp.

    The task explicitly updates its kinematic spoon from the measured palm.
    Newton's proxy solver steps only connected proxy groups, so an empty proxy
    graph does not advance either solver. The base coupled solver provides the
    independent stepping required by this mounted-tool approximation.
    """

    @classmethod
    def _build_proxy_coupled_solver(cls, model, entries, proxy_cfgs, solver_cfg) -> SolverCoupled:
        if proxy_cfgs:
            return super()._build_proxy_coupled_solver(model, entries, proxy_cfgs, solver_cfg)
        return SolverCoupled(model=model, entries=entries)


def configure_spoon_rim_margin(solver: CoffeeMPMSolver, scale: float) -> None:
    """Taper native MPM face margins toward the spoon rim before graph capture.

    The original closed shell remains intact. Its outermost 15% of elliptical
    radius blends to ``scale`` times the broad-face margin in a few material
    bands. The floor and handle retain their original margins. Native contact
    queries use the closest triangle's material, without extra per-step kernels.
    """
    if scale == 1.0:
        return
    model, mpm = solver.model, solver._mpm_model
    collider = mpm.collider
    flags = model.shape_flags.numpy()
    face_material = collider.face_material_index.numpy()
    offsets = collider.collider_face_offset.numpy()
    properties = {
        name: getattr(collider, name).numpy().tolist()
        for name in ("material_thickness", "material_friction", "material_adhesion", "material_projection_threshold")
    }
    for collider_id, body_id in enumerate(collider.collider_body_index.numpy()):
        if body_id < 0 or "/Spoon" not in str(model.body_label[body_id]):
            continue
        mesh = mpm._collider_meshes[collider_id]
        faces = mesh.indices.numpy().reshape(-1, 3)
        centers = mesh.points.numpy()[faces].mean(axis=1)
        start = offsets[collider_id]
        materials = face_material[start : start + len(faces)]
        # Native body colliders concatenate shapes in body_shapes order and
        # assign increasing material IDs to their face groups.
        shape_ids = [sid for sid in model.body_shapes[body_id] if flags[sid] & newton.ShapeFlags.COLLIDE_PARTICLES]
        for shape_id, material_id in zip(shape_ids, np.unique(materials), strict=True):
            if not str(model.shape_label[shape_id]).endswith("/BowlCollision"):
                continue
            radial = np.linalg.norm(centers[:, :2] / (SPOON_BOWL_LENGTH / 2.0, SPOON_BOWL_WIDTH / 2.0), axis=1)
            blend = np.clip((radial - 0.85) / 0.15, 0.0, 1.0)
            bands = np.rint(8.0 * blend * blend * (3.0 - 2.0 * blend)).astype(np.int32)
            bowl_faces = materials == material_id
            for band in np.unique(bands[bowl_faces]):
                if band == 0:
                    continue
                new_material = len(properties["material_thickness"])
                for name, values in properties.items():
                    value = values[material_id]
                    if name == "material_thickness":
                        value *= 1.0 - (1.0 - scale) * float(band) / 8.0
                    values.append(value)
                materials[bowl_faces & (bands == band)] = new_material
    collider.face_material_index.assign(face_material)
    for name, values in properties.items():
        setattr(collider, name, wp.array(values, dtype=float, device=model.device))


@configclass
class CoffeeCouplerCfg(CouplerProxyCfg):
    """Task-local MJWarp/MPM coupling with mounted and contact-grasp modes."""

    class_type: type | str = "{DIR}.physics:NewtonCoffeeCouplerManager"


def make_grasp_solver_cfg() -> MJWarpSolverCfg:
    """Rigid solver profile shared by the free-grasp screen and coffee scene."""
    return MJWarpSolverCfg(
        use_mujoco_contacts=True,
        integrator="implicitfast",
        cone="elliptic",
        impratio=20.0,
        enable_multiccd=True,
        iterations=100,
        ls_iterations=30,
        njmax=512,
        nconmax=128,
    )


def configure_grasp_contacts(builder: newton.ModelBuilder, shape_ids: Iterable[int]) -> None:
    """Resolve the same small-tool contact profile in both scene builders.

    The default 100 mm search gap is inappropriate for a millimetre-scale
    pinch. Six-dimensional friction represents a finite fingertip contact
    patch; these coefficients are simulation parameters, not hardware calibration.
    """
    for shape_id in shape_ids:
        if not builder.shape_flags[shape_id] & newton.ShapeFlags.COLLIDE_SHAPES:
            continue
        builder.shape_gap[shape_id] = 0.0005
        builder.shape_material_mu_torsional[shape_id] = 0.005
        builder.shape_material_mu_rolling[shape_id] = 0.005
        for name, value in (
            ("condim", 6),
            ("solref_mode", 1),  # Newton's raw MuJoCo solref interpretation.
            ("solref", (0.002, 1.0)),
            ("geom_solimp", (0.95, 0.99, 0.001, 0.5, 2.0)),
        ):
            builder.custom_attributes[f"mujoco:{name}"].values[shape_id] = value


def configure_contact_grasp(physics: NewtonCfg, *, rigid_substeps: int = 2, mass_scale: float = 20.0) -> None:
    """Give the free spoon to MJWarp and return MPM reaction impulses to it.

    Args:
        physics: Task coupling configuration, modified before model creation.
        rigid_substeps: Rigid integration steps per coupling tick.
        mass_scale: Finite MPM proxy inertia multiplier; the rigid mass is unchanged.
    """
    coupler = physics.solver_cfg
    rigid, media = coupler.entries
    rigid.solver_cfg = make_grasp_solver_cfg()
    rigid.bodies = [r"/World/envs/env_.*/Robot", r"/World/envs/env_.*/Spoon"]
    rigid.substeps = rigid_substeps
    media.bodies = []
    coupler.proxies = [
        CouplerProxyMappingCfg(
            source="robot",
            destination="media",
            bodies=[r"/World/envs/env_.*/Spoon"],
            # MPM's free-spoon inertia omits support from the finger contacts.
            # A finite virtual inertia prevents the particle solve from yielding
            # through the held tool. The source spoon remains 25 g and receives
            # the reaction impulses; opening the hand still releases it.
            mass_scale=mass_scale,
            mode="staggered",
            collision_pipeline=None,
        )
    ]


def make_physics_cfg(voxel_size: float, *, water: bool = False) -> NewtonCfg:
    """Configure rigid and material stepping with a liquid pressure budget."""
    return NewtonCfg(
        solver_cfg=CoffeeCouplerCfg(
            entries=[
                CouplerEntryCfg(
                    name="robot",
                    solver_cfg=MJWarpSolverCfg(
                        use_mujoco_contacts=False, integrator="implicitfast", njmax=1024, nconmax=128
                    ),
                    bodies=[r"/World/envs/env_.*/Robot"],
                ),
                CouplerEntryCfg(
                    name="media",
                    solver_cfg=CoffeeMPMSolverCfg(
                        voxel_size=voxel_size,
                        # configure_scene reserves a particle-count-based budget
                        # unless the caller supplies a positive capacity.
                        max_active_cell_count=-1,
                        max_lower_node_count=256,
                        max_upper_node_count=32,
                        solver="gs" if water else "auto",
                        # Tall liquid fills need a converged pressure solve;
                        # conserved particle mass alone does not preserve level.
                        max_iterations=500 if water else 100,
                        # Resolve contacts at particle sites and pressure within
                        # each cell. Hard projection can permanently collapse
                        # powder onto the concave tool's surface.
                        collider_basis="pic27",
                        strain_basis="P1d",
                        collider_velocity_mode="backward",
                        # Native background-air regularization bounds nearly empty grid nodes.
                        air_drag=1.0,
                        separate_worlds=True,
                    ),
                    bodies=[r"/World/envs/env_.*/Spoon"],
                    all_particles=True,
                    include_static_shapes=True,
                    in_place=True,
                ),
            ],
            proxies=[],
        ),
        collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
        num_substeps=1,
        use_cuda_graph=True,
    )


@wp.kernel
def _reset_grains(
    samples: wp.array2d[wp.vec3],
    grains: wp.array2d[wp.vec3],
    worlds: wp.array[int],
    mask: wp.array[wp.bool],
):
    i, j = wp.tid()
    world = worlds[i]
    if world < 0:
        world = mask.shape[0] - 1
    if mask[world]:
        grains[i, j] = samples[i, j]
