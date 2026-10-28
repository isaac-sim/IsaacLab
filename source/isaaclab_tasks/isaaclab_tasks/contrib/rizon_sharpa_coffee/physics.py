# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rigid contact profiles and material coupling for the coffee task."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import newton
import numpy as np
import warp as wp
from isaaclab_newton.physics import MPMSolver
from isaaclab_newton.physics.mpm_solver import mpm_solver_config
from newton import StateFlags
from newton.solvers import SolverImplicitMPM
from warp.fem import TemporaryStore

from .assets import SPOON_BOWL_LENGTH, SPOON_BOWL_WIDTH

if TYPE_CHECKING:
    from newton import Model, State

    from .demo_env_cfg import CoffeeMPMSolverCfg
    from .dissolution import CoffeeDissolution


class SolverCoffeeMPM(SolverImplicitMPM):
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


class CoffeeMPMSolver(MPMSolver):
    """Implicit MPM that builds :class:`SolverCoffeeMPM` with optional passive rendering grains."""

    solver_class = SolverCoffeeMPM

    @classmethod
    def build(
        cls, cfg: CoffeeMPMSolverCfg, model: Model, mode: wp.DeterministicMode, env_template: str | None = None
    ) -> SolverCoffeeMPM:
        """Construct :class:`SolverCoffeeMPM` from the shared MPM configuration resolution."""
        grains = cfg.grains_per_particle
        if type(grains) is not int or grains < 0:
            raise ValueError("grains_per_particle must be a nonnegative integer.")
        return cls.solver_class(model, mpm_solver_config(cfg), grains_per_particle=grains)


def configure_spoon_rim_margin(solver: SolverCoffeeMPM, scale: float) -> None:
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
