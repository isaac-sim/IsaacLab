# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mass-preserving conversion of wetted coffee into Newton MPM liquid."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from newton._src.solvers.implicit_mpm.implicit_mpm_model import MaterialParameters
from warp.utils import array_sum

if TYPE_CHECKING:
    from newton import State

    from .physics import SolverCoffeeMPM

_MATERIAL_FIELDS = (
    "young_modulus",
    "poisson_ratio",
    "damping",
    "friction",
    "yield_pressure",
    "tensile_yield_ratio",
    "yield_stress",
    "viscosity",
    "hardening",
    "hardening_rate",
    "softening_rate",
    "dilatancy",
)


class CoffeeDissolution:
    """Convert coffee after accumulated contact with water or the liquid mixture.

    The solver owns this state independently of rendering. Conversion copies the
    contacted water's constitutive parameters, relaxes granular stress, and sets
    reference volume to coffee mass / water density. Positions, velocities,
    masses, particle count and original coffee IDs remain unchanged. The fixed
    liquid mask includes water and converted coffee for surface reconstruction.

    This is an isothermal soluble-coffee approximation, not extraction chemistry.
    Contact searches and material writes run on the GPU inside physics capture.

    Args:
        solver: Task MPM solver, configured before its first captured step.
        coffee_ids: Global model indices of coffee particles.
        water_ids: Global model indices of the original solvent particles.
        wetting_time: Accumulated water contact required for conversion [s].
        contact_radius: Water neighbourhood radius [m].
    """

    def __init__(
        self,
        solver: SolverCoffeeMPM,
        coffee_ids: wp.array,
        water_ids: wp.array,
        *,
        wetting_time: float,
        contact_radius: float,
    ):
        if not math.isfinite(wetting_time) or wetting_time < 0.0:
            raise ValueError("wetting_time must be finite and nonnegative.")
        if not math.isfinite(contact_radius) or contact_radius <= 0.0:
            raise ValueError("contact_radius must be finite and positive.")
        if not coffee_ids.size or not water_ids.size:
            raise ValueError("Coffee dissolution requires both coffee and water particles.")
        self._model = solver.model
        self._mpm = solver._mpm_model
        self.coffee_ids, self.water_ids = coffee_ids, water_ids
        self.wetting_time, self.contact_radius = wetting_time, contact_radius
        self.liquid_ids = wp.array(
            np.concatenate((water_ids.numpy(), coffee_ids.numpy())), dtype=int, device=self._model.device
        )
        self.liquid_flags = wp.zeros(self.liquid_ids.size, dtype=int, device=self._model.device)
        self.liquid_flags[: water_ids.size].fill_(1)  # Newton ParticleFlags.ACTIVE.
        self.wet_time = wp.zeros(coffee_ids.size, dtype=float, device=self._model.device)
        # Warp captures one mutable host descriptor per grid dtype.
        # Rendering rebuilds float grids outside physics capture, overwriting
        # that descriptor. A double grid isolates the captured wetting queries;
        # particle state and MPM arithmetic retain their native precision.
        self._liquid_points = wp.empty(self.liquid_ids.size, dtype=wp.vec3d, device=self._model.device)
        self._liquid_groups = wp.empty(self.liquid_ids.size, dtype=int, device=self._model.device)
        self._grid = wp.HashGrid(64, 64, 64, device=self._model.device, dtype=wp.float64)
        self._grid.reserve(self.liquid_ids.size, with_groups=True)
        self._dry_material = MaterialParameters()
        for name in _MATERIAL_FIELDS:
            source = getattr(self._mpm.material_parameters, name)
            setattr(self._dry_material, name, wp.clone(wp.indexedarray(source, coffee_ids)))
        self._dry_radius = wp.clone(wp.indexedarray(self._model.particle_radius, coffee_ids))
        self._dry_volume = wp.clone(wp.indexedarray(self._mpm.particle_volume, coffee_ids))
        self._dry_density = wp.clone(wp.indexedarray(self._mpm.particle_density, coffee_ids))
        self._converted_masses = wp.empty(coffee_ids.size, dtype=wp.float64, device=self._model.device)
        self._converted_mass = wp.empty(1, dtype=wp.float64, device=self._model.device)

    @property
    def converted_mass_kg(self) -> float:
        """Mass of coffee-origin liquid [kg]; reading copies one scalar to the host."""
        with wp.ScopedDevice(self._model.device):
            wp.launch(
                _gather_converted_mass,
                dim=self.coffee_ids.size,
                inputs=[self.coffee_ids, self.water_ids.size, self.liquid_flags, self._model.particle_mass],
                outputs=[self._converted_masses],
                device=self._model.device,
            )
            array_sum(self._converted_masses, out=self._converted_mass)
        return float(self._converted_mass.numpy()[0])

    def step(self, state: State, dt: float) -> None:
        """Update wetting and convert individual particles before the MPM solve."""
        wp.launch(
            _gather_liquid,
            dim=self.liquid_ids.size,
            inputs=[self.liquid_ids, state.particle_q, self._model.particle_world, self.liquid_flags],
            outputs=[self._liquid_points, self._liquid_groups],
            device=self._model.device,
        )
        # Group membership is a snapshot before conversion. Newly liquid grains
        # can wet neighbours next step, without read/write races in this kernel.
        # Dry particles use an unused group and never enter liquid queries.
        self._grid.build(self._liquid_points, self.contact_radius, groups=self._liquid_groups)
        wp.launch(
            _convert_wet_coffee,
            dim=self.coffee_ids.size,
            inputs=[
                self.coffee_ids,
                self.liquid_ids,
                self.water_ids.size,
                self._liquid_points,
                self._grid.id,
                self.contact_radius,
                self.wetting_time,
                dt,
                self._model.particle_world,
                state.particle_q,
                self._model.particle_mass,
                self._mpm.material_parameters,
                self._model.particle_radius,
                self._mpm.particle_volume,
                self._mpm.particle_density,
                self._mpm.material_particle_volume,
                state.mpm.particle_elastic_strain,
                state.mpm.particle_stress,
                state.mpm.particle_Jp,
                state.mpm.particle_transform,
                self.wet_time,
                self.liquid_flags,
            ],
            device=self._model.device,
        )

    def reset(self, world_mask: wp.array | None = None) -> None:
        """Restore selected worlds' dry material and liquid membership in place."""
        wp.launch(
            _reset_dissolution,
            dim=self.coffee_ids.size,
            inputs=[
                self.coffee_ids,
                self.water_ids.size,
                self._model.particle_world,
                world_mask,
                self._dry_material,
                self._dry_radius,
                self._dry_volume,
                self._dry_density,
                self._mpm.material_parameters,
                self._model.particle_radius,
                self._mpm.particle_volume,
                self._mpm.particle_density,
                self._mpm.material_particle_volume,
                self.wet_time,
                self.liquid_flags,
            ],
            device=self._model.device,
        )


@wp.kernel
def _gather_converted_mass(
    coffee_ids: wp.array[int],
    water_count: int,
    liquid_flags: wp.array[int],
    particle_mass: wp.array[float],
    converted_mass: wp.array[wp.float64],
):
    i = wp.tid()
    converted_mass[i] = wp.float64(0.0)
    if liquid_flags[water_count + i] != 0:
        converted_mass[i] = wp.float64(particle_mass[coffee_ids[i]])


@wp.func
def _copy_material(source: MaterialParameters, i: int, target: MaterialParameters, j: int):
    target.young_modulus[j] = source.young_modulus[i]
    target.poisson_ratio[j] = source.poisson_ratio[i]
    target.damping[j] = source.damping[i]
    target.friction[j] = source.friction[i]
    target.yield_pressure[j] = source.yield_pressure[i]
    target.tensile_yield_ratio[j] = source.tensile_yield_ratio[i]
    target.yield_stress[j] = source.yield_stress[i]
    target.viscosity[j] = source.viscosity[i]
    target.hardening[j] = source.hardening[i]
    target.hardening_rate[j] = source.hardening_rate[i]
    target.softening_rate[j] = source.softening_rate[i]
    target.dilatancy[j] = source.dilatancy[i]


@wp.kernel
def _gather_liquid(
    ids: wp.array[int],
    positions: wp.array[wp.vec3],
    worlds: wp.array[int],
    flags: wp.array[int],
    result: wp.array[wp.vec3d],
    groups: wp.array[int],
):
    i = wp.tid()
    result[i] = wp.vec3d(positions[ids[i]])
    groups[i] = -2
    if flags[i] != 0:
        # Keep world -1 (shared particles) distinct from the inactive group.
        groups[i] = worlds[ids[i]] + 1


@wp.kernel
def _convert_wet_coffee(
    coffee_ids: wp.array[int],
    liquid_ids: wp.array[int],
    water_count: int,
    water: wp.array[wp.vec3d],
    grid: wp.uint64,
    radius: float,
    wetting_time: float,
    dt: float,
    worlds: wp.array[int],
    positions: wp.array[wp.vec3],
    mass: wp.array[float],
    material: MaterialParameters,
    particle_radius: wp.array[float],
    volume: wp.array[float],
    density: wp.array[float],
    material_volume: wp.array[float],
    strain: wp.array[wp.mat33],
    stress: wp.array[wp.mat33],
    plastic_volume: wp.array[float],
    transform: wp.array[wp.mat33],
    wet_time: wp.array[float],
    liquid_flags: wp.array[int],
):
    i = wp.tid()
    if liquid_flags[water_count + i] != 0:
        return
    coffee_id = coffee_ids[i]
    closest = wp.float64(radius) * wp.float64(radius)
    solvent = int(-1)
    position = wp.vec3d(positions[coffee_id])
    for j in wp.hash_grid_query(grid, position, wp.float64(radius), worlds[coffee_id] + 1):
        distance = wp.length_sq(position - water[j])
        if distance < closest:
            closest = distance
            solvent = liquid_ids[j]
    if solvent < 0:
        return
    wet_time[i] += dt
    if wet_time[i] + 1.0e-7 < wetting_time:
        return
    _copy_material(material, solvent, material, coffee_id)
    # Material arrays are bound by reference. Updating the derived volumes here
    # avoids notify_model_changed(), which scans on the host and replaces masks.
    # Cached material extrema still cover both original coffee and original water.
    density[coffee_id] = density[solvent]
    volume[coffee_id] = mass[coffee_id] / density[solvent]
    material_volume[coffee_id] = volume[coffee_id]
    particle_radius[coffee_id] = 0.5 * wp.pow(volume[coffee_id], 1.0 / 3.0)
    strain[coffee_id] = wp.identity(n=3, dtype=float)
    stress[coffee_id] = wp.mat33(0.0)
    plastic_volume[coffee_id] = 1.0
    transform[coffee_id] = wp.identity(n=3, dtype=float)
    liquid_flags[water_count + i] = 1


@wp.kernel
def _reset_dissolution(
    ids: wp.array[int],
    water_count: int,
    worlds: wp.array[int],
    world_mask: wp.array[wp.bool],
    dry_material: MaterialParameters,
    dry_radius: wp.array[float],
    dry_volume: wp.array[float],
    dry_density: wp.array[float],
    material: MaterialParameters,
    radius: wp.array[float],
    volume: wp.array[float],
    density: wp.array[float],
    material_volume: wp.array[float],
    wet_time: wp.array[float],
    liquid_flags: wp.array[int],
):
    i = wp.tid()
    j = ids[i]
    world = worlds[j]
    if world_mask:
        if world < 0:
            world = world_mask.shape[0] - 1
        if not world_mask[world]:
            return
    _copy_material(dry_material, i, material, j)
    radius[j] = dry_radius[i]
    volume[j] = dry_volume[i]
    density[j] = dry_density[i]
    material_volume[j] = dry_volume[i]
    wet_time[i] = 0.0
    liquid_flags[water_count + i] = 0
