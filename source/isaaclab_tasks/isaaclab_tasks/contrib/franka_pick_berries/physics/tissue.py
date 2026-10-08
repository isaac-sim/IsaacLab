# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Berry tissue as Newton MPM particles: loading, placement in the punnet, scene configuration and a runtime handle.

The tissue solvers (:mod:`.grasp_explicit_mpm`, :mod:`.grasp_implicit_mpm`) model the tissue's deformation and
damage; this module only describes and places the particles.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.spawners.mpm import MPMPointsCfg
from scipy.spatial.transform import Rotation

from ..assets.berry_asset import load_berry_asset, rotate_berry_sh
from ..scene.tableware import PUNNET
from .materials import particle_material

if TYPE_CHECKING:
    from isaaclab_newton.assets import MPMObject
    from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg

    from ..pick_berries_env_cfg import BerryPickEnvCfg

PARTICLES_PER_CELL = 8
"""Particles per background-grid cell. Below about this density the material is too sparse to carry load."""


@dataclass
class TissueSpec:
    """One berry's asset, tissue particles and world placement, resolved from the task configuration."""

    name: str
    """Instance name, numbered when there are several berries (``raspberry_1``)."""
    stage: object
    """The berry's USD asset, opened."""
    gaussians: dict
    """Gaussians at rest: ``xyz``, ``scales``, ``rotations``, ``alpha``, ``sh`` and ``regions`` arrays."""
    particles: dict
    """Tissue particles at rest: ``xyz`` [m], ``regions``, ``interface``, ``spacing`` [m] and ``particle_volume``
    [m³]."""
    profile: dict
    """The asset's simulation metadata."""
    material: MPMParticleMaterialCfg
    offset: tuple[float, float, float]
    """World position [m] of the berry's frame."""

    @property
    def scene_name(self) -> str:
        """Name of the scene asset holding this berry's particles."""
        return f"berry_{self.name}"

    @property
    def particle_mass(self) -> float:
        """Mass of one tissue particle [kg]."""
        return float(self.material.density * self.particles["particle_volume"])

    @property
    def particle_radius(self) -> float:
        """Radius [m] of a sphere with the volume one particle represents."""
        return float((3.0 * self.particles["particle_volume"] / (4.0 * np.pi)) ** (1.0 / 3.0))

    @property
    def voxel_size(self) -> float:
        """MPM grid spacing [m] giving :data:`PARTICLES_PER_CELL` particles per cell."""
        return float(PARTICLES_PER_CELL ** (1.0 / 3.0) * self.particles["spacing"])


def load_tissue(cfg: BerryPickEnvCfg, name: str, offset: tuple[float, float, float]) -> TissueSpec:
    """Load one berry's asset, select its tissue particles and give it its material."""
    stage, gaussians, particles, profile = load_berry_asset(cfg.berry_asset)
    if cfg.tissue_resolution == "half":
        particles = _half_resolution(particles)
    elif cfg.tissue_resolution != "full":
        raise ValueError(f"Unknown tissue resolution: {cfg.tissue_resolution}")
    return TissueSpec(name, stage, gaussians, particles, profile, particle_material(profile), offset)


def _half_resolution(particles: dict) -> dict:
    """Keep alternating sites of the tissue's cubic lattice, with twice the volume each: the mass is preserved.

    The Gaussians are untouched; they bind to the remaining particles. The explicit solver keeps its grid and rate,
    which come from the material; the implicit solver sizes its grid from the particle spacing, so it coarsens too.
    """
    xyz = np.asarray(particles["xyz"])
    lattice = (xyz - xyz.min(0)) / float(particles["spacing"])
    cells = np.rint(lattice).astype(np.int64)
    if not np.allclose(lattice, cells, atol=1e-3, rtol=0):
        raise ValueError("Half resolution requires regular-lattice tissue; use full resolution")
    if len(np.unique(particles["regions"])) != 1:
        raise ValueError("Half resolution requires a single tissue material region")
    keep = cells.sum(1) % 2 == 0
    if keep.sum() < 32 or keep.all():
        raise ValueError("Not enough spatially distributed particles for half resolution")
    volume = float(particles["particle_volume"]) * len(xyz) / int(keep.sum())
    return {
        "xyz": xyz[keep].copy(),
        "regions": np.asarray(particles["regions"])[keep].copy(),
        "interface": np.asarray(particles["interface"])[keep].copy(),
        "spacing": np.float32(np.cbrt(volume)),
        "particle_volume": np.float64(volume),
    }


# Offsets [m] of the fixed layout about the punnet center, by number of berries.
_FIXED_LAYOUT = {
    1: ((0.0, 0.0),),
    2: ((-0.022, 0.0), (0.022, 0.0)),
    3: ((0.0, -0.018), (-0.024, 0.018), (0.024, 0.018)),
}


def fixed_layout(cfg: BerryPickEnvCfg) -> list[tuple[str, tuple[float, float, float]]]:
    """Return each berry's instance name and world offset [m] in the fixed layout."""
    if cfg.num_berries not in _FIXED_LAYOUT:
        raise ValueError(f"num_berries must be 1, 2 or 3, not {cfg.num_berries}")
    x, y, z = cfg.berry_position
    if cfg.num_berries == 1:
        return [("raspberry", (x, y, z))]
    return [(f"raspberry_{i}", (x + dx, y + dy, z)) for i, (dx, dy) in enumerate(_FIXED_LAYOUT[cfg.num_berries], 1)]


def random_punnet_poses(particles: dict, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample separated centers and full 3D orientations for ``count`` copies of a tissue inside the punnet.

    Returns each copy's rotation matrix and its translation [m] in the berry frame, about the punnet center, that
    keeps the rotated tissue's bottom at the height of the unrotated tissue's, on the punnet floor. Extra rim and
    berry clearance leaves room for a vertical gripper approach.
    """
    rng = np.random.default_rng(seed)
    rotations = Rotation.random(count, random_state=rng).as_matrix()
    center = (particles["xyz"].min(0) + particles["xyz"].max(0)) / 2
    rotated = [(particles["xyz"] - center) @ r.T for r in rotations]
    radius = max(float(np.linalg.norm(p[:, :2], axis=1).max()) for p in rotated)
    bounds = np.array(PUNNET[2:4]) - radius - 0.016
    if np.any(bounds <= 0):
        raise ValueError("The berries must fit inside the punnet with gripper clearance")
    for _ in range(10000):
        centers = rng.uniform(-bounds, bounds, size=(count, 2))
        if all(np.linalg.norm(centers[i] - centers[j]) >= 2 * radius + 0.014 for i in range(count) for j in range(i)):
            break
    else:
        raise ValueError("Unable to fit separated berries with gripper clearance in the punnet")
    centers = np.asarray(sorted(centers, key=lambda c: c[1]))
    floor = float(particles["xyz"][:, 2].min())
    shifts = [np.array([*xy, floor - p[:, 2].min()]) - r @ center for r, p, xy in zip(rotations, rotated, centers)]
    return rotations, np.asarray(shifts, np.float32)


def transform_tissue(spec: TissueSpec, rotation: np.ndarray, shift: np.ndarray, offset) -> TissueSpec:
    """Return the berry rotated by ``rotation`` and moved by ``shift`` [m] in its frame, at world ``offset`` [m].

    Tissue and Gaussians move together; the Gaussians' orientations and spherical harmonics turn with them.
    """
    particles = dict(spec.particles, xyz=(spec.particles["xyz"] @ rotation.T + shift).astype(np.float32))
    gaussians = dict(spec.gaussians, xyz=(spec.gaussians["xyz"] @ rotation.T + shift).astype(np.float32))
    turn = Rotation.from_matrix(rotation)
    gaussians["rotations"] = (turn * Rotation.from_quat(spec.gaussians["rotations"])).as_quat().astype(np.float32)
    gaussians["sh"] = rotate_berry_sh(spec.gaussians["sh"], rotation).astype(np.float32)
    return replace(spec, particles=particles, gaussians=gaussians, offset=tuple(offset))


def tissue_object_cfg(spec: TissueSpec) -> MPMObjectCfg:
    """Return the scene asset that spawns a berry's tissue particles at its world offset."""
    return MPMObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{spec.scene_name}",
        spawn=MPMPointsCfg(
            positions=spec.particles["xyz"].astype(np.float32).tolist(),
            mass=spec.particle_mass,
            radius=spec.particle_radius,
            material=spec.material,
            visual_color=(0.8, 0.1, 0.15),
        ),
        init_state=MPMObjectCfg.InitialStateCfg(pos=spec.offset, rot=(0.0, 0.0, 0.0, 1.0)),
    )


@wp.kernel
def _to_local(world: wp.array[wp.vec3], offset: wp.vec3, local: wp.array[wp.vec3]):
    i = wp.tid()
    local[i] = world[i] - offset


class BerryTissue:
    """Runtime handle over one berry's MPM particles, in the berry's own frame (world minus its offset).

    Args:
        spec: The berry's description.
        mpm_object: The scene asset holding its particles.
    """

    def __init__(self, spec: TissueSpec, mpm_object: MPMObject):
        self.name = spec.name
        self.stage = spec.stage
        self.gaussians = spec.gaussians
        self.particles = spec.particles
        self.profile = spec.profile
        self.mpm_object = mpm_object
        self.device = mpm_object.device
        self.offset = np.asarray(spec.offset, np.float32)
        self.rest = np.asarray(spec.particles["xyz"], np.float32)
        """Particle rest positions in the berry frame [m]."""
        self.particle_start = int(mpm_object._particle_offsets.numpy()[0])
        """Index of the berry's first particle in the solver."""
        self.damage = None
        """Damage of each particle, from 0 (intact) to 1, on the device; set by :meth:`bind_damage`."""
        self._offset = wp.vec3(*self.offset)
        self._local = wp.zeros(len(self.rest), dtype=wp.vec3, device=self.device)

    def positions(self) -> np.ndarray:
        """Particle positions in the berry frame [m], on the host."""
        return self.mpm_object.data.particle_pos_w.torch[0].cpu().numpy() - self.offset

    def positions_warp(self) -> wp.array:
        """Particle positions in the berry frame [m] as a device array, refreshed on every call."""
        world = self.mpm_object.data.particle_pos_w.warp[0]
        wp.launch(_to_local, dim=len(self.rest), inputs=[world, self._offset, self._local], device=self.device)
        return self._local

    def bind_damage(self, solver) -> None:
        """Read the berry's damage from the tissue solver that models it."""
        self.damage = solver.damage[self.particle_start : self.particle_start + len(self.rest)]
