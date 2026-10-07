# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Berry tissue as Newton MPM particles: asset loading, layout, particle configuration and a runtime handle.

It is shared by the tissue solvers; each models the tissue's deformation and damage itself.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.spawners.mpm import MPMPointsCfg
from scipy.spatial.transform import Rotation

from ..assets.sh_rotation import rotate_sh
from ..assets.usd_asset import load_berry
from ..scene.tableware import BOWL, PUNNET
from .coupling import tissue_solver
from .materials import particle_material
from .resolution import physics_resolution

if TYPE_CHECKING:
    from isaaclab_newton.assets import MPMObject
    from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg

    from ..pick_berries_env_cfg import BerryPickEnvCfg

BERRY_NAMES = ("raspberry", "blackberry", "blueberry", "strawberry")

PARTICLES_PER_CELL = 8
"""Particles per background-grid cell. Below about this density the material is too sparse to carry load."""


@dataclass
class TissueSpec:
    """One berry's asset, tissue particles and world placement, resolved from the task configuration."""

    name: str
    """Instance name: the species, numbered when several berries share it (``raspberry_1``)."""
    species: str
    path: str
    stage: object
    asset: dict
    proxy: dict
    profile: dict
    resolution: dict
    material: MPMParticleMaterialCfg
    offset: tuple[float, float, float]

    @property
    def scene_name(self) -> str:
        """Name of the scene asset holding this berry's particles."""
        return f"berry_{self.name}"

    @property
    def particle_mass(self) -> float:
        """Mass of one tissue particle [kg]."""
        return float(self.material.density * self.proxy["particle_volume"])

    @property
    def particle_radius(self) -> float:
        """Radius [m] of a sphere with the volume one particle represents."""
        return float((3.0 * self.proxy["particle_volume"] / (4.0 * np.pi)) ** (1.0 / 3.0))

    @property
    def voxel_size(self) -> float:
        """MPM grid spacing [m] giving :data:`PARTICLES_PER_CELL` particles per cell."""
        return float(PARTICLES_PER_CELL ** (1.0 / 3.0) * self.proxy["spacing"])


# Offsets [m] about the berry position: four species at the punnet's corners, or two or three of one species.
_SPECIES_LAYOUT = ((-0.025, -0.025), (0.025, -0.025), (-0.025, 0.025), (0.025, 0.025))
_COUNT_LAYOUT = {2: ((-0.022, 0.0), (0.022, 0.0)), 3: ((0.0, -0.018), (-0.024, 0.018), (0.024, 0.018))}


def berry_layout(cfg: BerryPickEnvCfg) -> list[tuple[str, str, tuple[float, float, float]]]:
    """Return each berry's instance name, species and world offset [m].

    ``berry="all"`` separates the four species about the punnet; ``berry_count`` of two or three places that many
    berries of one species there.
    """
    x, y, z = cfg.berry_position
    if cfg.berry == "all":
        if cfg.berry_asset_path:
            raise ValueError("A custom --asset can only be used with a single berry")
        if cfg.berry_count != 1:
            raise ValueError("berry_count applies to a single species, not berry='all'")
        return [(name, name, (x + dx, y + dy, z)) for name, (dx, dy) in zip(BERRY_NAMES, _SPECIES_LAYOUT)]
    if cfg.berry_count == 1:
        return [(cfg.berry, cfg.berry, (x, y, z))]
    if cfg.berry_count not in _COUNT_LAYOUT:
        raise ValueError(f"berry_count must be 1, 2 or 3, not {cfg.berry_count}")
    return [
        (f"{cfg.berry}_{i}", cfg.berry, (x + dx, y + dy, z))
        for i, (dx, dy) in enumerate(_COUNT_LAYOUT[cfg.berry_count], start=1)
    ]


def random_punnet_poses(proxy: dict, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample separated centers and full 3D orientations for ``count`` copies of a tissue inside the punnet.

    Returns each copy's rotation matrix and its translation [m] in the berry frame, about the punnet center, that
    keeps the rotated tissue's bottom at the height of the unrotated tissue's, on the punnet floor. Extra rim and
    berry clearance leaves room for a vertical gripper approach.
    """
    rng = np.random.default_rng(seed)
    rotations = Rotation.random(count, random_state=rng).as_matrix()
    center = (proxy["xyz"].min(0) + proxy["xyz"].max(0)) / 2
    rotated = [(proxy["xyz"] - center) @ r.T for r in rotations]
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
    floor = float(proxy["xyz"][:, 2].min())
    shifts = [np.array([*xy, floor - p[:, 2].min()]) - r @ center for r, p, xy in zip(rotations, rotated, centers)]
    return rotations, np.asarray(shifts, np.float32)


def place_tissue(spec: TissueSpec, rotation: np.ndarray, shift: np.ndarray, offset) -> TissueSpec:
    """Return the berry rotated by ``rotation`` and moved by ``shift`` [m] in its frame, at world ``offset`` [m].

    Tissue and Gaussians move together; the Gaussians' orientations and spherical harmonics turn with them.
    """
    proxy = dict(spec.proxy, xyz=(spec.proxy["xyz"] @ rotation.T + shift).astype(np.float32))
    asset = dict(spec.asset, xyz=(spec.asset["xyz"] @ rotation.T + shift).astype(np.float32))
    turn = Rotation.from_matrix(rotation)
    asset["rotations"] = (turn * Rotation.from_quat(spec.asset["rotations"])).as_quat().astype(np.float32)
    asset["sh"] = rotate_sh(spec.asset["sh"], rotation).astype(np.float32)
    return replace(spec, proxy=proxy, asset=asset, offset=tuple(offset))


def load_tissue(cfg: BerryPickEnvCfg, name: str, species: str, offset: tuple[float, float, float]) -> TissueSpec:
    """Load one berry's asset and select its tissue particles and material."""
    if cfg.berry_asset_version not in ("v1", "v2", "v3"):
        raise ValueError(f"Unknown berry asset version: {cfg.berry_asset_version}")
    suffix = "" if cfg.berry_asset_version == "v1" else f"_{cfg.berry_asset_version}"
    path = cfg.berry_asset_path or f"{cfg.asset_root}/{species}/{species}{suffix}.usdz"
    stage, asset, proxy, profile = load_berry(path)
    if profile["berry"] != species:
        raise ValueError(f"Asset is for {profile['berry']}, but --berry selects {species}")
    material = particle_material(profile)
    parameters = {"density": material.density}
    if "particle_volume" in proxy:
        # v1 tissue has none; its particles then fill one lattice cell each.
        parameters["particle_volume"] = float(proxy["particle_volume"])
    proxy, _, resolution = physics_resolution(proxy, parameters, cfg.physics_resolution)
    proxy = dict(proxy, particle_volume=resolution["particle_volume_m3"])
    return TissueSpec(name, species, path, stage, asset, proxy, profile, resolution, material, offset)


def tissue_object_cfg(spec: TissueSpec) -> MPMObjectCfg:
    """Return the scene asset that spawns a berry's tissue particles at its world offset."""
    return MPMObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{spec.scene_name}",
        spawn=MPMPointsCfg(
            positions=spec.proxy["xyz"].astype(np.float32).tolist(),
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
    """Runtime handle over one berry's MPM particles, in the berry's own frame (world minus its offset)."""

    def __init__(self, spec: TissueSpec, particles: MPMObject):
        self.spec = spec
        self.particles = particles
        self.berry = spec.name
        self.usd_path, self.usd_stage = spec.path, spec.stage
        self.asset, self.proxy, self.profile, self.resolution = spec.asset, spec.proxy, spec.profile, spec.resolution
        self.offset = np.asarray(spec.offset, np.float32)
        self.rest = np.asarray(spec.proxy["xyz"], np.float32)
        device = particles.device
        self._offset_vec = wp.vec3(*self.offset)
        self._local = wp.zeros(len(self.rest), dtype=wp.vec3, device=device)
        self.particle_start = int(particles._particle_offsets.numpy()[0])
        self._solver = None
        # Damage, tearing, plastic strain history and bruise dose of each particle, bound from the tissue solver.
        self.damage = self.tear = self.history = self.dose = None

    def positions(self) -> np.ndarray:
        """Particle positions in the berry frame [m]."""
        return self.particles.data.particle_pos_w.torch[0].cpu().numpy() - self.offset

    def velocities(self) -> np.ndarray:
        """Particle velocities [m/s]."""
        return self.particles.data.particle_vel_w.torch[0].cpu().numpy()

    def positions_wp(self) -> wp.array:
        """Particle positions in the berry frame as a Warp array, refreshed on every call."""
        world = self.particles.data.particle_pos_w.warp[0]
        wp.launch(
            _to_local, dim=len(self.rest), inputs=[world, self._offset_vec, self._local], device=self._local.device
        )
        return self._local

    def bind_solver(self, solver) -> None:
        """Read this berry's damage and elastic deformation from ``solver``, which models them."""
        self._solver = solver
        view = solver.damage_view(self.particle_start, len(self.rest))
        self.damage, self.tear, self.history, self.dose = (view[k] for k in ("damage", "tear", "history", "dose"))

    def metrics(self) -> dict:
        """Berry state summary [m, m/s]: extent, center, speed and damage."""
        x = self.positions()
        damage = self.damage.numpy()
        metrics = {
            "center_m": (x.mean(0) + self.offset).tolist(),
            "height_m": float(np.ptp(x[:, 2])),
            "mean_damage": float(damage.mean()),
            "max_damage": float(damage.max()),
            "mean_tear": float(self.tear.numpy().mean()),
            "max_strain_history": float(self.history.numpy().max()),
            "max_bruise_dose": float(self.dose.numpy().max()),
            "tissue_span_m": np.ptp(x, axis=0).tolist(),
            "max_speed_m_s": float(np.linalg.norm(self.velocities(), axis=1).max()),
        }
        # Largest principal compression [Pa]: linear elasticity in the principal stretches of the elastic deformation.
        _, state, model = tissue_solver()
        span = slice(self.particle_start, self.particle_start + len(self.rest))
        elastic = self._solver.elastic_strain(state, self.particle_start, len(self.rest)).numpy()
        strain = np.linalg.svd(elastic, compute_uv=False) - 1.0
        young = model.mpm.young_modulus.numpy()[span, None]
        poisson = model.mpm.poisson_ratio.numpy()[span, None]
        lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        principal = lame * strain.sum(1, keepdims=True) + young / (1.0 + poisson) * strain
        metrics["max_compression_pa"] = float(max(0.0, -principal.min()))
        world = x + self.offset
        inside = (np.linalg.norm(world[:, :2] - BOWL[:2], axis=1) < BOWL[3]) & (
            (world[:, 2] >= BOWL[4] - 0.001) & (world[:, 2] < BOWL[5])
        )
        metrics["fraction_in_bowl"] = float(inside.mean())
        return metrics
