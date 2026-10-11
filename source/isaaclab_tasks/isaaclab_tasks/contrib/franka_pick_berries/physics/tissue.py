# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Berry tissue as Newton MPM particles, or as the nodes of a tetrahedral mesh for VBD: loading, placement in the
punnet, scene configuration and a runtime handle.

The tissue solvers (:mod:`.grasp_explicit_mpm`, :mod:`.grasp_implicit_mpm`, :mod:`.grasp_vbd`) model the tissue's
deformation and damage; this module only describes and places the particles or nodes.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg
from isaaclab_newton.sim.spawners.mpm import MPMPointsCfg
from scipy.spatial.transform import Rotation

from pxr import Usd, UsdGeom, Vt

from isaaclab.assets import DeformableObjectCfg
from isaaclab.sim.schemas import apply_volume_deformable_properties
from isaaclab.sim.spawners.materials import spawn_physics_material
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg
from isaaclab.sim.utils import bind_physics_material, clone, create_prim, get_current_stage
from isaaclab.utils import configclass

from ..assets.berry_asset import load_berry_asset, load_berry_tetmesh, rotate_berry_sh
from ..scene.tableware import PUNNET
from . import grasp_vbd
from .materials import particle_material

if TYPE_CHECKING:
    from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg

    from isaaclab.assets import BaseDeformableObject

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
    [m³]; or, for VBD, the tetrahedral mesh's nodes at rest: ``xyz`` [m] and ``tets``."""
    profile: dict
    """The asset's simulation metadata."""
    material: MPMParticleMaterialCfg | NewtonDeformableBodyMaterialCfg
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
    """Load one berry's asset, select its tissue particles, or its tetrahedral mesh for VBD, and give it its
    material."""
    stage, gaussians, particles, profile = load_berry_asset(cfg.berry_asset)
    if cfg.tissue_solver == "vbd":
        if cfg.tissue_resolution != "full":
            raise ValueError("The VBD tissue solver uses the asset's tetrahedral mesh: use full tissue resolution")
        return TissueSpec(
            name, stage, gaussians, load_berry_tetmesh(stage), profile, grasp_vbd.tissue_material(profile), offset
        )
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


def tissue_object_cfg(spec: TissueSpec) -> MPMObjectCfg | DeformableObjectCfg:
    """Return the scene asset that spawns a berry's tissue particles, or its tetrahedral mesh, at its world offset."""
    if "tets" in spec.particles:
        return DeformableObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{spec.scene_name}",
            spawn=TetMeshCfg(
                points=spec.particles["xyz"],
                tets=spec.particles["tets"],
                volume_deformable_props=[],
                physics_material=spec.material,
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=spec.offset, rot=(0.0, 0.0, 0.0, 1.0)),
        )
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


@clone
def spawn_tet_mesh(
    prim_path: str,
    cfg: TetMeshCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Author a volume deformable from a given tetrahedral mesh, with its boundary as the (unrendered) visual mesh."""
    stage = get_current_stage()
    root = create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
    mesh = UsdGeom.TetMesh.Define(stage, f"{prim_path}/sim_mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(cfg.points, np.float32)))
    mesh.CreateTetVertexIndicesAttr(Vt.Vec4iArray.FromNumpy(np.ascontiguousarray(cfg.tets, np.int32)))
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    # Simulates this mesh, and derives the visual mesh from its surface.
    apply_volume_deformable_properties(prim_path, cfg.volume_deformable_props, create_if_missing=True, stage=stage)
    UsdGeom.Imageable(stage.GetPrimAtPath(f"{prim_path}/vis_mesh")).MakeInvisible()
    material_path = f"{prim_path}/physics_material"
    spawn_physics_material(material_path, cfg.physics_material, stage=stage)
    bind_physics_material(prim_path, material_path, stage=stage)
    return root


@configclass
class TetMeshCfg(DeformableObjectSpawnerCfg):
    """A volume deformable from a given tetrahedral mesh, for :func:`spawn_tet_mesh`."""

    func = spawn_tet_mesh
    points: np.ndarray = MISSING
    """Node positions [m], in the asset's frame."""
    tets: np.ndarray = MISSING
    """The four node indices of each tetrahedron, with positive volume."""
    physics_material: NewtonDeformableBodyMaterialCfg = MISSING
    """The tissue's material, bound to the mesh."""


@wp.kernel
def _to_local(world: wp.array[wp.vec3], offset: wp.vec3, local: wp.array[wp.vec3]):
    i = wp.tid()
    local[i] = world[i] - offset


class BerryTissue:
    """Runtime handle over one berry's MPM particles or mesh nodes, in the berry's own frame (world minus its offset).

    Args:
        spec: The berry's description.
        asset: The scene asset holding its particles or nodes.
    """

    def __init__(self, spec: TissueSpec, asset: BaseDeformableObject):
        self.name = spec.name
        self.stage = spec.stage
        self.gaussians = spec.gaussians
        self.particles = spec.particles
        self.profile = spec.profile
        self.asset = asset
        self.device = asset.device
        self.offset = np.asarray(spec.offset, np.float32)
        self.rest = np.asarray(spec.particles["xyz"], np.float32)
        """Particle rest positions in the berry frame [m]."""
        self.particle_start = int(asset._particle_offsets.numpy()[0])
        """Index of the berry's first particle in the solver."""
        self.damage = wp.zeros(len(self.rest), dtype=float, device=self.device)
        """Damage of each particle, from 0 (intact) to 1, on the device; set by :meth:`bind_damage` for a solver that
        models damage."""
        self._offset = wp.vec3(*self.offset)
        self._local = wp.zeros(len(self.rest), dtype=wp.vec3, device=self.device)

    def positions(self) -> np.ndarray:
        """Particle positions in the berry frame [m], on the host."""
        return self.asset.data.nodal_pos_w.torch[0].cpu().numpy() - self.offset

    def positions_warp(self) -> wp.array:
        """Particle positions in the berry frame [m] as a device array, refreshed on every call."""
        world = self.asset.data.nodal_pos_w.warp[0]
        wp.launch(_to_local, dim=len(self.rest), inputs=[world, self._offset, self._local], device=self.device)
        return self._local

    def bind_damage(self, solver) -> None:
        """Read the berry's damage from the tissue solver that models it."""
        self.damage = solver.damage[self.particle_start : self.particle_start + len(self.rest)]
