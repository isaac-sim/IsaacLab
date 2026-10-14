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
from ..scene.tableware import PunnetSpec
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
    """Instance name, numbered (``raspberry_1``)."""
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


FIXED_LAYOUT = (
    ((0.186517, -0.195973, 0.950047, 0.155616), (-0.02376, -0.02324)),
    ((-0.308495, 0.208245, 0.750981, 0.545429), (0.02265, -0.01804)),
    ((0.446269, 0.802458, 0.395245, -0.026207), (0.00001, 0.01965)),
)
"""The fixed layout of several berries, which the scripted demo's camera work is framed for: each berry's orientation
(quaternion x, y, z, w) and the center (x, y) [m] of its Gaussians' bounds about the punnet center. Fewer berries take
the first ones. The berries are scattered and turned, yet each grasp's fingers clear the punnet's wall and the berries
still in it."""


def fixed_punnet_poses(shape: np.ndarray, tissue: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the :data:`FIXED_LAYOUT` poses of ``count`` copies of a berry, as :func:`punnet_poses` does."""
    if count > len(FIXED_LAYOUT):
        raise ValueError(f"The fixed layout holds up to {len(FIXED_LAYOUT)} berries, not {count}")
    orientations, centers = zip(*FIXED_LAYOUT[:count])
    return punnet_poses(shape, tissue, Rotation.from_quat(orientations).as_matrix(), np.asarray(centers))


def random_punnet_poses(
    shape: np.ndarray, tissue: np.ndarray, count: int, seed: int, punnet: PunnetSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Sample separated centers and full 3D orientations for ``count`` copies of a berry inside ``punnet``, and return
    their poses as :func:`punnet_poses` does.

    Extra wall and berry clearance leaves room for a vertical gripper approach, whose fingers close along y: along x,
    the berries are wider than the fingers, and keep clear of the walls only.
    """
    rng = np.random.default_rng(seed)
    rotations = Rotation.random(count, random_state=rng).as_matrix()
    center = (shape.min(0) + shape.max(0)) / 2
    radius = max(float(np.linalg.norm(((shape - center) @ r.T)[:, :2], axis=1).max()) for r in rotations)
    bounds = np.array(punnet.floor_outline[:2]) - radius - np.array([0.004, 0.016])
    if np.any(bounds <= 0):
        raise ValueError("The berries must fit inside the punnet with gripper clearance")
    # A small punnet leaves few separated layouts: test candidates in batches, and take the first that fits.
    pairs = [(i, j) for i in range(count) for j in range(i)]
    for _ in range(20):
        candidates = rng.uniform(-bounds, bounds, size=(100000, count, 2))
        fits = np.ones(len(candidates), bool)
        for i, j in pairs:
            fits &= np.linalg.norm(candidates[:, i] - candidates[:, j], axis=1) >= 2 * radius + 0.014
        if fits.any():
            centers = candidates[np.argmax(fits)]
            break
    else:
        raise ValueError("Unable to fit separated berries with gripper clearance in the punnet")
    return punnet_poses(shape, tissue, rotations, np.asarray(sorted(centers, key=lambda c: c[1])))


def punnet_poses(
    shape: np.ndarray, tissue: np.ndarray, rotations: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return each copy of a berry's rotation matrix and its translation [m] in the berry frame, about the punnet
    center, that puts its rotated shape's center at ``centers`` and keeps the rotated tissue's bottom at the height
    of the unrotated tissue's, on the punnet floor.

    Args:
        shape: Points [m] that lay the berries out: the Gaussians' centers, so that every tissue solver gets the same
            layout.
        tissue: The tissue's particles or mesh nodes [m], which rest on the floor.
        rotations: Each copy's rotation matrix.
        centers: Each copy's horizontal center [m] about the punnet center.
    """
    center = (shape.min(0) + shape.max(0)) / 2
    floor = float(tissue[:, 2].min())
    bottoms = [float(((tissue - center) @ r.T)[:, 2].min()) for r in rotations]
    shifts = [np.array([*xy, floor - bottom]) - r @ center for r, bottom, xy in zip(rotations, bottoms, centers)]
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
