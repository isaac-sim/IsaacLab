# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import trimesh
import trimesh.transformations

from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

from isaaclab.sim import schemas
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone, create_prim, get_current_stage

from ..materials import DeformableBodyMaterialCfg, RigidBodyMaterialCfg, SurfaceDeformableBodyMaterialCfg

if TYPE_CHECKING:
    from . import meshes_cfg


# ---------------------------------------------------------------------------
# TetMesh data class
# ---------------------------------------------------------------------------


class TetMesh:
    """Lightweight container for tetrahedral mesh data.

    Holds vertex positions, tet connectivity, and (optionally) surface triangle
    indices. Surface triangles are computed lazily from the tet connectivity on
    first access if not provided at construction time.

    Use the classmethods to construct from different sources:

    - :meth:`from_meshio` -- from a ``meshio.Mesh`` (e.g. loaded from a ``.msh`` file).
    - :meth:`cuboid` -- procedurally generated cuboid with hex->tet decomposition.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        tet_indices: np.ndarray,
        surface_indices: np.ndarray | None = None,
    ):
        """
        Args:
            vertices: Vertex positions, shape ``(N, 3)``, float.
            tet_indices: Tet connectivity, shape ``(T, 4)``, int.
            surface_indices: Surface triangle indices, shape ``(F, 3)``, int.
                If ``None``, computed automatically from ``tet_indices`` on
                first access via :attr:`surface_indices`.
        """
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.tet_indices = np.asarray(tet_indices, dtype=np.int32)
        self._surface_indices = np.asarray(surface_indices, dtype=np.int32) if surface_indices is not None else None

    @property
    def surface_indices(self) -> np.ndarray:
        """Surface triangle indices, shape ``(F, 3)``. Computed lazily if not provided."""
        if self._surface_indices is None:
            self._surface_indices = self._extract_surface_triangles()
        return self._surface_indices

    def _extract_surface_triangles(self) -> np.ndarray:
        """Extract boundary triangles via open-face tracking."""
        faces: dict[tuple[int, ...], tuple[int, int, int]] = {}

        def add_face(i: int, j: int, k: int):
            key = tuple(sorted((i, j, k)))
            if key not in faces:
                faces[key] = (i, j, k)
            else:
                del faces[key]

        for tet in self.tet_indices:
            i, j, k, m = int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])
            add_face(i, k, j)
            add_face(j, k, m)
            add_face(i, j, m)
            add_face(i, m, k)

        return np.array(list(faces.values()), dtype=np.int32)

    @classmethod
    def from_meshio(cls, mesh) -> TetMesh:
        """Construct from a ``meshio.Mesh`` with ``"tetra"`` cells.

        Args:
            mesh: A ``meshio.Mesh`` object.

        Raises:
            ValueError: If the mesh has no tetrahedral cells.
        """
        tets = mesh.cells_dict.get("tetra")
        if tets is None:
            raise ValueError("meshio.Mesh has no tetrahedral cells.")
        return cls(vertices=mesh.points, tet_indices=tets)

    @classmethod
    def from_file(cls, file_path: str) -> TetMesh:
        """Load a tet mesh from a file (Gmsh ``.msh`` or any meshio-supported format).

        Args:
            file_path: Path to the mesh file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ImportError: If meshio is not installed.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Tet mesh file not found: {file_path}")

        import meshio

        return cls.from_meshio(meshio.read(file_path))

    @classmethod
    def cuboid(
        cls,
        size: tuple[float, float, float],
        resolution: int = 4,
    ) -> TetMesh:
        """Generate a cuboid tet mesh centered at the origin.

        Decomposes a regular hexahedral grid into 5 tetrahedra per cell
        (matching Newton's ``add_soft_grid`` convention).

        Args:
            size: Cuboid dimensions ``(sx, sy, sz)`` [m].
            resolution: Number of cells along each axis. Total tets = ``resolution^3 * 5``.
        """
        sx, sy, sz = size
        n = resolution
        cell_x, cell_y, cell_z = sx / n, sy / n, sz / n
        ox, oy, oz = -sx / 2, -sy / 2, -sz / 2

        vertices = []
        for iz in range(n + 1):
            for iy in range(n + 1):
                for ix in range(n + 1):
                    vertices.append([ox + ix * cell_x, oy + iy * cell_y, oz + iz * cell_z])

        def grid_index(x, y, z):
            return (n + 1) * (n + 1) * z + (n + 1) * y + x

        tet_indices = []
        for iz in range(n):
            for iy in range(n):
                for ix in range(n):
                    v0 = grid_index(ix, iy, iz)
                    v1 = grid_index(ix + 1, iy, iz)
                    v2 = grid_index(ix + 1, iy, iz + 1)
                    v3 = grid_index(ix, iy, iz + 1)
                    v4 = grid_index(ix, iy + 1, iz)
                    v5 = grid_index(ix + 1, iy + 1, iz)
                    v6 = grid_index(ix + 1, iy + 1, iz + 1)
                    v7 = grid_index(ix, iy + 1, iz + 1)

                    if (ix & 1) ^ (iy & 1) ^ (iz & 1):
                        tets = [
                            (v0, v1, v4, v3),
                            (v2, v3, v6, v1),
                            (v5, v4, v1, v6),
                            (v7, v6, v3, v4),
                            (v4, v1, v6, v3),
                        ]
                    else:
                        tets = [
                            (v1, v2, v5, v0),
                            (v3, v0, v7, v2),
                            (v4, v7, v0, v5),
                            (v6, v5, v2, v7),
                            (v5, v2, v7, v0),
                        ]

                    tet_indices.extend(tets)

        return cls(
            vertices=np.array(vertices, dtype=np.float32),
            tet_indices=np.array(tet_indices, dtype=np.int32),
        )


@clone
def spawn_mesh_sphere(
    prim_path: str,
    cfg: meshes_cfg.MeshSphereCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh sphere prim with the given attributes.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # create a trimesh sphere
    sphere = trimesh.creation.uv_sphere(radius=cfg.radius)

    # obtain stage handle
    stage = get_current_stage()
    # spawn the sphere as a mesh
    _spawn_mesh_geom_from_mesh(prim_path, cfg, sphere, translation, orientation, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_mesh_cuboid(
    prim_path: str,
    cfg: meshes_cfg.MeshCuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh cuboid prim with the given attributes.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # create a trimesh box
    box = trimesh.creation.box(cfg.size)

    # obtain stage handle
    stage = get_current_stage()
    # spawn the cuboid as a mesh
    _spawn_mesh_geom_from_mesh(prim_path, cfg, box, translation, orientation, None, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_mesh_cylinder(
    prim_path: str,
    cfg: meshes_cfg.MeshCylinderCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh cylinder prim with the given attributes.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # align axis from "Z" to input by rotating the cylinder
    axis = cfg.axis.upper()
    if axis == "X":
        transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    elif axis == "Y":
        transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    else:
        transform = None
    # create a trimesh cylinder
    cylinder = trimesh.creation.cylinder(radius=cfg.radius, height=cfg.height, transform=transform)

    # obtain stage handle
    stage = get_current_stage()
    # spawn the cylinder as a mesh
    _spawn_mesh_geom_from_mesh(prim_path, cfg, cylinder, translation, orientation, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_mesh_capsule(
    prim_path: str,
    cfg: meshes_cfg.MeshCapsuleCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh capsule prim with the given attributes.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # align axis from "Z" to input by rotating the cylinder
    axis = cfg.axis.upper()
    if axis == "X":
        transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    elif axis == "Y":
        transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    else:
        transform = None
    # create a trimesh capsule
    capsule = trimesh.creation.capsule(radius=cfg.radius, height=cfg.height, transform=transform)

    # obtain stage handle
    stage = get_current_stage()
    # spawn capsule if it doesn't exist.
    _spawn_mesh_geom_from_mesh(prim_path, cfg, capsule, translation, orientation, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_mesh_cone(
    prim_path: str,
    cfg: meshes_cfg.MeshConeCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh cone prim with the given attributes.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # align axis from "Z" to input by rotating the cylinder
    axis = cfg.axis.upper()
    if axis == "X":
        transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    elif axis == "Y":
        transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    else:
        transform = None
    # create a trimesh cone
    cone = trimesh.creation.cone(radius=cfg.radius, height=cfg.height, transform=transform)

    # obtain stage handle
    stage = get_current_stage()
    # spawn cone if it doesn't exist.
    _spawn_mesh_geom_from_mesh(prim_path, cfg, cone, translation, orientation, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_mesh_square(
    prim_path: str,
    cfg: meshes_cfg.MeshSquareCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh 2D square prim with the given attributes.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # create a 2D triangle mesh grid
    from omni.physx.scripts import deformableUtils

    vertices, faces = deformableUtils.create_triangle_mesh_square(cfg.resolution[0], cfg.resolution[1], scale=cfg.size)
    grid = trimesh.Trimesh(vertices=vertices, faces=np.array(faces).reshape(-1, 3), process=False)

    # obtain stage handle
    stage = get_current_stage()
    # spawn the square as a mesh
    _spawn_mesh_geom_from_mesh(prim_path, cfg, grid, translation, orientation, None, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_tet_mesh_cuboid(
    prim_path: str,
    cfg: meshes_cfg.TetMeshCuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a ``UsdGeom.TetMesh`` cuboid prim with tetrahedral volumetric data.

    Generates a regular hexahedral grid and decomposes each hex cell into 5 tetrahedra
    (matching Newton's ``add_soft_grid`` convention). The prim is authored as a
    ``UsdGeom.TetMesh`` with:

    - ``points``: vertex positions
    - ``tetVertexIndices``: tet connectivity (4 ints per tet, flattened)
    - ``surfaceFaceVertexIndices``: surface triangles for rendering (3 ints per face, flattened)

    The Newton backend detects the ``TetMesh`` type and uses ``builder.add_soft_mesh()``.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern.

    Args:
        prim_path: The prim path or pattern to spawn the asset at.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    tet_mesh = TetMesh.cuboid(size=cfg.size, resolution=cfg.resolution)
    return _spawn_tet_mesh_geom_from_tet_mesh(prim_path, cfg, tet_mesh, translation, orientation)


@clone
def spawn_tet_mesh_from_file(
    prim_path: str,
    cfg: meshes_cfg.TetMeshFromFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a ``UsdGeom.TetMesh`` prim from a Gmsh ``.msh`` tet mesh file.

    Loads tetrahedral mesh data (vertices + tet indices) via meshio and authors the
    same prim structure as :func:`spawn_tet_mesh_cuboid`: a ``UsdGeom.TetMesh`` with
    ``points``, ``tetVertexIndices``, and ``surfaceFaceVertexIndices``.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern.

    Args:
        prim_path: The prim path or pattern to spawn the asset at.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
        FileNotFoundError: If the mesh file does not exist.
        ImportError: If meshio is not installed.
    """
    tet_mesh = TetMesh.from_file(cfg.file_path)
    if cfg.scale is not None:
        scale = cfg.scale if isinstance(cfg.scale, tuple) else (cfg.scale, cfg.scale, cfg.scale)
        tet_mesh.vertices = tet_mesh.vertices * np.array(scale, dtype=np.float32)
    return _spawn_tet_mesh_geom_from_tet_mesh(prim_path, cfg, tet_mesh, translation, orientation)


@clone
def spawn_mesh_from_file(
    prim_path: str,
    cfg: meshes_cfg.MeshFromFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh prim from mesh geometry in an external USD file.

    Reads vertices and face indices from a USD file and spawns them as a
    ``UsdGeom.Mesh`` prim in the current stage. This avoids USD reference
    composition issues with files that lack a ``defaultPrim``.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
        FileNotFoundError: If the USD file does not exist.
        RuntimeError: If no mesh geometry is found in the USD file.
    """
    # Open the external USD file
    ext_stage = Usd.Stage.Open(cfg.usd_path)
    if ext_stage is None:
        raise FileNotFoundError(f"USD file not found at path: '{cfg.usd_path}'.")

    # Find the mesh prim in the external stage
    mesh_prim = None
    if cfg.usd_prim_path is not None:
        mesh_prim = ext_stage.GetPrimAtPath(cfg.usd_prim_path)
        if not mesh_prim.IsValid():
            raise RuntimeError(f"Prim '{cfg.usd_prim_path}' not found in '{cfg.usd_path}'.")
    else:
        # Try default prim
        default_prim = ext_stage.GetDefaultPrim()
        if default_prim and default_prim.IsValid():
            mesh_prim = default_prim
        else:
            # Search for first Mesh child
            for child in ext_stage.GetPseudoRoot().GetAllChildren():
                if child.IsA(UsdGeom.Mesh):
                    mesh_prim = child
                    break
                for grandchild in child.GetAllChildren():
                    if grandchild.IsA(UsdGeom.Mesh):
                        mesh_prim = grandchild
                        break
                if mesh_prim is not None:
                    break

    if mesh_prim is None:
        raise RuntimeError(f"No mesh geometry found in '{cfg.usd_path}'.")

    # Read mesh data
    usd_mesh = UsdGeom.Mesh(mesh_prim)
    pts = np.array(usd_mesh.GetPointsAttr().Get(), dtype=np.float32)
    face_indices = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    face_counts = np.array(usd_mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)

    # Triangulate if needed (trimesh expects all triangles)
    if np.all(face_counts == 3):
        faces = face_indices.reshape(-1, 3)
    else:
        # Manual fan triangulation for mixed quads/tris
        tris = []
        idx = 0
        for count in face_counts:
            for i in range(1, count - 1):
                tris.append([face_indices[idx], face_indices[idx + i], face_indices[idx + i + 1]])
            idx += count
        faces = np.array(tris, dtype=np.int32)

    # Apply vertex scale if specified (e.g. cm -> m conversion)
    if cfg.scale is not None:
        if isinstance(cfg.scale, (int, float)):
            scale_vec = np.array([cfg.scale, cfg.scale, cfg.scale], dtype=np.float32)
        else:
            scale_vec = np.array(cfg.scale, dtype=np.float32)
        pts = pts * scale_vec

    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=pts, faces=faces, process=False)

    # Obtain stage handle
    stage = get_current_stage()
    # Spawn the mesh
    _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation, stage=stage)
    # Return the prim
    return stage.GetPrimAtPath(prim_path)


"""
Helper functions.
"""


def _spawn_tet_mesh_geom_from_tet_mesh(
    prim_path: str,
    cfg: meshes_cfg.MeshCfg,
    tet_mesh: TetMesh,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    stage: Usd.Stage | None = None,
) -> Usd.Prim:
    """Create a ``UsdGeom.TetMesh`` prim from a :class:`TetMesh` and apply property schemas.

    This is the tet-mesh counterpart to :func:`_spawn_mesh_geom_from_mesh`.  It writes
    vertex positions, tet connectivity, and surface triangles to a ``UsdGeom.TetMesh``
    prim, then applies mass, deformable body, collision, visual material, physics material,
    and rigid body properties from the config.

    Args:
        prim_path: The prim path to spawn the asset at.
        cfg: The config containing the properties to apply.
        tet_mesh: A :class:`TetMesh` with vertices, tet indices, and surface indices.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None.
        stage: The stage to spawn the asset at. Defaults to None, in which case the current stage is used.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    stage = stage if stage is not None else get_current_stage()

    # Convert TetMesh data to USD types
    vertices = Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in tet_mesh.vertices])
    tet_verts = Vt.Vec4iArray([Gf.Vec4i(int(t[0]), int(t[1]), int(t[2]), int(t[3])) for t in tet_mesh.tet_indices])
    surf_faces = Vt.Vec3iArray([Gf.Vec3i(int(f[0]), int(f[1]), int(f[2])) for f in tet_mesh.surface_indices])

    # Create prim hierarchy: parent Xform with transform, child TetMesh with geometry
    create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)

    geom_prim_path = prim_path + "/geometry"
    mesh_prim_path = geom_prim_path + "/mesh"

    create_prim(
        mesh_prim_path,
        prim_type="TetMesh",
        attributes={
            "points": vertices,
            "tetVertexIndices": tet_verts,
            "surfaceFaceVertexIndices": surf_faces,
        },
        stage=stage,
    )

    # Apply property schemas (mirrors _spawn_mesh_geom_from_mesh)
    if cfg.deformable_props is not None:
        # deformable body properties are PhysX-specific
        schemas.define_deformable_body_properties(prim_path, cfg.deformable_props, stage=stage)
        if cfg.mass_props is not None:
            raise ValueError(
                """MassPropertiesCfg are not supported for deformable bodies
                and should be set through DeformableBodyPropertiesCfg(mass=<value>)."""
            )
    elif cfg.collision_props is not None:
        schemas.define_collision_properties(mesh_prim_path, cfg.collision_props, stage=stage)

    # Visual material
    if cfg.visual_material is not None:
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.visual_material_path}"
        else:
            material_path = cfg.visual_material_path
        cfg.visual_material.func(material_path, cfg.visual_material)
        bind_visual_material(mesh_prim_path, material_path, stage=stage)

    # Physics material
    if cfg.physics_material is not None:
        if not cfg.physics_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.physics_material_path}"
        else:
            material_path = cfg.physics_material_path
        cfg.physics_material.func(material_path, cfg.physics_material)
        bind_physics_material(mesh_prim_path, material_path, stage=stage)

    # Rigid body properties (applied to parent prim)
    if cfg.rigid_props is not None:
        if cfg.mass_props is not None:
            schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)

    return stage.GetPrimAtPath(prim_path)


def _spawn_mesh_geom_from_mesh(
    prim_path: str,
    cfg: meshes_cfg.MeshCfg,
    mesh: trimesh.Trimesh,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    scale: tuple[float, float, float] | None = None,
    stage: Usd.Stage | None = None,
    **kwargs,
):
    """Create a `USDGeomMesh`_ prim from the given mesh.

    This function is similar to :func:`shapes._spawn_geom_from_prim_type` but spawns the prim from a given mesh.
    In case of the mesh, it is spawned as a USDGeomMesh prim with the given vertices and faces.

    There is a difference in how the properties are applied to the prim based on the type of object:

    - Deformable body properties: The properties are applied to the mesh prim: ``{prim_path}/geometry/mesh``.
    - Collision properties: The properties are applied to the mesh prim: ``{prim_path}/geometry/mesh``.
    - Rigid body properties: The properties are applied to the parent prim: ``{prim_path}``.

    Args:
        prim_path: The prim path to spawn the asset at.
        cfg: The config containing the properties to apply.
        mesh: The mesh to spawn the prim from.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        scale: The scale to apply to the prim. Defaults to None, in which case this is set to identity.
        stage: The stage to spawn the asset at. Defaults to None, in which case the current stage is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Raises:
        ValueError: If a prim already exists at the given path.
        ValueError: If both deformable and rigid properties are used.
        ValueError: If both deformable and collision properties are used.
        ValueError: If the physics material is not of the correct type. Deformable properties require a deformable
            physics material, and rigid properties require a rigid physics material.

    .. _USDGeomMesh: https://openusd.org/dev/api/class_usd_geom_mesh.html
    """
    # obtain stage handle
    stage = stage if stage is not None else get_current_stage()

    # spawn geometry if it doesn't exist.
    if not stage.GetPrimAtPath(prim_path).IsValid():
        create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")
    # check that invalid schema types are not used
    if cfg.deformable_props is not None and cfg.rigid_props is not None:
        raise ValueError("Cannot use both deformable and rigid properties at the same time.")
    if cfg.deformable_props is not None and cfg.collision_props is not None:
        raise ValueError("Cannot use both deformable and collision properties at the same time.")
    # check material types are correct
    if cfg.deformable_props is not None and cfg.physics_material is not None:
        if not isinstance(cfg.physics_material, DeformableBodyMaterialCfg):
            raise ValueError("Deformable properties require a deformable physics material.")
    if cfg.rigid_props is not None and cfg.physics_material is not None:
        if not isinstance(cfg.physics_material, RigidBodyMaterialCfg):
            raise ValueError("Rigid properties require a rigid physics material.")

    # create all the paths we need for clarity
    geom_prim_path = prim_path + "/geometry"
    mesh_prim_path = geom_prim_path + "/mesh"

    # create the mesh prim
    mesh_prim = create_prim(
        mesh_prim_path,
        prim_type="Mesh",
        scale=scale,
        attributes={
            "points": mesh.vertices,
            "faceVertexIndices": mesh.faces.flatten(),
            "faceVertexCounts": np.asarray([3] * len(mesh.faces)),
            "subdivisionScheme": "bilinear",
        },
        stage=stage,
    )

    if cfg.deformable_props is not None:
        # apply deformable body properties
        deformable_type = "surface" if isinstance(cfg.physics_material, SurfaceDeformableBodyMaterialCfg) else "volume"
        schemas.define_deformable_body_properties(
            prim_path, cfg.deformable_props, stage=stage, deformable_type=deformable_type
        )
        if cfg.mass_props is not None:
            raise ValueError(
                """MassPropertiesCfg are not supported for deformable bodies
                and should be set through DeformableBodyPropertiesCfg(mass=<value>)."""
            )
    elif cfg.collision_props is not None:
        # decide on type of collision approximation based on the mesh
        if cfg.__class__.__name__ == "MeshSphereCfg":
            collision_approximation = "boundingSphere"
        elif cfg.__class__.__name__ == "MeshCuboidCfg":
            collision_approximation = "boundingCube"
        else:
            # for: MeshCylinderCfg, MeshCapsuleCfg, MeshConeCfg
            collision_approximation = "convexHull"
        # apply collision approximation to mesh
        # note: for primitives, we use the convex hull approximation -- this should be sufficient for most cases.
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mesh_collision_api.GetApproximationAttr().Set(collision_approximation)
        # apply collision properties
        schemas.define_collision_properties(mesh_prim_path, cfg.collision_props, stage=stage)

    # apply visual material
    if cfg.visual_material is not None:
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.visual_material_path}"
        else:
            material_path = cfg.visual_material_path
        # create material
        cfg.visual_material.func(material_path, cfg.visual_material)
        # apply material
        bind_visual_material(mesh_prim_path, material_path, stage=stage)

    # apply physics material
    if cfg.physics_material is not None:
        if not cfg.physics_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.physics_material_path}"
        else:
            material_path = cfg.physics_material_path
        # create material
        cfg.physics_material.func(material_path, cfg.physics_material)
        # apply material
        bind_physics_material(prim_path, material_path, stage=stage)

    # note: we apply the rigid properties to the parent prim in case of rigid objects.
    if cfg.rigid_props is not None:
        # apply mass properties
        if cfg.mass_props is not None:
            schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
        # apply rigid properties
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
