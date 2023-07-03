# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import torch
import trimesh
from typing import Dict, Optional, Tuple

import omni.isaac.core.utils.prims as prim_utils
import warp
from pxr import UsdGeom

from omni.isaac.orbit.compat.markers import StaticMarker
from omni.isaac.orbit.utils.kit import create_ground_plane

from .terrain_cfg import TerrainImporterCfg
from .trimesh.utils import make_plane
from .utils import convert_to_warp_mesh, create_prim_from_mesh


class TerrainImporter:
    r"""A class to handle terrain meshes and import them into the simulator.

    We assume that a terrain mesh comprises of sub-terrains that are arranged in a grid with
    rows ``num_rows`` and columns ``num_cols``. The terrain origins are the positions of the sub-terrains
    where the robot should be spawned.

    Based on the configuration, the terrain importer handles computing the environment origins from the sub-terrain
    origins. In a typical setup, the number of sub-terrains (:math:`num\_rows \times num\_cols`) is smaller than
    the number of environments (:math:`num\_envs`). In this case, the environment origins are computed by
    sampling the sub-terrain origins.

    If a curriculum is used, it is possible to update the environment origins to terrain origins that correspond
    to a harder difficulty. This is done by calling :func:`update_terrain_levels`. The idea comes from game-based
    curriculum. For example, in a game, the player starts with easy levels and progresses to harder levels.
    """

    meshes: Dict[str, trimesh.Trimesh]
    """A dictionary containing the names of the meshes and their keys."""
    warp_meshes: Dict[str, warp.Mesh]
    """A dictionary containing the names of the warp meshes and their keys."""
    terrain_origins: Optional[torch.Tensor]
    """The origins of the sub-terrains in the added terrain mesh. Shape is (num_rows, num_cols, 3).

    If :obj:`None`, then it is assumed no sub-terrains exist. The environment origins are computed in a grid.
    """
    env_origins: torch.Tensor
    """The origins of the environment instances. Shape is (num_envs, 3)."""

    def __init__(self, cfg: TerrainImporterCfg, device: str = "cuda"):
        """Initialize the terrain importer.

        Args:
            cfg (TerrainImporterCfg): The configuration for the terrain importer.
            device (str, optional): The device to use. Defaults to "cuda".
        """
        # store inputs
        self.cfg = cfg
        self.device = device

        # create a dict of meshes
        self.meshes = dict()
        self.warp_meshes = dict()
        self.origins = None

    def import_ground_plane(self, size: Tuple[int, int] = (2.0e6, 2.0e6), key: str = "terrain", **kwargs):
        """Add a plane to the terrain importer.

        Args:
            size (Tuple[int, int], optional): The size of the plane. Defaults to (2.0e6, 2.0e6).
            key (str, optional): The key to store the mesh. Defaults to "terrain".

        Raises:
            ValueError: If a terrain with the same key already exists.
        """
        # create a plane
        mesh = make_plane(size, height=0.0)
        # store the mesh
        self.meshes[key] = mesh
        # create a warp mesh
        device = "cuda" if "cuda" in self.device else "cpu"
        self.warp_meshes[key] = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)

        # properties for the terrain
        mesh_props = {
            "color": self.cfg.color,
            "static_friction": self.cfg.static_friction,
            "dynamic_friction": self.cfg.dynamic_friction,
            "restitution": self.cfg.restitution,
            "improve_patch_friction": self.cfg.improve_patch_friction,
            "combine_mode": self.cfg.combine_mode,
        }
        # update the properties
        mesh_props.update(kwargs)
        # import the grid mesh
        create_ground_plane(self.cfg.prim_path, **mesh_props)

    def import_mesh(self, mesh: trimesh.Trimesh, key: str = "terrain", **kwargs):
        """Import a mesh into the simulator.

        The mesh is imported into the simulator under the prim path ``cfg.prim_path/{key}``. The created path
        contains the mesh as a :class:`pxr.UsdGeom` instance along with visual or physics material prims.

        Args:
            mesh (trimesh.Trimesh): The mesh to import.
            key (str, optional): The key to store the mesh. Defaults to "terrain".
            **kwargs: The properties of the mesh. If not provided, the default properties are used.

        Raises:
            ValueError: If a terrain with the same key already exists.
        """
        # add mesh to the dict
        if key in self.meshes:
            raise ValueError(f"Mesh with key {key} already exists. Existing keys: {self.meshes.keys()}.")
        # store the mesh
        self.meshes[key] = mesh
        # create a warp mesh
        device = "cuda" if "cuda" in self.device else "cpu"
        self.warp_meshes[key] = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)

        # get the mesh
        mesh = self.meshes[key]
        mesh_prim_path = self.cfg.prim_path + f"/{key}"
        # properties for the terrain
        mesh_props = {
            "color": self.cfg.color,
            "static_friction": self.cfg.static_friction,
            "dynamic_friction": self.cfg.dynamic_friction,
            "restitution": self.cfg.restitution,
            "improve_patch_friction": self.cfg.improve_patch_friction,
            "combine_mode": self.cfg.combine_mode,
        }
        # update the properties
        mesh_props.update(kwargs)
        # import the mesh
        create_prim_from_mesh(mesh_prim_path, mesh.vertices, mesh.faces, **mesh_props)

    def import_usd(self, usd_path: str, key: str = "terrain"):
        """Import a mesh from a USD file.

        We assume that the USD file contains a single mesh. If the USD file contains multiple meshes, then
        the first mesh is used. The function mainly helps in registering the mesh into the warp meshes
        and the meshes dictionary.

        Note:
            We do not apply any material properties to the mesh. The material properties should
            be defined in the USD file.

        Args:
            usd_path (str): The path to the USD file.
            key (str, optional): The key to store the mesh. Defaults to "terrain".

        Raises:
            ValueError: If a terrain with the same key already exists.
        """
        # add mesh to the dict
        if key in self.meshes:
            raise ValueError(f"Mesh with key {key} already exists. Existing keys: {self.meshes.keys()}.")
        # add the prim path
        prim_utils.create_prim(self.cfg.prim_path + f"/{key}", usd_path=usd_path)
        # traverse the prim and get the collision mesh
        # THINK: Should the user specify the collision mesh?
        mesh_prim = prim_utils.get_first_matching_child_prim(
            self.cfg.prim_path + f"/{key}", lambda p: prim_utils.get_prim_type_name(p) == "Mesh"
        )
        # check if the mesh is valid
        if mesh_prim is None:
            raise ValueError(f"Could not find any collision mesh in {usd_path}. Please check asset.")
        # cast into UsdGeomMesh
        mesh_prim = UsdGeom.Mesh(mesh_prim)
        # store the mesh
        vertices = np.asarray(mesh_prim.GetPointsAttr().Get())
        faces = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        self.meshes[key] = trimesh.Trimesh(vertices=vertices, faces=faces)
        # create a warp mesh
        device = "cuda" if "cuda" in self.device else "cpu"
        self.warp_meshes[key] = convert_to_warp_mesh(vertices, faces, device=device)

    def configure_env_origins(self, num_envs: int, origins: Optional[np.ndarray] = None):
        """Configure the origins of the environments based on the added terrain.

        Args:
            num_envs (int): The number of environment origins to define.
            origins (Optional[np.ndarray]): The origins of the sub-terrains. Shape: (num_rows, num_cols, 3).
        """
        # decide whether to compute origins in a grid or based on curriculum
        if origins is not None:
            # convert to numpy
            if isinstance(origins, np.ndarray):
                origins = torch.from_numpy(origins)
            # store the origins
            self.terrain_origins = origins.to(self.device, dtype=torch.float)
            # compute environment origins
            self.env_origins = self._compute_env_origins_curriculum(num_envs, self.terrain_origins)
            # create markers for terrain origins
            num_rows, num_cols = self.terrain_origins.shape[:2]
            markers = StaticMarker(f"{self.cfg.prim_path}/originMarkers", count=num_rows * num_cols, scale=[0.5] * 3)
            markers.set_world_poses(self.terrain_origins.reshape(-1, 3))
        else:
            self.terrain_origins = None
            # compute environment origins
            self.env_origins = self._compute_env_origins_grid(num_envs, self.cfg.env_spacing)

    def update_env_origins(self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor):
        """Update the environment origins based on the terrain levels."""
        # check if grid-like spawning
        if self.terrain_origins is None:
            return
        # update terrain level for the envs
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # robots that solve the last level are sent to a random one
        # the minimum level is zero
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )
        # update the env origins
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    """
    Internal helpers.
    """

    def _compute_env_origins_curriculum(self, num_envs: int, origins: torch.Tensor) -> torch.Tensor:
        """Compute the origins of the environments defined by the sub-terrains origins."""
        # extract number of rows and cols
        num_rows, num_cols = origins.shape[:2]
        # maximum initial level possible for the terrains
        if self.cfg.max_init_terrain_level is None:
            max_init_level = num_rows - 1
        else:
            max_init_level = min(self.cfg.max_init_terrain_level, num_rows - 1)
        # store maximum terrain level possible
        self.max_terrain_level = num_rows
        # define all terrain levels and types available
        self.terrain_levels = torch.randint(0, max_init_level + 1, (num_envs,), device=self.device)
        self.terrain_types = torch.div(
            torch.arange(num_envs, device=self.device),
            (num_envs / num_cols),
            rounding_mode="floor",
        ).to(torch.long)
        # create tensor based on number of environments
        env_origins = torch.zeros(num_envs, 3, device=self.device)
        env_origins[:] = origins[self.terrain_levels, self.terrain_types]
        return env_origins

    def _compute_env_origins_grid(self, num_envs: int, env_spacing: float) -> torch.Tensor:
        """Compute the origins of the environments in a grid based on configured spacing."""
        # create tensor based on number of environments
        env_origins = torch.zeros(num_envs, 3, device=self.device)
        # create a grid of origins
        num_cols = np.floor(np.sqrt(num_envs))
        num_rows = np.ceil(num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
        env_origins[:, 0] = env_spacing * xx.flatten()[:num_envs] - env_spacing * (num_rows - 1) / 2
        env_origins[:, 1] = env_spacing * yy.flatten()[:num_envs] - env_spacing * (num_cols - 1) / 2
        env_origins[:, 2] = 0.0
        return env_origins
