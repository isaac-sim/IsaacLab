# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import trimesh
from typing import TYPE_CHECKING

import omni.log

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

from .utils import create_prim_from_mesh

if TYPE_CHECKING:
    from .terrain_importer_cfg import TerrainImporterCfg


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

    terrain_prim_paths: list[str]
    """A list containing the USD prim paths to the imported terrains."""

    terrain_origins: torch.Tensor | None
    """The origins of the sub-terrains in the added terrain mesh. Shape is (num_rows, num_cols, 3).

    If terrain origins is not None, the environment origins are computed based on the terrain origins.
    Otherwise, the environment origins are computed based on the grid spacing.
    """

    env_origins: torch.Tensor
    """The origins of the environments. Shape is (num_envs, 3)."""

    def __init__(self, cfg: TerrainImporterCfg):
        """Initialize the terrain importer.

        Args:
            cfg: The configuration for the terrain importer.

        Raises:
            ValueError: If input terrain type is not supported.
            ValueError: If terrain type is 'generator' and no configuration provided for ``terrain_generator``.
            ValueError: If terrain type is 'usd' and no configuration provided for ``usd_path``.
            ValueError: If terrain type is 'usd' or 'plane' and no configuration provided for ``env_spacing``.
        """
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        # create buffers for the terrains
        self.terrain_prim_paths = list()
        self.terrain_origins = None
        self.env_origins = None  # assigned later when `configure_env_origins` is called
        # private variables
        self._terrain_flat_patches = dict()

        # auto-import the terrain based on the config
        if self.cfg.terrain_type == "generator":
            # check config is provided
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            # generate the terrain
            terrain_generator = self.cfg.terrain_generator.class_type(
                cfg=self.cfg.terrain_generator, device=self.device
            )
            self.import_mesh("terrain", terrain_generator.terrain_mesh)
            # configure the terrain origins based on the terrain generator
            self.configure_env_origins(terrain_generator.terrain_origins)
            # refer to the flat patches
            self._terrain_flat_patches = terrain_generator.flat_patches
        elif self.cfg.terrain_type == "usd":
            # check if config is provided
            if self.cfg.usd_path is None:
                raise ValueError("Input terrain type is 'usd' but no value provided for 'usd_path'.")
            # import the terrain
            self.import_usd("terrain", self.cfg.usd_path)
            # configure the origins in a grid
            self.configure_env_origins()
        elif self.cfg.terrain_type == "plane":
            # load the plane
            self.import_ground_plane("terrain")
            # configure the origins in a grid
            self.configure_env_origins()
        else:
            raise ValueError(f"Terrain type '{self.cfg.terrain_type}' not available.")

        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    """
    Properties.
    """

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the terrain importer has a debug visualization implemented.

        This always returns True.
        """
        return True

    @property
    def flat_patches(self) -> dict[str, torch.Tensor]:
        """A dictionary containing the sampled valid (flat) patches for the terrain.

        This is only available if the terrain type is 'generator'. For other terrain types, this feature
        is not available and the function returns an empty dictionary.

        Please refer to the :attr:`TerrainGenerator.flat_patches` for more information.
        """
        return self._terrain_flat_patches

    @property
    def terrain_names(self) -> list[str]:
        """A list of names of the imported terrains."""
        return [f"'{path.split('/')[-1]}'" for path in self.terrain_prim_paths]

    """
    Operations - Visibility.
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Set the debug visualization of the terrain importer.

        Args:
            debug_vis: Whether to visualize the terrain origins.

        Returns:
            Whether the debug visualization was successfully set. False if the terrain
            importer does not support debug visualization.

        Raises:
            RuntimeError: If terrain origins are not configured.
        """
        # create a marker if necessary
        if debug_vis:
            if not hasattr(self, "origin_visualizer"):
                self.origin_visualizer = VisualizationMarkers(
                    cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/TerrainOrigin")
                )
                if self.terrain_origins is not None:
                    self.origin_visualizer.visualize(self.terrain_origins.reshape(-1, 3))
                elif self.env_origins is not None:
                    self.origin_visualizer.visualize(self.env_origins.reshape(-1, 3))
                else:
                    raise RuntimeError("Terrain origins are not configured.")
            # set visibility
            self.origin_visualizer.set_visibility(True)
        else:
            if hasattr(self, "origin_visualizer"):
                self.origin_visualizer.set_visibility(False)
        # report success
        return True

    """
    Operations - Import.
    """

    def import_ground_plane(self, name: str, size: tuple[float, float] = (2.0e6, 2.0e6)):
        """Add a plane to the terrain importer.

        Args:
            name: The name of the imported terrain. This name is used to create the USD prim
                corresponding to the terrain.
            size: The size of the plane. Defaults to (2.0e6, 2.0e6).

        Raises:
            ValueError: If a terrain with the same name already exists.
        """
        # create prim path for the terrain
        prim_path = self.cfg.prim_path + f"/{name}"
        # check if key exists
        if prim_path in self.terrain_prim_paths:
            raise ValueError(
                f"A terrain with the name '{name}' already exists. Existing terrains: {', '.join(self.terrain_names)}."
            )
        # store the mesh name
        self.terrain_prim_paths.append(prim_path)

        # obtain ground plane color from the configured visual material
        color = (0.0, 0.0, 0.0)
        if self.cfg.visual_material is not None:
            material = self.cfg.visual_material.to_dict()
            # defaults to the `GroundPlaneCfg` color if diffuse color attribute is not found
            if "diffuse_color" in material:
                color = material["diffuse_color"]
            else:
                omni.log.warn(
                    "Visual material specified for ground plane but no diffuse color found."
                    " Using default color: (0.0, 0.0, 0.0)"
                )

        # get the mesh
        ground_plane_cfg = sim_utils.GroundPlaneCfg(physics_material=self.cfg.physics_material, size=size, color=color)
        ground_plane_cfg.func(prim_path, ground_plane_cfg)

    def import_mesh(self, name: str, mesh: trimesh.Trimesh):
        """Import a mesh into the simulator.

        The mesh is imported into the simulator under the prim path ``cfg.prim_path/{key}``. The created path
        contains the mesh as a :class:`pxr.UsdGeom` instance along with visual or physics material prims.

        Args:
            name: The name of the imported terrain. This name is used to create the USD prim
                corresponding to the terrain.
            mesh: The mesh to import.

        Raises:
            ValueError: If a terrain with the same name already exists.
        """
        # create prim path for the terrain
        prim_path = self.cfg.prim_path + f"/{name}"
        # check if key exists
        if prim_path in self.terrain_prim_paths:
            raise ValueError(
                f"A terrain with the name '{name}' already exists. Existing terrains: {', '.join(self.terrain_names)}."
            )
        # store the mesh name
        self.terrain_prim_paths.append(prim_path)

        # import the mesh
        create_prim_from_mesh(
            prim_path, mesh, visual_material=self.cfg.visual_material, physics_material=self.cfg.physics_material
        )

    def import_usd(self, name: str, usd_path: str):
        """Import a mesh from a USD file.

        This function imports a USD file into the simulator as a terrain. It parses the USD file and
        stores the mesh under the prim path ``cfg.prim_path/{key}``. If multiple meshes are present in
        the USD file, only the first mesh is imported.

        The function doe not apply any material properties to the mesh. The material properties should
        be defined in the USD file.

        Args:
            name: The name of the imported terrain. This name is used to create the USD prim
                corresponding to the terrain.
            usd_path: The path to the USD file.

        Raises:
            ValueError: If a terrain with the same name already exists.
        """
        # create prim path for the terrain
        prim_path = self.cfg.prim_path + f"/{name}"
        # check if key exists
        if prim_path in self.terrain_prim_paths:
            raise ValueError(
                f"A terrain with the name '{name}' already exists. Existing terrains: {', '.join(self.terrain_names)}."
            )
        # store the mesh name
        self.terrain_prim_paths.append(prim_path)

        # add the prim path
        cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
        cfg.func(prim_path, cfg)

    """
    Operations - Origins.
    """

    def configure_env_origins(self, origins: np.ndarray | torch.Tensor | None = None):
        """Configure the origins of the environments based on the added terrain.

        Args:
            origins: The origins of the sub-terrains. Shape is (num_rows, num_cols, 3).
        """
        # decide whether to compute origins in a grid or based on curriculum
        if origins is not None:
            # convert to numpy
            if isinstance(origins, np.ndarray):
                origins = torch.from_numpy(origins)
            # store the origins
            self.terrain_origins = origins.to(self.device, dtype=torch.float)
            # compute environment origins
            self.env_origins = self._compute_env_origins_curriculum(self.cfg.num_envs, self.terrain_origins)
        else:
            self.terrain_origins = None
            # check if env spacing is valid
            if self.cfg.env_spacing is None:
                raise ValueError("Environment spacing must be specified for configuring grid-like origins.")
            # compute environment origins
            self.env_origins = self._compute_env_origins_grid(self.cfg.num_envs, self.cfg.env_spacing)

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
            torch.arange(num_envs, device=self.device), (num_envs / num_cols), rounding_mode="floor"
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
        num_rows = np.ceil(num_envs / int(np.sqrt(num_envs)))
        num_cols = np.ceil(num_envs / num_rows)
        ii, jj = torch.meshgrid(
            torch.arange(num_rows, device=self.device), torch.arange(num_cols, device=self.device), indexing="ij"
        )
        env_origins[:, 0] = -(ii.flatten()[:num_envs] - (num_rows - 1) / 2) * env_spacing
        env_origins[:, 1] = (jj.flatten()[:num_envs] - (num_cols - 1) / 2) * env_spacing
        env_origins[:, 2] = 0.0
        return env_origins

    """
    Deprecated.
    """

    @property
    def warp_meshes(self):
        """A dictionary containing the terrain's names and their warp meshes.

        .. deprecated:: v2.1.0
            The `warp_meshes` attribute is deprecated. It is no longer stored inside the class.
        """
        omni.log.warn(
            "The `warp_meshes` attribute is deprecated. It is no longer stored inside the `TerrainImporter` class."
            " Returning an empty dictionary."
        )
        return {}

    @property
    def meshes(self) -> dict[str, trimesh.Trimesh]:
        """A dictionary containing the terrain's names and their tri-meshes.

        .. deprecated:: v2.1.0
            The `meshes` attribute is deprecated. It is no longer stored inside the class.
        """
        omni.log.warn(
            "The `meshes` attribute is deprecated. It is no longer stored inside the `TerrainImporter` class."
            " Returning an empty dictionary."
        )
        return {}
