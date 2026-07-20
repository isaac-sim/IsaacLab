# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.warp import ProxyArray

from .utils import create_prim_from_mesh

if TYPE_CHECKING:
    from .terrain_importer_cfg import TerrainImporterCfg

# import logger
logger = logging.getLogger(__name__)


@wp.kernel
def _update_env_origins_mask(
    env_mask: wp.array(dtype=wp.bool),
    move_up: wp.array(dtype=wp.bool),
    move_down: wp.array(dtype=wp.bool),
    rng_state: wp.array(dtype=wp.uint32),
    terrain_levels: wp.array(dtype=wp.int64),
    terrain_types: wp.array(dtype=wp.int64),
    terrain_origins: wp.array(dtype=wp.vec3f, ndim=2),
    env_origins: wp.array(dtype=wp.vec3f),
    max_terrain_level: int,
):
    """Update terrain levels and origins selected by an environment mask."""
    env_id = wp.tid()
    if env_mask[env_id]:
        state = rng_state[env_id]
        random_level = wp.randi(state, 0, max_terrain_level)
        # Keep shared terrain indices int64 so their zero-copy Torch views support advanced indexing.
        level = terrain_levels[env_id] + wp.int64(move_up[env_id]) - wp.int64(move_down[env_id])
        if level >= wp.int64(max_terrain_level):
            level = wp.int64(random_level)
        elif level < wp.int64(0):
            level = wp.int64(0)
        terrain_levels[env_id] = level
        env_origins[env_id] = terrain_origins[level, terrain_types[env_id]]
        rng_state[env_id] = state


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
        self._terrain_origins: ProxyArray | None = None
        self._env_origins: ProxyArray
        self._terrain_levels: ProxyArray | None = None
        self._terrain_types: ProxyArray | None = None
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
            if self.cfg.use_terrain_origins:
                # configure the terrain origins based on the terrain generator
                self.configure_env_origins(terrain_generator.terrain_origins)
            else:
                self.configure_env_origins()
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

    @property
    def terrain_origins(self) -> ProxyArray | None:
        """Sub-terrain origins [m], shape ``(num_rows, num_cols)`` of ``wp.vec3f``.

        The :attr:`~isaaclab.utils.warp.ProxyArray.torch` view has shape ``(num_rows, num_cols, 3)``.
        This is ``None`` when environment origins are configured from grid spacing.
        """
        return self._terrain_origins

    @property
    def env_origins(self) -> ProxyArray:
        """Environment origins [m], shape ``(num_envs,)`` of ``wp.vec3f``.

        The :attr:`~isaaclab.utils.warp.ProxyArray.torch` view has shape ``(num_envs, 3)``.
        """
        return self._env_origins

    @property
    def terrain_levels(self) -> ProxyArray | None:
        """Terrain level per environment, shape ``(num_envs,)`` of ``wp.int64``.

        This is ``None`` when environment origins are configured from grid spacing.
        """
        return self._terrain_levels

    @property
    def terrain_types(self) -> ProxyArray | None:
        """Terrain type per environment, shape ``(num_envs,)`` of ``wp.int64``.

        This is ``None`` when environment origins are configured from grid spacing.
        """
        return self._terrain_types

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
                    self.origin_visualizer.visualize(self.terrain_origins.torch.reshape(-1, 3))
                else:
                    self.origin_visualizer.visualize(self.env_origins.torch.reshape(-1, 3))
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
                logger.warning(
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

    def configure_env_origins(self, origins: np.ndarray | torch.Tensor | None = None) -> None:
        """Configure the origins of the environments based on the added terrain.

        Args:
            origins: The origins [m] of the sub-terrains. Shape is (num_rows, num_cols, 3).

        Note:
            This setup operation replaces the :class:`~isaaclab.utils.warp.ProxyArray` instances. Call it before
            managers or kernels cache their ``.warp`` views.
        """
        # decide whether to compute origins in a grid or based on curriculum
        if origins is not None:
            self._configure_env_origins_curriculum(self.cfg.num_envs, origins)
        else:
            # check if env spacing is valid
            if self.cfg.env_spacing is None:
                raise ValueError("Environment spacing must be specified for configuring grid-like origins.")
            self._configure_env_origins_grid(self.cfg.num_envs, self.cfg.env_spacing)

    def update_env_origins(self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor) -> None:
        """Update environment origins through the Torch ID-based interface.

        Args:
            env_ids: Environment indices to update.
            move_up: Flags that increment the selected terrain levels.
            move_down: Flags that decrement the selected terrain levels.
        """
        # check if grid-like spawning
        if self.terrain_origins is None or self.terrain_levels is None or self.terrain_types is None:
            return
        terrain_origins = self.terrain_origins.torch
        terrain_levels = self.terrain_levels.torch
        terrain_types = self.terrain_types.torch
        env_origins = self.env_origins.torch
        # update terrain level for the envs
        terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # robots that solve the last level are sent to a random one
        # the minimum level is zero
        terrain_levels[env_ids] = torch.where(
            terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(terrain_levels[env_ids], 0),
        )
        # update the env origins
        env_origins[env_ids] = terrain_origins[terrain_levels[env_ids], terrain_types[env_ids]]

    def update_env_origins_mask(
        self,
        env_mask: wp.array(dtype=wp.bool),
        move_up: wp.array(dtype=wp.bool),
        move_down: wp.array(dtype=wp.bool),
        rng_state: wp.array(dtype=wp.uint32),
    ) -> None:
        """Update terrain levels and origins through the Warp mask-based interface.

        Moving past the maximum level wraps to a random level, moving below zero clamps to zero, and
        unselected environments remain unchanged.

        Args:
            env_mask: Boolean mask selecting environments, shape ``(num_envs,)``.
            move_up: Flags that increment terrain levels, shape ``(num_envs,)``.
            move_down: Flags that decrement terrain levels, shape ``(num_envs,)``.
            rng_state: Per-environment random-number-generator state, shape ``(num_envs,)``.

        Raises:
            TypeError: If an input is not a Warp array with the required data type.
            ValueError: If an input has the wrong shape or device.
        """
        if self.terrain_origins is None or self.terrain_levels is None or self.terrain_types is None:
            return
        num_envs = self.terrain_levels.shape[0]
        arrays = {
            "env_mask": (env_mask, wp.bool),
            "move_up": (move_up, wp.bool),
            "move_down": (move_down, wp.bool),
            "rng_state": (rng_state, wp.uint32),
        }
        for name, (array, dtype) in arrays.items():
            if not isinstance(array, wp.array) or array.dtype != dtype:
                raise TypeError(f"{name} must be a Warp array with dtype {dtype}; received {array}.")
            if array.ndim != 1 or array.shape[0] != num_envs:
                raise ValueError(f"{name} must have shape ({num_envs},); received {array.shape}.")
            if array.device != wp.get_device(self.device):
                raise ValueError(f"{name} must be on device {self.device}; received {array.device}.")

        wp.launch(
            kernel=_update_env_origins_mask,
            dim=num_envs,
            inputs=[
                env_mask,
                move_up,
                move_down,
                rng_state,
                self.terrain_levels.warp,
                self.terrain_types.warp,
                self.terrain_origins.warp,
                self.env_origins.warp,
                self.max_terrain_level,
            ],
            device=self.device,
        )

    """
    Internal helpers.
    """

    def _configure_env_origins_curriculum(self, num_envs: int, origins: np.ndarray | torch.Tensor) -> None:
        """Allocate and initialize origin buffers from sub-terrain origins."""
        origins_torch = torch.as_tensor(origins, dtype=torch.float32, device=self.device).contiguous()
        # ``wp.from_torch`` is zero-copy and keeps the source tensor alive through the Warp array.
        self._terrain_origins = ProxyArray(wp.from_torch(origins_torch, dtype=wp.vec3f))
        # extract number of rows and cols
        num_rows, num_cols = self._terrain_origins.shape[:2]
        # maximum initial level possible for the terrains
        if self.cfg.max_init_terrain_level is None:
            max_init_level = num_rows - 1
        else:
            max_init_level = min(self.cfg.max_init_terrain_level, num_rows - 1)
        # store maximum terrain level possible
        self.max_terrain_level = num_rows
        # define all terrain levels and types available
        terrain_levels = torch.randint(0, max_init_level + 1, (num_envs,), device=self.device)
        terrain_types = torch.div(
            torch.arange(num_envs, device=self.device), (num_envs / num_cols), rounding_mode="floor"
        ).to(torch.long)
        env_origins = self._terrain_origins.torch[terrain_levels, terrain_types].contiguous()

        self._terrain_levels = ProxyArray(wp.from_torch(terrain_levels, dtype=wp.int64))
        self._terrain_types = ProxyArray(wp.from_torch(terrain_types, dtype=wp.int64))
        self._env_origins = ProxyArray(wp.from_torch(env_origins, dtype=wp.vec3f))

    def _configure_env_origins_grid(self, num_envs: int, env_spacing: float) -> None:
        """Allocate and initialize origin buffers from grid spacing."""
        from isaaclab.cloner import grid_transforms

        env_origins, _ = grid_transforms(num_envs, env_spacing, device=self.device)
        self._terrain_origins = None
        self._terrain_levels = None
        self._terrain_types = None
        self._env_origins = ProxyArray(wp.from_torch(env_origins.contiguous(), dtype=wp.vec3f))

    """
    Deprecated.
    """

    @property
    def warp_meshes(self):
        """A dictionary containing the terrain's names and their warp meshes.

        .. deprecated:: v2.1.0
            The `warp_meshes` attribute is deprecated. It is no longer stored inside the class.
        """
        logger.warning(
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
        logger.warning(
            "The `meshes` attribute is deprecated. It is no longer stored inside the `TerrainImporter` class."
            " Returning an empty dictionary."
        )
        return {}
