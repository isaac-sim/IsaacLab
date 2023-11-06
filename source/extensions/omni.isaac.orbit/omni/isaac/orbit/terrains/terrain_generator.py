# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import os
import torch
import trimesh

from omni.isaac.orbit.utils.dict import dict_to_md5_hash
from omni.isaac.orbit.utils.io import dump_yaml
from omni.isaac.orbit.utils.timer import Timer

from .height_field import HfTerrainBaseCfg
from .terrain_generator_cfg import SubTerrainBaseCfg, TerrainGeneratorCfg
from .trimesh.utils import make_border
from .utils import color_meshes_by_height


class TerrainGenerator:
    """Terrain generator to handle different terrain generation functions.

    The terrains are represented as meshes. These are obtained either from height fields or by using the
    `trimesh <https://trimsh.org/trimesh.html>`__ library. The height field representation is more
    flexible, but it is less computationally and memory efficient than the trimesh representation.

    All terrain generation functions take in the argument :obj:`difficulty` which determines the complexity
    of the terrain. The difficulty is a number between 0 and 1, where 0 is the easiest and 1 is the hardest.
    In most cases, the difficulty is used for linear interpolation between different terrain parameters.
    For example, in a pyramid stairs terrain the step height is interpolated between the specified minimum
    and maximum step height.

    Each sub-terrain has a corresponding configuration class that can be used to specify the parameters
    of the terrain. The configuration classes are inherited from the :class:`SubTerrainBaseCfg` class
    which contains the common parameters for all terrains.

    If a curriculum is used, the terrains are generated based on their difficulty parameter.
    The difficulty is varied linearly over the number of rows (i.e. along x). If a curriculum
    is not used, the terrains are generated randomly.

    If the flag :obj:`cfg.use_cache` is set to True, the terrains are cached based on their
    sub-terrain configurations. This means that if the same sub-terrain configuration is used
    multiple times, the terrain is only generated once and then reused. This is useful when
    generating complex sub-terrains that take a long time to generate.
    """

    terrain_mesh: trimesh.Trimesh
    """A single trimesh.Trimesh object for all the generated sub-terrains."""
    terrain_meshes: list[trimesh.Trimesh]
    """List of trimesh.Trimesh objects for all the generated sub-terrains."""
    terrain_origins: np.ndarray
    """The origin of each sub-terrain. Shape is (num_rows, num_cols, 3)."""

    def __init__(self, cfg: TerrainGeneratorCfg):
        """Initialize the terrain generator.

        Args:
            cfg: Configuration for the terrain generator.
        """
        # check inputs
        if len(cfg.sub_terrains) == 0:
            raise ValueError("No sub-terrains specified! Please add at least one sub-terrain.")
        # store inputs
        self.cfg = cfg
        # set common values to all sub-terrains config
        for sub_cfg in self.cfg.sub_terrains.values():
            # size of all terrains
            sub_cfg.size = self.cfg.size
            # params for height field terrains
            if isinstance(sub_cfg, HfTerrainBaseCfg):
                sub_cfg.horizontal_scale = self.cfg.horizontal_scale
                sub_cfg.vertical_scale = self.cfg.vertical_scale
                sub_cfg.slope_threshold = self.cfg.slope_threshold

        # set the seed for reproducibility
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)
        # create a list of all sub-terrains
        self.terrain_meshes = list()
        self.terrain_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3))
        # parse configuration and add sub-terrains
        # create terrains based on curriculum or randomly
        if self.cfg.curriculum:
            with Timer("[INFO] Generating terrains based on curriculum took"):
                self._generate_curriculum_terrains()
        else:
            with Timer("[INFO] Generating terrains randomly took"):
                self._generate_random_terrains()
        # add a border around the terrains
        self._add_terrain_border()
        # combine all the sub-terrains into a single mesh
        self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)
        # color the terrain mesh
        if self.cfg.color_scheme == "height":
            self.terrain_mesh = color_meshes_by_height(self.terrain_mesh)
        elif self.cfg.color_scheme == "random":
            self.terrain_mesh.visual.vertex_colors = np.random.choice(
                range(256), size=(len(self.terrain_mesh.vertices), 4)
            )
        elif self.cfg.color_scheme == "none":
            pass
        else:
            raise ValueError(f"Invalid color scheme: {self.cfg.color_scheme}.")
        # offset the entire terrain and origins so that it is centered
        # -- terrain mesh
        transform = np.eye(4)
        transform[:2, -1] = -self.cfg.size[0] * self.cfg.num_rows * 0.5, -self.cfg.size[1] * self.cfg.num_cols * 0.5
        self.terrain_mesh.apply_transform(transform)
        # -- terrain origins
        self.terrain_origins += transform[:3, -1]

    """
    Terrain generator functions.
    """

    def _generate_random_terrains(self):
        """Add terrains based on randomly sampled difficulty parameter."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        # randomly sample sub-terrains
        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            # coordinate index of the sub-terrain
            (sub_row, sub_col) = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            # randomly sample terrain index
            sub_index = np.random.choice(len(proportions), p=proportions)
            # randomly sample difficulty parameter
            difficulty = np.random.choice(self.cfg.difficulty_choices)
            # generate terrain
            mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_index])
            # add to sub-terrains
            self._add_sub_terrain(mesh, origin, sub_row, sub_col)

    def _generate_curriculum_terrains(self):
        """Add terrains based on the difficulty parameter."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        # find the sub-terrain index for each column
        # we generate the terrains based on their proportion (not randomly sampled)
        sub_indices = []
        for index in range(self.cfg.num_cols):
            sub_index = np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
            sub_indices.append(sub_index)
        sub_indices = np.array(sub_indices, dtype=np.int32)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        # curriculum-based sub-terrains
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                # vary the difficulty parameter
                difficulty = sub_row / self.cfg.num_rows
                # generate terrain
                mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_indices[sub_col]])
                # add to sub-terrains
                self._add_sub_terrain(mesh, origin, sub_row, sub_col)

    """
    Internal helper functions.
    """

    def _add_terrain_border(self):
        """Add a surrounding border over all the sub-terrains into the terrain meshes."""
        # border parameters
        border_size = (
            self.cfg.num_rows * self.cfg.size[0] + 2 * self.cfg.border_width,
            self.cfg.num_cols * self.cfg.size[1] + 2 * self.cfg.border_width,
        )
        inner_size = (self.cfg.num_rows * self.cfg.size[0], self.cfg.num_cols * self.cfg.size[1])
        border_center = (self.cfg.num_rows * self.cfg.size[0] / 2, self.cfg.num_cols * self.cfg.size[1] / 2, -0.5)
        # border mesh
        border_meshes = make_border(border_size, inner_size, height=1.0, position=border_center)
        border = trimesh.util.concatenate(border_meshes)
        # update the faces to have minimal triangles
        selector = ~(np.asarray(border.triangles)[:, :, 2] < -0.1).any(1)
        border.update_faces(selector)
        # add the border to the list of meshes
        self.terrain_meshes.append(border)

    def _add_sub_terrain(self, mesh: trimesh.Trimesh, origin: np.ndarray, row: int, col: int):
        """Add input sub-terrain to the list of sub-terrains meshes and origins.

        Args:
            mesh: The mesh of the sub-terrain.
            origin: The origin of the sub-terrain.
            row: The row index of the sub-terrain.
            col: The column index of the sub-terrain.
        """
        transform = np.eye(4)
        transform[0:2, -1] = (row + 0.5) * self.cfg.size[0], (col + 0.5) * self.cfg.size[1]
        mesh.apply_transform(transform)
        # add mesh to the list
        self.terrain_meshes.append(mesh)
        # add origin to the list
        self.terrain_origins[row, col] = origin + transform[:3, -1]

    def _get_terrain_mesh(self, difficulty: float, cfg: SubTerrainBaseCfg) -> tuple[trimesh.Trimesh, np.ndarray]:
        """Generate a sub-terrain mesh based on the input difficulty parameter.

        If caching is enabled, the sub-terrain is cached and loaded from the cache if it exists.
        The cache is stored in the cache directory specified in the configuration.

        .. Note:
            This function centers the 2D center of the mesh and its specified origin such that the
            2D center becomes :math:`(0, 0)` instead of :math:`(size[0] / 2, size[1] / 2).

        Args:
            difficulty: The difficulty parameter.
            cfg: The configuration of the sub-terrain.

        Returns:
            The sub-terrain mesh and origin.
        """
        # add other parameters to the sub-terrain configuration
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        # generate hash for the sub-terrain
        sub_terrain_hash = dict_to_md5_hash(cfg.to_dict())
        # generate the file name
        sub_terrain_cache_dir = os.path.join(self.cfg.cache_dir, sub_terrain_hash)
        sub_terrain_stl_filename = os.path.join(sub_terrain_cache_dir, "mesh.stl")
        sub_terrain_csv_filename = os.path.join(sub_terrain_cache_dir, "origin.csv")
        sub_terrain_meta_filename = os.path.join(sub_terrain_cache_dir, "cfg.yaml")

        # check if hash exists - if true, load the mesh and origin and return
        if self.cfg.use_cache and os.path.exists(sub_terrain_stl_filename):
            # load existing mesh
            mesh = trimesh.load_mesh(sub_terrain_stl_filename)
            origin = np.loadtxt(sub_terrain_csv_filename, delimiter=",")
            # return the generated mesh
            return mesh, origin

        # generate the terrain
        meshes, origin = cfg.function(difficulty, cfg)
        mesh = trimesh.util.concatenate(meshes)
        # offset mesh such that they are in their center
        transform = np.eye(4)
        transform[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        # change origin to be in the center of the sub-terrain
        origin += transform[0:3, -1]

        # if caching is enabled, save the mesh and origin
        if self.cfg.use_cache:
            # create the cache directory
            os.makedirs(sub_terrain_cache_dir, exist_ok=True)
            # save the data
            mesh.export(sub_terrain_stl_filename)
            np.savetxt(sub_terrain_csv_filename, origin, delimiter=",", header="x,y,z")
            dump_yaml(sub_terrain_meta_filename, cfg)
        # return the generated mesh
        return mesh, origin
