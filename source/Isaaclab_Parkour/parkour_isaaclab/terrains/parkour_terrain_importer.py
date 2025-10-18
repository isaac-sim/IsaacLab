# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.log

import isaaclab.sim as sim_utils
from .parkour_terrain_generator import ParkourTerrainGenerator
from isaaclab.terrains.terrain_importer import TerrainImporter
import numpy as np   

if TYPE_CHECKING:
    from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg

class ParkourTerrainImporter(TerrainImporter):

    terrain_prim_paths: list[str]

    terrain_origins: torch.Tensor | None

    env_origins: torch.Tensor

    def __init__(self, cfg: TerrainImporterCfg):

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
            self._terrain_generator_class = ParkourTerrainGenerator(cfg=self.cfg.terrain_generator, 
                                                                    device=self.device,
                                                                    )
            self.import_mesh("terrain", self._terrain_generator_class.terrain_mesh)
            # configure the terrain origins based on the terrain generator
            self.configure_env_origins(self._terrain_generator_class.terrain_origins)
            # refer to the flat patches
            self._terrain_flat_patches = self._terrain_generator_class.flat_patches
        else:
            TypeError(f'Parkour Terrain type only support generator, not {self.cfg.terrain_type}')
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    @property
    def terrain_generator_class(self):
        return self._terrain_generator_class

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