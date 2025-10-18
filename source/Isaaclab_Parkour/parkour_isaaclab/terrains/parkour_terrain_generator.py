import numpy as np
import trimesh

import omni.log
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.terrains.terrain_generator import TerrainGenerator
from .parkour_terrain_generator_cfg import ParkourTerrainGeneratorCfg, ParkourSubTerrainBaseCfg

class ParkourTerrainGenerator(TerrainGenerator):
    def __init__(self, cfg: ParkourTerrainGeneratorCfg, device: str = "cpu"):
        self.num_goals = cfg.num_goals 
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols)) # If num_rows = 1 and num_cols = 3 → you get a 1×3 grid of terrains.
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, self.num_goals, 3))
        self.terrain_names = np.zeros((cfg.num_rows, cfg.num_cols, 1)).astype(str) 
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale) + 1
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale) + 1 # The width (along x) and length (along y) of each sub-terrain (in m).
        self.total_width_pixels = width_pixels * cfg.num_rows
        self.total_length_pixels = length_pixels * cfg.num_cols
        self.goal_heights = np.zeros((cfg.num_rows, cfg.num_cols, self.num_goals), dtype=np.int16) 
        self.x_edge_maskes = np.zeros((cfg.num_rows, cfg.num_cols, width_pixels, length_pixels), dtype=np.int16)

        super().__init__(cfg=cfg, device=device)
        self.cfg:ParkourTerrainGeneratorCfg

    def _generate_random_terrains(self):
        """Add terrains based on randomly sampled difficulty parameter."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions) # the generator will pick the proportions with probability
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())
        sub_terrains_names = list(self.cfg.sub_terrains.keys())
        # randomly sample sub-terrains
        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            # coordinate index of the sub-terrain
            (sub_row, sub_col) = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            # randomly sample terrain index
            sub_index = self.np_rng.choice(len(proportions), p=proportions)
            # randomly sample difficulty parameter
            difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
            # generate terrain
            sub_terrains_name = sub_terrains_names[sub_index]
            self.terrain_type[sub_row, sub_col] = sub_col
            sub_terrains_cfg = sub_terrains_cfgs[sub_index]
            mesh, origin, sub_terrain_goal, goal_heights, x_edge_mask = self._get_terrain_mesh(difficulty, sub_terrains_cfg)
            # add to sub-terrains
            self.terrain_names[sub_row, sub_col] = sub_terrains_name
            self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrain_goal)
            self.goal_heights[sub_row, sub_col, :] = goal_heights
            self.x_edge_maskes[sub_row, sub_col,: ,:] = x_edge_mask 

    def _generate_curriculum_terrains(self):
        """Add terrains based on the difficulty parameter."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        sub_indices = []
        for index in range(self.cfg.num_cols):
            sub_index = np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
            sub_indices.append(sub_index) # difficulty increases along the columns
        sub_indices = np.array(sub_indices, dtype=np.int32)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())
        sub_terrains_names = list(self.cfg.sub_terrains.keys())
        # curriculum-based sub-terrains
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                lower, upper = self.cfg.difficulty_range
                if self.cfg.random_difficulty:
                    difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                else:
                    difficulty = sub_row / (self.cfg.num_rows-1)

                difficulty = lower + (upper - lower) * difficulty
                # generate terrain
                sub_terrains_cfg = sub_terrains_cfgs[sub_indices[sub_col]]
                sub_terrains_name = sub_terrains_names[sub_indices[sub_col]]
                mesh, origin, sub_terrain_goal, goal_heights, x_edge_mask = self._get_terrain_mesh(difficulty, sub_terrains_cfg)
                # add to sub-terrains
                self.terrain_type[sub_row, sub_col] = sub_indices[sub_col] # all of rows have the same terrain type
                self.terrain_names[sub_row, sub_col] = sub_terrains_name # all of rows have the same terrain name
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrain_goal) 
                self.goal_heights[sub_row, sub_col, :] = goal_heights
                self.x_edge_maskes[sub_row, sub_col,: ,:] = x_edge_mask

    def _get_terrain_mesh(
        self, 
        difficulty: float, 
        cfg: ParkourSubTerrainBaseCfg,
        ) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
        # copy the configuration
        cfg:ParkourSubTerrainBaseCfg = cfg.copy()
        # add other parameters to the sub-terrain configuration
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        # generate hash for the sub-terrain
        # generate the terrain
        meshes, origin, goals, goal_heights, x_edge_mask = cfg.function(difficulty, cfg, self.num_goals)
        mesh = trimesh.util.concatenate(meshes)
        # offset mesh such that they are in their center
        transform = np.eye(4)
        transform[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        # change origin to be in the center of the sub-terrain
        origin += transform[0:3, -1]

        # if caching is enabled, save the mesh and origin

        return mesh, origin, goals, goal_heights, x_edge_mask
    
    def _add_terrain_border(self):
        """Add a surrounding border over all the sub-terrains into the terrain meshes."""
        # border parameters
        border_size = (
            self.cfg.num_rows * self.cfg.size[0] + 2 * self.cfg.border_width,
            self.cfg.num_cols * self.cfg.size[1] + 2 * self.cfg.border_width,
        )
        inner_size = (self.cfg.num_rows * self.cfg.size[0], self.cfg.num_cols * self.cfg.size[1])
        border_center = (
            self.cfg.num_rows * self.cfg.size[0] / 2,
            self.cfg.num_cols * self.cfg.size[1] / 2,
            -self.cfg.border_height / 2,
        )
        # border mesh
        border_meshes = make_border(border_size, inner_size, height=self.cfg.border_height, position=border_center)
        border = trimesh.util.concatenate(border_meshes)
        # update the faces to have minimal triangles
        selector = ~(np.asarray(border.triangles)[:, :, 2] < -0.1).any(1)
        border.update_faces(selector)
        # add the border to the list of meshes
        self.terrain_meshes.append(border)

    def _add_sub_terrain(
        self, 
        mesh: trimesh.Trimesh, 
        origin: np.ndarray, 
        row: int, 
        col: int, 
        sub_terrain_goal: np.ndarray, 
    ):
        # transform the mesh to the correct position
        transform = np.eye(4)
        transform[0:2, -1] = (row + 0.5) * self.cfg.size[0], (col + 0.5) * self.cfg.size[1]
        mesh.apply_transform(transform)
        # add mesh to the list
        self.terrain_meshes.append(mesh)
        # add origin to the list
        self.terrain_origins[row, col] = origin + transform[:3, -1]
        self.goals[row, col, :, :2] = sub_terrain_goal 
