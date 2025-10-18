

from __future__ import annotations
from isaaclab.utils import configclass
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.height_field import HfTerrainBaseCfg

@configclass
class ParkourSubTerrainBaseCfg(HfTerrainBaseCfg):
    border_width: float = 0.0 # 
    horizontal_scale: float = 0.05
    """The discretization of the terrain along the x and y axes (in m). Defaults to 0.1."""
    vertical_scale: float = 0.005
    """The discretization of the terrain along the z axis (in m). Defaults to 0.005."""
    platform_len: float = 2.5
    platform_height: float = 0.
    slope_threshold: float | None = 1.5
    edge_width_thresh = 0.05
    use_simplified: bool = False
    
@configclass
class ParkourTerrainGeneratorCfg(TerrainGeneratorCfg):
    num_goals: int = 8 
    terrain_names: list[str] = [] 
    random_difficulty: bool = False 
    

