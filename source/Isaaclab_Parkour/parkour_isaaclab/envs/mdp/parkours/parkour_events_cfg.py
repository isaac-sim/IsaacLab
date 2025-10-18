import math
from dataclasses import MISSING

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils

from parkour_isaaclab.managers import ParkourTermCfg
from .parkour_event import ParkourEvent

CURRENT_GOAL_MARKER_CFG = VisualizationMarkersCfg(
    markers = {
        'frame': sim_utils.SphereCfg(            
            radius=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    }
)

FUTURE_GOAL_MARKER_CFG = VisualizationMarkersCfg(
    markers = {
        'frame': sim_utils.SphereCfg(            
            radius=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)

CURRENT_ARROW_MARKER_CFG = VisualizationMarkersCfg(
    markers = {
        'frame': sim_utils.SphereCfg(            
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    }
)

FUTURE_ARROWS_MARKER_CFG = VisualizationMarkersCfg(
    markers = {
        'frame': sim_utils.SphereCfg(            
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)

@configclass
class ParkourEventsCfg(ParkourTermCfg):
    
    class_type: type = ParkourEvent

    asset_name: str = MISSING

    num_future_goal_obs: int = 2
    
    arrow_num: int = 8  

    reach_goal_delay: float = 0.1
    next_goal_threshold: float = 0.2

    future_goal_poses_visualizer_cfg: VisualizationMarkersCfg \
        = FUTURE_GOAL_MARKER_CFG.replace(prim_path="/Visuals/Command/future_goal_poses")

    current_goal_pose_visualizer_cfg: VisualizationMarkersCfg \
        = CURRENT_GOAL_MARKER_CFG.replace(prim_path="/Visuals/Command/current_goal_pose")

    future_goal_poses_visualizer_cfg.markers["frame"].scale = (1.1, 1.1, 1.1)
    current_goal_pose_visualizer_cfg.markers["frame"].scale = (1.1, 1.1, 1.1)

    future_arrow_visualizer_cfg: VisualizationMarkersCfg \
        = FUTURE_ARROWS_MARKER_CFG.replace(prim_path="/Visuals/Command/future_arrow")

    current_arrow_visualizer_cfg: VisualizationMarkersCfg \
        = CURRENT_ARROW_MARKER_CFG.replace(prim_path="/Visuals/Command/current_arrow")

    future_arrow_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_arrow_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
