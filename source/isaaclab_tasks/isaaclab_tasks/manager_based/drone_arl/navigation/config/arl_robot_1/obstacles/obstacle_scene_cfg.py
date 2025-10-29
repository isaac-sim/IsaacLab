from isaaclab.utils import configclass
from dataclasses import MISSING

@configclass
class ObstaclesSceneCfg():
    """Configuration for a terrain with floating obstacles."""

    min_num_obstacles: int = 1
    max_num_obstacles: int = 40
    ground_offset: float = 3.0
    
    env_size: tuple[float, float, float] = MISSING
    @configclass
    class BoxCfg():
        """Configuration for a box-shaped obstacle or wall.
    
        Defines the size and placement constraints for rectangular obstacles within
        the environment. The center position is specified as ratios of the environment
        size, allowing for flexible scaling.
        
        Attributes:
            size: Tuple of (length, width, height) in meters.
            center_ratio_min: Minimum position as ratio of env_size (0.0 to 1.0) for
                each axis. Used for random placement bounds.
            center_ratio_max: Maximum position as ratio of env_size (0.0 to 1.0) for
                each axis. For fixed positions, set equal to center_ratio_min.
        """
        size: tuple[float, float, float] = MISSING
        center_ratio_min: tuple[float, float, float] = MISSING
        center_ratio_max: tuple[float, float, float] = MISSING

    # Obstacle configurations
    panel_obs_cfg = BoxCfg()
    panel_obs_cfg.size = (0.1, 1.2, 3.0)
    panel_obs_cfg.center_ratio_min = (0.3, 0.05, 0.05)
    panel_obs_cfg.center_ratio_max = (0.85, 0.95, 0.95)

    small_wall_obs_cfg  = BoxCfg()
    small_wall_obs_cfg.size = (0.1, 0.5, 0.5)
    small_wall_obs_cfg.center_ratio_min = (0.3, 0.05, 0.05)
    small_wall_obs_cfg.center_ratio_max = (0.85, 0.9, 0.9)

    big_wall_obs_cfg = BoxCfg()
    big_wall_obs_cfg.size = (0.1, 1.0, 1.0)
    big_wall_obs_cfg.center_ratio_min = (0.3, 0.05, 0.05)
    big_wall_obs_cfg.center_ratio_max = (0.85, 0.9, 0.9)

    small_cube_obs_cfg = BoxCfg()
    small_cube_obs_cfg.size = (0.4, 0.4, 0.4)
    small_cube_obs_cfg.center_ratio_min = (0.3, 0.05, 0.05)
    small_cube_obs_cfg.center_ratio_max = (0.85, 0.9, 0.9)

    rod_obs_cfg = BoxCfg()
    rod_obs_cfg.size = (0.1, 0.1, 2.0)
    rod_obs_cfg.center_ratio_min = (0.3, 0.05, 0.05)
    rod_obs_cfg.center_ratio_max = (0.85, 0.9, 0.9)
    
    
    # Wall configurations
    left_wall_cfg = BoxCfg()
    left_wall_cfg.size = (12.0, 0.2, 6.0)
    left_wall_cfg.center_ratio_min = (0.5, 1.0, 0.5)
    left_wall_cfg.center_ratio_max = (0.5, 1.0, 0.5)

    right_wall_cfg = BoxCfg()
    right_wall_cfg.size = (12.0, 0.2, 6.0)
    right_wall_cfg.center_ratio_min = (0.5, 0.0, 0.5)
    right_wall_cfg.center_ratio_max = (0.5, 0.0, 0.5)

    back_wall_cfg = BoxCfg()
    back_wall_cfg.size = (0.2, 8.0, 6.0)
    back_wall_cfg.center_ratio_min = (0.0, 0.5, 0.5)
    back_wall_cfg.center_ratio_max = (0.0, 0.5, 0.5)

    front_wall_cfg = BoxCfg()
    front_wall_cfg.size = (0.2, 8.0, 6.0)
    front_wall_cfg.center_ratio_min = (1.0, 0.5, 0.5)
    front_wall_cfg.center_ratio_max = (1.0, 0.5, 0.5)

    top_wall_cfg = BoxCfg()
    top_wall_cfg.size = (12.0, 8.0, 0.2)
    top_wall_cfg.center_ratio_min = (0.5, 0.5, 1.0)
    top_wall_cfg.center_ratio_max = (0.5, 0.5, 1.0)

    bottom_wall_cfg = BoxCfg()
    bottom_wall_cfg.size = (12.0, 8.0, 0.2)
    bottom_wall_cfg.center_ratio_min = (0.5, 0.5, 0.0)
    bottom_wall_cfg.center_ratio_max = (0.5, 0.5, 0.0)

    wall_cfgs = {
        "left_wall": left_wall_cfg,
        "right_wall": right_wall_cfg,
        "back_wall": back_wall_cfg,
        "front_wall": front_wall_cfg,
        "bottom_wall": bottom_wall_cfg,
        # "top_wall": top_wall_cfg,
    }

    obstacle_cfgs = {
        "panel": panel_obs_cfg,
        "small_wall": small_wall_obs_cfg,
        "big_wall": big_wall_obs_cfg,
        "small_cube": small_cube_obs_cfg,
        "rod": rod_obs_cfg,
    }
    
