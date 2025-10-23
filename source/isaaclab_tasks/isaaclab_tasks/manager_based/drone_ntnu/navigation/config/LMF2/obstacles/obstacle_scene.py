import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
from .obstacle_scene_cfg import ObstaclesSceneCfg

from isaaclab.assets import (
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)

OBSTACLE_SCENE_CFG = ObstaclesSceneCfg(
                    env_size=(12.0, 8.0, 6.0),
                    max_num_obstacles=40,
                    ground_offset=3.0,
                    )

def generate_obstacle_collection(cfg: ObstaclesSceneCfg) -> RigidObjectCollectionCfg:
    max_num_obstacles = cfg.max_num_obstacles
    
    rigid_objects = {}
    
    for wall_name, wall_cfg in cfg.wall_cfgs.items():
        # Walls get their specific size and default center
        default_center = [0.0, 0.0, 0.0]  # Will be set properly at reset
        
        p_path = f"{{ENV_REGEX_NS}}/obstacle_{wall_name}"

        rigid_objects[wall_name] = RigidObjectCfg(
            prim_path=p_path,
            spawn=sim_utils.CuboidCfg(
                size=wall_cfg.size,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.5, 0.5, 0.5), metallic=0.2
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                    disable_gravity=True,
                    kinematic_enabled=False,
                    linear_damping=999.0,
                    angular_damping=999.0,
                    max_linear_velocity=0.0,  
                    max_angular_velocity=0.0, 
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1000000.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(default_center)),
        )
    
    obstacle_types = list(cfg.obstacle_cfgs.values())
    for i in range(max_num_obstacles):
        obj_name = f"obstacle_{i}"
        obs_cfg = obstacle_types[i % len(obstacle_types)]
        
        default_center = [0.0, 0.0, 0.0]
        color = np.random.randint(0, 256, size=3, dtype=np.uint8)
        color_normalized = tuple(float(c) / 255.0 for c in color)
        
        p_path = f"{{ENV_REGEX_NS}}/{obj_name}"
        
        rigid_objects[obj_name] = RigidObjectCfg(
            prim_path=p_path,
            spawn=sim_utils.CuboidCfg(
                size=obs_cfg.size,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color_normalized, metallic=0.2
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                    disable_gravity=True,
                    kinematic_enabled=False,
                    linear_damping=999.0,
                    angular_damping=999.0,
                    max_linear_velocity=0.0, 
                    max_angular_velocity=0.0, 
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1000000.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(default_center)),
        )

    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


def reset_obstacles_with_individual_ranges(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    obstacle_configs: dict,  
    wall_configs: dict, 
    env_size: tuple[float, float, float],
    use_curriculum: bool = True,
    max_num_obstacles: int = 10,
    ground_offset: float = 0.1,
) -> None:
    """Reset walls and obstacles with curriculum-based density control."""
    obstacles: RigidObjectCollection = env.scene[asset_cfg.name]
    
    num_objects = obstacles.num_objects
    num_envs = len(env_ids)
    
    object_names = obstacles.object_names
    
    # Get difficulty levels per environment
    if use_curriculum and hasattr(env, "_obstacle_difficulty_levels"):
        difficulty_levels = env._obstacle_difficulty_levels[env_ids]
        max_difficulty = env._max_obstacle_difficulty
    else:
        difficulty_levels = torch.ones(num_envs, device=env.device) * max_num_obstacles
        max_difficulty = max_num_obstacles
        
    # Calculate active obstacles per env based on difficulty
    min_obstacles = 1
    obstacles_per_env = (
        min_obstacles + 
        (difficulty_levels / max_difficulty) * (max_num_obstacles - min_obstacles)
    ).long()
    
    # Prepare tensors
    all_poses = torch.zeros(num_envs, num_objects, 7, device=env.device)
    all_velocities = torch.zeros(num_envs, num_objects, 6, device=env.device)
    
    # Extract wall names and obstacle types
    wall_names = list(wall_configs.keys())
    obstacle_types = list(obstacle_configs.values())
    
    for i, env_id in enumerate(env_ids):
        # Place walls (always active)
        for wall_name, wall_cfg in wall_configs.items():
            if wall_name in object_names:
                wall_idx = object_names.index(wall_name)
                
                min_ratio = torch.tensor(wall_cfg.center_ratio_min, device=env.device)
                max_ratio = torch.tensor(wall_cfg.center_ratio_max, device=env.device)
                env_size_t = torch.tensor(env_size, device=env.device)

                if torch.allclose(min_ratio, max_ratio):
                    center_ratio = min_ratio
                else:
                    ratios = torch.rand(3, device=env.device)
                    center_ratio = ratios * (max_ratio - min_ratio) + min_ratio

                # Position relative to env bounds, not env center
                # Walls should be at boundaries: ratio 0.0 = -env_size/2, ratio 1.0 = +env_size/2
                position = (center_ratio - 0.5) * env_size_t
                position[2] += ground_offset  # Add offset to z-position
                position += env.scene.env_origins[env_id]
                
                all_poses[i, wall_idx, 0:3] = position
                all_poses[i, wall_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
            
        # Place active obstacles
        num_active = obstacles_per_env[i].item()
        
        # Get non-wall obstacle indices
        obstacle_indices = [
            idx for idx, name in enumerate(object_names) 
            if name not in wall_names
        ]
        
        # Randomly pick which obstacles to activate
        if len(obstacle_indices) > 0:
            active_indices = torch.randperm(len(obstacle_indices), device=env.device)[:num_active].tolist()
            
            for active_slot, obj_list_idx in enumerate(active_indices):
                obj_idx = obstacle_indices[obj_list_idx]
                
                # Use the obstacle's pre-assigned type (round-robin from spawn)
                config_idx = obj_list_idx % len(obstacle_types)
                obs_cfg = obstacle_types[config_idx]
                
                # Sample position
                min_ratio = torch.tensor(obs_cfg.center_ratio_min, device=env.device)
                max_ratio = torch.tensor(obs_cfg.center_ratio_max, device=env.device)
                env_size_t = torch.tensor(env_size, device=env.device)
                
                ratios = torch.rand(3, device=env.device)
                position = (ratios * (max_ratio - min_ratio) + min_ratio - 0.5) * env_size_t
                position[2] += ground_offset  # Add offset to z-position
                position += env.scene.env_origins[env_id]
                
                quat = math_utils.random_orientation(1, device=env.device).squeeze(0)
                
                all_poses[i, obj_idx, 0:3] = position
                all_poses[i, obj_idx, 3:7] = quat
            
            # Move inactive obstacles far away
            inactive_indices = [
                obstacle_indices[j] for j in range(len(obstacle_indices))
                if j not in active_indices
            ]
            
            for obj_idx in inactive_indices:
                all_poses[i, obj_idx, 0:3] = env.scene.env_origins[env_id] + torch.tensor([0.0, 0.0, -1000.0], device=env.device)
                all_poses[i, obj_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    
    # Write to sim
    obstacles.write_object_pose_to_sim(all_poses, env_ids=env_ids)
    obstacles.write_object_velocity_to_sim(all_velocities, env_ids=env_ids)
