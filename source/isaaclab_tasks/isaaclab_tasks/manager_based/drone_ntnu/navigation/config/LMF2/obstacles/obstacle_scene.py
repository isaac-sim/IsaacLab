import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
from .obstacle_scene_cfg import ObstaclesSceneCfg

from isaaclab.assets import (
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)

OBSTACLE_SCENE_CFG = ObstaclesSceneCfg(
                    env_size=(12.0, 8.0, 6.0),
                    min_num_obstacles=20,
                    max_num_obstacles=40,
                    ground_offset=3.0,
                    )

def generate_obstacle_collection(cfg: ObstaclesSceneCfg) -> RigidObjectCollectionCfg:
    max_num_obstacles = cfg.max_num_obstacles
    
    rigid_objects = {}
    
    for wall_name, wall_cfg in cfg.wall_cfgs.items():
        # Walls get their specific size and default center
        default_center = [0.0, 0.0, 0.0]  # Will be set properly at reset

        rigid_objects[wall_name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/obstacle_{wall_name}",
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
                    linear_damping=9999.0,
                    angular_damping=9999.0,
                    max_linear_velocity=0.0,  
                    max_angular_velocity=0.0, 
                ),
                # mass of walls needs to be way larger than weight of obstacles to make them not move during reset
                mass_props=sim_utils.MassPropertiesCfg(mass=10000000.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(default_center)),
            collision_group=0
        )
    
    obstacle_types = list(cfg.obstacle_cfgs.values())
    for i in range(max_num_obstacles):
        obj_name = f"obstacle_{i}"
        obs_cfg = obstacle_types[i % len(obstacle_types)]
        
        default_center = [0.0, 0.0, 0.0]
        color = np.random.randint(0, 256, size=3, dtype=np.uint8)
        color_normalized = tuple(float(c) / 255.0 for c in color)
        
        rigid_objects[obj_name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{obj_name}",
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
                    linear_damping=1.0,
                    angular_damping=1.0,
                    max_linear_velocity=0.0, 
                    max_angular_velocity=0.0, 
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(default_center)),
            collision_group=0
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
    min_num_obstacles: int = 1,
    max_num_obstacles: int = 10,
    ground_offset: float = 0.1,
) -> None:
    """Reset walls and obstacles without collision checking (fast)."""
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
    obstacles_per_env = (
        min_num_obstacles + 
        (difficulty_levels / max_difficulty) * (max_num_obstacles - min_num_obstacles)
    ).long()
    
    # Prepare tensors
    all_poses = torch.zeros(num_envs, num_objects, 7, device=env.device)
    all_velocities = torch.zeros(num_envs, num_objects, 6, device=env.device)
    
    wall_names = list(wall_configs.keys())
    obstacle_types = list(obstacle_configs.values())
    env_size_t = torch.tensor(env_size, device=env.device)
    
    # place walls
    for wall_name, wall_cfg in wall_configs.items():
        if wall_name in object_names:
            wall_idx = object_names.index(wall_name)
            
            min_ratio = torch.tensor(wall_cfg.center_ratio_min, device=env.device)
            max_ratio = torch.tensor(wall_cfg.center_ratio_max, device=env.device)
            
            if torch.allclose(min_ratio, max_ratio):
                center_ratios = min_ratio.unsqueeze(0).repeat(num_envs, 1)
            else:
                ratios = torch.rand(num_envs, 3, device=env.device)
                center_ratios = ratios * (max_ratio - min_ratio) + min_ratio
            
            positions = (center_ratios - 0.5) * env_size_t
            positions[:, 2] += ground_offset
            positions += env.scene.env_origins[env_ids]
            
            all_poses[:, wall_idx, 0:3] = positions
            all_poses[:, wall_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(num_envs, 1)
    
    # Get obstacle indices
    obstacle_indices = [
        idx for idx, name in enumerate(object_names) 
        if name not in wall_names
    ]
    
    if len(obstacle_indices) == 0:
        obstacles.write_object_pose_to_sim(all_poses, env_ids=env_ids)
        obstacles.write_object_velocity_to_sim(all_velocities, env_ids=env_ids)
        return
    
    # Determine which obstacles are active per env
    active_masks = torch.zeros(num_envs, len(obstacle_indices), dtype=torch.bool, device=env.device)
    for env_idx in range(num_envs):
        num_active = obstacles_per_env[env_idx].item()
        perm = torch.randperm(len(obstacle_indices), device=env.device)[:num_active]
        active_masks[env_idx, perm] = True
    
    # place obstacles
    for obj_list_idx in range(len(obstacle_indices)):
        obj_idx = obstacle_indices[obj_list_idx]
        
        # Which envs need this obstacle?
        envs_need_obstacle = active_masks[:, obj_list_idx]
        
        if not envs_need_obstacle.any():
            # Move all to -1000
            all_poses[:, obj_idx, 0:3] = (
                env.scene.env_origins[env_ids] + 
                torch.tensor([0.0, 0.0, -1000.0], device=env.device)
            )
            all_poses[:, obj_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
            continue
        
        # Get obstacle config
        config_idx = obj_list_idx % len(obstacle_types)
        obs_cfg = obstacle_types[config_idx]
        
        min_ratio = torch.tensor(obs_cfg.center_ratio_min, device=env.device)
        max_ratio = torch.tensor(obs_cfg.center_ratio_max, device=env.device)
        
        # sample obejct positions
        num_active_envs = envs_need_obstacle.sum().item()
        ratios = torch.rand(num_active_envs, 3, device=env.device)
        positions = (ratios * (max_ratio - min_ratio) + min_ratio - 0.5) * env_size_t
        positions[:, 2] += ground_offset
        
        # Add env origins
        active_env_indices = torch.where(envs_need_obstacle)[0]
        positions += env.scene.env_origins[env_ids[active_env_indices]]
        
        # Generate quaternions
        quats = math_utils.random_orientation(num_envs, device=env.device)
        
        # Write poses
        all_poses[envs_need_obstacle, obj_idx, 0:3] = positions
        all_poses[envs_need_obstacle, obj_idx, 3:7] = quats[envs_need_obstacle]
        
        # Move inactive obstacles far away
        inactive = ~envs_need_obstacle
        all_poses[inactive, obj_idx, 0:3] = (
            env.scene.env_origins[env_ids[inactive]] + 
            torch.tensor([0.0, 0.0, -1000.0], device=env.device)
        )
        all_poses[inactive, obj_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    
    # Write to sim
    obstacles.write_object_pose_to_sim(all_poses, env_ids=env_ids)
    obstacles.write_object_velocity_to_sim(all_velocities, env_ids=env_ids)