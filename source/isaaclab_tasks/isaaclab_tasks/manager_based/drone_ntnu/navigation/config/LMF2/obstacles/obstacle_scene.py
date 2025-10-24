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
                    linear_damping=999.0,
                    angular_damping=999.0,
                    max_linear_velocity=0.0,  
                    max_angular_velocity=0.0, 
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1000000.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(default_center)),
            collision_group=-1
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
                    linear_damping=999.0,
                    angular_damping=999.0,
                    max_linear_velocity=0.0, 
                    max_angular_velocity=0.0, 
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1000000.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(default_center)),
            collision_group=-1
        )

    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


# def check_aabb_collision(pos1, size1, pos2, size2, margin):
#     """Check if two axis-aligned bounding boxes collide.
    
#     Args:
#         pos1, pos2: Center positions (x, y, z)
#         size1, size2: Box sizes (width, depth, height)
#         margin: Safety margin to add
#     """
#     # Half extents for each box
#     half1 = size1 / 2.0 + margin
#     half2 = size2 / 2.0 + margin
    
#     # Check overlap on each axis
#     overlap_x = abs(pos1[0] - pos2[0]) < (half1[0] + half2[0])
#     overlap_y = abs(pos1[1] - pos2[1]) < (half1[1] + half2[1])
#     overlap_z = abs(pos1[2] - pos2[2]) < (half1[2] + half2[2])
    
#     # Collision if overlap on ALL axes
#     return overlap_x and overlap_y and overlap_z


# def reset_obstacles_with_individual_ranges(
#     env: ManagerBasedRLEnv,
#     env_ids: torch.Tensor,
#     asset_cfg: SceneEntityCfg,
#     obstacle_configs: dict,  
#     wall_configs: dict, 
#     env_size: tuple[float, float, float],
#     use_curriculum: bool = True,
#     max_num_obstacles: int = 10,
#     ground_offset: float = 0.1,
#     min_obstacle_separation: float = 0.3,
#     max_placement_attempts: int = 100,
# ) -> None:
#     """Reset walls and obstacles with curriculum-based density control and collision avoidance."""
#     obstacles: RigidObjectCollection = env.scene[asset_cfg.name]
    
#     num_objects = obstacles.num_objects
#     num_envs = len(env_ids)
#     object_names = obstacles.object_names
    
#     # Get difficulty levels
#     if use_curriculum and hasattr(env, "_obstacle_difficulty_levels"):
#         difficulty_levels = env._obstacle_difficulty_levels[env_ids]
#         max_difficulty = env._max_obstacle_difficulty
#     else:
#         difficulty_levels = torch.ones(num_envs, device=env.device) * max_num_obstacles
#         max_difficulty = max_num_obstacles
        
#     # Calculate active obstacles per env
#     min_obstacles = 1
#     obstacles_per_env = (
#         min_obstacles + 
#         (difficulty_levels / max_difficulty) * (max_num_obstacles - min_obstacles)
#     ).long()
    
#     # Prepare tensors
#     all_poses = torch.zeros(num_envs, num_objects, 7, device=env.device)
#     all_velocities = torch.zeros(num_envs, num_objects, 6, device=env.device)
    
#     wall_names = list(wall_configs.keys())
#     obstacle_types = list(obstacle_configs.values())
    
#     for i, env_id in enumerate(env_ids):
#         # Track placed objects (position, size) for AABB collision
#         placed_objects = []  # List of (position_tensor, size_tensor)
        
#         # Place walls
#         for wall_name, wall_cfg in wall_configs.items():
#             if wall_name in object_names:
#                 wall_idx = object_names.index(wall_name)
                
#                 min_ratio = torch.tensor(wall_cfg.center_ratio_min, device=env.device)
#                 max_ratio = torch.tensor(wall_cfg.center_ratio_max, device=env.device)
#                 env_size_t = torch.tensor(env_size, device=env.device)

#                 if torch.allclose(min_ratio, max_ratio):
#                     center_ratio = min_ratio
#                 else:
#                     ratios = torch.rand(3, device=env.device)
#                     center_ratio = ratios * (max_ratio - min_ratio) + min_ratio

#                 position = (center_ratio - 0.5) * env_size_t
#                 position[2] += ground_offset
#                 position += env.scene.env_origins[env_id]
                
#                 all_poses[i, wall_idx, 0:3] = position
#                 all_poses[i, wall_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
                
#                 # Track wall with actual size for AABB collision
#                 placed_objects.append((position.clone(), torch.tensor(wall_cfg.size, device=env.device)))
        
#         # Place active obstacles
#         num_active = obstacles_per_env[i].item()
        
#         obstacle_indices = [
#             idx for idx, name in enumerate(object_names) 
#             if name not in wall_names
#         ]
        
#         if len(obstacle_indices) > 0:
#             active_indices = torch.randperm(len(obstacle_indices), device=env.device)[:num_active].tolist()
            
#             for active_slot, obj_list_idx in enumerate(active_indices):
#                 obj_idx = obstacle_indices[obj_list_idx]
                
#                 config_idx = obj_list_idx % len(obstacle_types)
#                 obs_cfg = obstacle_types[config_idx]
#                 obs_size = torch.tensor(obs_cfg.size, device=env.device)
                
#                 min_ratio = torch.tensor(obs_cfg.center_ratio_min, device=env.device)
#                 max_ratio = torch.tensor(obs_cfg.center_ratio_max, device=env.device)
#                 env_size_t = torch.tensor(env_size, device=env.device)
                
#                 # Try to find collision-free position
#                 position_found = False
#                 for attempt in range(max_placement_attempts):
#                     ratios = torch.rand(3, device=env.device)
#                     position = (ratios * (max_ratio - min_ratio) + min_ratio - 0.5) * env_size_t
#                     position[2] += ground_offset
#                     position += env.scene.env_origins[env_id]
                    
#                     # Check AABB collision with all placed objects
#                     collision_detected = False
#                     for placed_pos, placed_size in placed_objects:
#                         if check_aabb_collision(position, obs_size, placed_pos, placed_size, min_obstacle_separation):
#                             collision_detected = True
#                             break
                    
#                     if not collision_detected:
#                         position_found = True
#                         break
                
#                 if position_found:
#                     quat = math_utils.random_orientation(1, device=env.device).squeeze(0)
#                     all_poses[i, obj_idx, 0:3] = position
#                     all_poses[i, obj_idx, 3:7] = quat
                    
#                     # Track with actual size
#                     placed_objects.append((position.clone(), obs_size.clone()))
#                 else:
#                     # Move far away
#                     all_poses[i, obj_idx, 0:3] = env.scene.env_origins[env_id] + torch.tensor([0.0, 0.0, -1000.0], device=env.device)
#                     all_poses[i, obj_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
            
#             # Move inactive obstacles far away
#             inactive_indices = [
#                 obstacle_indices[j] for j in range(len(obstacle_indices))
#                 if j not in active_indices
#             ]
            
#             for obj_idx in inactive_indices:
#                 all_poses[i, obj_idx, 0:3] = env.scene.env_origins[env_id] + torch.tensor([0.0, 0.0, -1000.0], device=env.device)
#                 all_poses[i, obj_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    
#     # Write to sim
#     obstacles.write_object_pose_to_sim(all_poses, env_ids=env_ids)
#     obstacles.write_object_velocity_to_sim(all_velocities, env_ids=env_ids)
    
    # if "depth_camera" in env.scene.sensors:
    #     env.scene.sensors["depth_camera"].reset(env_ids)
    
def check_box_collision(pos1: torch.Tensor, size1: torch.Tensor, 
                        pos2: torch.Tensor, size2: torch.Tensor, 
                        margin: float) -> bool:
    """Check if two axis-aligned bounding boxes collide.
    
    Args:
        pos1, pos2: Center positions (3D tensors)
        size1, size2: Box dimensions (3D tensors)
        margin: Safety margin to add
    
    Returns:
        True if boxes collide (overlap), False otherwise
    """
    half1 = size1 / 2.0 + margin
    half2 = size2 / 2.0 + margin
    
    overlap_x = abs(pos1[0] - pos2[0]) < (half1[0] + half2[0])
    overlap_y = abs(pos1[1] - pos2[1]) < (half1[1] + half2[1])
    overlap_z = abs(pos1[2] - pos2[2]) < (half1[2] + half2[2])
    
    return overlap_x and overlap_y and overlap_z


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
    min_obstacle_separation: float = 0.3,
    max_placement_attempts: int = 100,
) -> None:
    """Reset walls and obstacles with fully vectorized collision avoidance."""
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
    
    wall_names = list(wall_configs.keys())
    obstacle_types = list(obstacle_configs.values())
    env_size_t = torch.tensor(env_size, device=env.device)

    for wall_name, wall_cfg in wall_configs.items():
        if wall_name in object_names:
            wall_idx = object_names.index(wall_name)
            
            min_ratio = torch.tensor(wall_cfg.center_ratio_min, device=env.device)
            max_ratio = torch.tensor(wall_cfg.center_ratio_max, device=env.device)
            
            if torch.allclose(min_ratio, max_ratio):
                # Fixed position for all envs
                center_ratios = min_ratio.unsqueeze(0).repeat(num_envs, 1)
            else:
                # Random position per env
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
    
    # Track placed obstacles per environment
    max_possible_placed = len(wall_configs) + max_num_obstacles
    placed_positions = torch.zeros(num_envs, max_possible_placed, 3, device=env.device)
    placed_sizes = torch.zeros(num_envs, max_possible_placed, 3, device=env.device)
    placed_count = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    
    # Add walls to placed list for all envs
    for wall_name, wall_cfg in wall_configs.items():
        if wall_name in object_names:
            wall_idx = object_names.index(wall_name)
            wall_size = torch.tensor(wall_cfg.size, device=env.device)
            
            # Vectorized: add wall for all envs at once
            wall_positions_for_envs = all_poses[:, wall_idx, 0:3]
            current_counts = placed_count.clone()
            
            for env_idx in range(num_envs):
                placed_positions[env_idx, current_counts[env_idx]] = wall_positions_for_envs[env_idx]
                placed_sizes[env_idx, current_counts[env_idx]] = wall_size
            
            placed_count += 1
            
    # Create mask: (num_envs, num_obstacles) - which obstacles are active per env
    active_masks = torch.zeros(num_envs, len(obstacle_indices), dtype=torch.bool, device=env.device)
    for env_idx in range(num_envs):
        num_active = obstacles_per_env[env_idx].item()
        perm = torch.randperm(len(obstacle_indices), device=env.device)[:num_active]
        active_masks[env_idx, perm] = True

    for obj_list_idx in range(len(obstacle_indices)):
        obj_idx = obstacle_indices[obj_list_idx]
        
        # Which envs need this obstacle?
        envs_need_obstacle = active_masks[:, obj_list_idx]
        num_envs_active = envs_need_obstacle.sum().item()
        
        if num_envs_active == 0:
            # No env needs this obstacle - move all far away
            all_poses[:, obj_idx, 0:3] = (
                env.scene.env_origins[env_ids] + 
                torch.tensor([0.0, 0.0, -1000.0], device=env.device)
            )
            all_poses[:, obj_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
            continue
        
        # Get obstacle config
        config_idx = obj_list_idx % len(obstacle_types)
        obs_cfg = obstacle_types[config_idx]
        obs_size = torch.tensor(obs_cfg.size, device=env.device)
        
        min_ratio = torch.tensor(obs_cfg.center_ratio_min, device=env.device)
        max_ratio = torch.tensor(obs_cfg.center_ratio_max, device=env.device)
        
        # Track which envs found valid positions
        position_found = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        final_positions = torch.zeros(num_envs, 3, device=env.device)
        
        # Try to find positions for all envs in parallel
        for attempt in range(max_placement_attempts):
            # Only sample for envs that need this obstacle and haven't found a position yet
            needs_position = envs_need_obstacle & ~position_found
            num_to_sample = needs_position.sum().item()
            
            if num_to_sample == 0:
                break  # All active envs found positions
            
            # Sample positions for envs that need them
            ratios = torch.rand(num_to_sample, 3, device=env.device)
            positions = (ratios * (max_ratio - min_ratio) + min_ratio - 0.5) * env_size_t
            positions[:, 2] += ground_offset
            
            # Add env origins
            env_indices_needing = torch.where(needs_position)[0]
            positions += env.scene.env_origins[env_ids[env_indices_needing]]
            
            # Check collisions for each sampled position
            for pos_idx, env_idx in enumerate(env_indices_needing):
                position = positions[pos_idx]
                collision = False
                
                # Check against all placed objects in this env
                for placed_idx in range(placed_count[env_idx]):
                    placed_pos = placed_positions[env_idx, placed_idx]
                    placed_size = placed_sizes[env_idx, placed_idx]
                    
                    if check_box_collision(position, obs_size, placed_pos, placed_size, min_obstacle_separation):
                        collision = True
                        break
                
                if not collision:
                    position_found[env_idx] = True
                    final_positions[env_idx] = position
        
        # Generate quaternions for all envs
        quats = math_utils.random_orientation(num_envs, device=env.device)
        
        # Apply positions where found (vectorized boolean indexing)
        success_mask = envs_need_obstacle & position_found
        all_poses[success_mask, obj_idx, 0:3] = final_positions[success_mask]
        all_poses[success_mask, obj_idx, 3:7] = quats[success_mask]
        
        # Update placed tracking (only sequential part left)
        for env_idx in torch.where(success_mask)[0]:
            placed_positions[env_idx, placed_count[env_idx]] = final_positions[env_idx]
            placed_sizes[env_idx, placed_count[env_idx]] = obs_size
            placed_count[env_idx] += 1
        
        # Move failed/inactive far away (vectorized)
        failed_or_inactive = ~success_mask
        all_poses[failed_or_inactive, obj_idx, 0:3] = (
            env.scene.env_origins[env_ids[failed_or_inactive]] + 
            torch.tensor([0.0, 0.0, -1000.0], device=env.device)
        )
        all_poses[failed_or_inactive, obj_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    
    # Write to sim
    obstacles.write_object_pose_to_sim(all_poses, env_ids=env_ids)
    obstacles.write_object_velocity_to_sim(all_velocities, env_ids=env_ids)
