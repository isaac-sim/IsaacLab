
from __future__ import annotations

import numpy as np
import random
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING
from ..utils import parkour_field_to_mesh
if TYPE_CHECKING:
    from . import extreme_parkour_terrains_cfg

"""
Reference from https://arxiv.org/pdf/2309.14341
"""

def padding_height_field_raw(
    height_field_raw:np.ndarray, 
    cfg:extreme_parkour_terrains_cfg.ExtremeParkourRoughTerrainCfg
    )->np.ndarray:
    pad_width = int(cfg.pad_width // cfg.horizontal_scale)
    pad_height = int(cfg.pad_height // cfg.vertical_scale)
    height_field_raw[:, :pad_width] = pad_height
    height_field_raw[:, -pad_width:] = pad_height
    height_field_raw[:pad_width, :] = pad_height
    height_field_raw[-pad_width:, :] = pad_height
    height_field_raw = np.rint(height_field_raw).astype(np.int16)
    return height_field_raw

def random_uniform_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourRoughTerrainCfg,
    height_field_raw: np.ndarray,
    ):
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    # -- height
    max_height = (cfg.noise_range[1] - cfg.noise_range[0]) * difficulty + cfg.noise_range[0]
    height_min = int(-cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(max_height / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)

    # create range of heights possible
    height_range = np.arange(height_min, height_max + height_step, height_step)
    # sample heights randomly from the range along a grid
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))
    # create interpolation function for the sampled heights
    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)
    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # round off the interpolated heights to the nearest vertical step
    z_upsampled = np.rint(z_upsampled).astype(np.int16)
    height_field_raw += z_upsampled 
    return height_field_raw 

@parkour_field_to_mesh
def parkour_gap_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourGapTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
        height_field_raw = np.zeros((width_pixels, length_pixels))
        mid_y = length_pixels // 2  # length is actually y width
        gap_size = eval(cfg.gap_size,{"difficulty":difficulty})
        gap_size = round(gap_size / cfg.horizontal_scale)

        dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale) + gap_size
        dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale) + gap_size

        dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
        dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)

        platform_len = round(cfg.platform_len / cfg.horizontal_scale)
        platform_height = round(cfg.platform_height / cfg.vertical_scale)
        height_field_raw[0:platform_len, :] = platform_height

        gap_depth = -round(np.random.uniform(cfg.gap_depth[0], cfg.gap_depth[1]) / cfg.vertical_scale)
        half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)
        goals = np.zeros((num_goals, 2))
        goal_heights = np.ones((num_goals)) * platform_height
        goals[0] = [platform_len - 1, mid_y]
        dis_x = platform_len
        last_dis_x = dis_x
        for i in range(num_goals - 2):
            rand_x = np.random.randint(dis_x_min, dis_x_max)
            dis_x += rand_x
            rand_y = np.random.randint(dis_y_min, dis_y_max)
            if not cfg.apply_flat:
                height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, :] = gap_depth

            height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = gap_depth
            height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = gap_depth
            
            last_dis_x = dis_x
            goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
        final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)

        if final_dis_x > width_pixels:
            final_dis_x = width_pixels - 0.5 // cfg.horizontal_scale
        goals[-1] = [final_dis_x, mid_y]
        height_field_raw = padding_height_field_raw(height_field_raw,cfg)
        if cfg.apply_roughness:
            height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
        return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale

@parkour_field_to_mesh
def parkour_hurdle_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourHurdleTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        stone_len = eval(cfg.stone_len, {"difficulty": difficulty})
        stone_len = round(stone_len / cfg.horizontal_scale)

        width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
        height_field_raw = np.zeros((width_pixels, length_pixels))

        mid_y = length_pixels // 2  # length is actually y width
        dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale)
        dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale) 
        dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
        dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)

        half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)
        hurdle_height_range = eval(cfg.hurdle_height_range, {"difficulty": difficulty})
        hurdle_height_max = round(hurdle_height_range[1] / cfg.vertical_scale)
        hurdle_height_min = round(hurdle_height_range[0] / cfg.vertical_scale)

        platform_len = round(cfg.platform_len / cfg.horizontal_scale)
        platform_height = round(cfg.platform_height / cfg.vertical_scale)
        height_field_raw[0:platform_len, :] = platform_height
        dis_x = platform_len
        goals = np.zeros((num_goals, 2))
        goal_heights = np.ones((num_goals)) * platform_height

        goals[0] = [platform_len - 1, mid_y]

        for i in range(num_goals-2):
            rand_x = np.random.randint(dis_x_min, dis_x_max)
            rand_y = np.random.randint(dis_y_min, dis_y_max)
            dis_x += rand_x
            if not cfg.apply_flat:
                height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
                height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, :mid_y+rand_y-half_valid_width] = 0
                height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, mid_y+rand_y+half_valid_width:] = 0
            goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
        final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)

        if final_dis_x > width_pixels:
            final_dis_x = width_pixels - 0.5 // cfg.horizontal_scale
        goals[-1] = [final_dis_x, mid_y]
        height_field_raw = padding_height_field_raw(height_field_raw,cfg)
        if cfg.apply_roughness:
            height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
        return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale


@parkour_field_to_mesh
def parkour_step_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourStepTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
        step_height = eval(cfg.step_height,{'difficulty':difficulty} )
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
        height_field_raw = np.zeros((width_pixels, length_pixels))

        mid_y = length_pixels // 2  # length is actually y width
        dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale)
        dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale) 
        dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
        dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)

        step_height = round(step_height / cfg.vertical_scale)

        half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)

        platform_len = round(cfg.platform_len / cfg.horizontal_scale)
        platform_height = round(cfg.platform_height / cfg.vertical_scale)
        height_field_raw[0:platform_len, :] = platform_height

        dis_x = platform_len
        last_dis_x = dis_x
        stair_height = 0
        goals = np.zeros((num_goals, 2))
        goals[0] = [platform_len - round(1 / cfg.horizontal_scale), mid_y]
        goal_heights = np.ones((num_goals)) * platform_height

        num_stones = num_goals - 2
        for i in range(num_stones):
            rand_x = np.random.randint(dis_x_min, dis_x_max)
            rand_y = np.random.randint(dis_y_min, dis_y_max)
            if i < num_stones // 2:
                stair_height += step_height
            elif i > num_stones // 2:
                stair_height -= step_height
            height_field_raw[dis_x:dis_x+rand_x, ] = stair_height
            dis_x += rand_x
            height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = 0
            height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = 0
            
            last_dis_x = dis_x
            goals[i+1] = [dis_x-rand_x//2, mid_y+rand_y]
            goal_heights[i+1] = stair_height
        final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
        # import ipdb; ipdb.set_trace()
        if final_dis_x > width_pixels:
            final_dis_x = width_pixels - 0.5 // cfg.horizontal_scale
        goals[-1] = [final_dis_x, mid_y]
        height_field_raw = padding_height_field_raw(height_field_raw,cfg)
        if cfg.apply_roughness:
            height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
        return height_field_raw, goals * cfg.horizontal_scale, goal_heights 

@parkour_field_to_mesh
def parkour_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
        height_field_raw = np.zeros((width_pixels, length_pixels))
        height_field_raw[:] = -round(np.random.uniform(cfg.pit_depth[0], cfg.pit_depth[1]) / cfg.vertical_scale)
        mid_y = length_pixels // 2  # length is actually y width
        stone_len = eval(cfg.stone_len, {"difficulty": difficulty})
        stone_len = np.random.uniform(*stone_len)
        stone_len = 2 * round(stone_len / 2.0, 1)
        stone_len = round(stone_len / cfg.horizontal_scale)
        x_range = eval(cfg.x_range, {"difficulty": difficulty})
        y_range = eval(cfg.y_range, {"difficulty": difficulty})
        dis_x_min = stone_len + round(x_range[0] / cfg.horizontal_scale)
        dis_x_max = stone_len + round(x_range[1] / cfg.horizontal_scale)
        dis_y_min = round(y_range[0] / cfg.horizontal_scale)
        dis_y_max = round(y_range[1] / cfg.horizontal_scale)

        platform_len = round(cfg.platform_len / cfg.horizontal_scale)
        platform_height = round(cfg.platform_height / cfg.vertical_scale)
        height_field_raw[0:platform_len, :] = platform_height
        
        stone_width = round(cfg.stone_width / cfg.horizontal_scale)
        last_stone_len = round(cfg.last_stone_len / cfg.horizontal_scale)

        incline_height = eval(cfg.incline_height, {"difficulty": difficulty})
        last_incline_height = eval(cfg.last_incline_height, {"difficulty": difficulty, "incline_height":incline_height})
        last_incline_height = round(last_incline_height / cfg.vertical_scale)
        incline_height = round(incline_height / cfg.vertical_scale)

        dis_x = platform_len - np.random.randint(dis_x_min, dis_x_max) + stone_len // 2
        goals = np.zeros((num_goals, 2))
        goal_heights = np.ones((num_goals)) * platform_height
        goals[0] = [platform_len -  stone_len // 2, mid_y]
        left_right_flag = np.random.randint(0, 2)
        dis_z = 0
        num_stones = num_goals - 2
        for i in range(num_stones):
            dis_x += np.random.randint(dis_x_min, dis_x_max)
            pos_neg = round(2*(left_right_flag - 0.5))
            dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
            if i == num_stones - 1:
                dis_x += last_stone_len // 4
                heights = np.tile(np.linspace(-last_incline_height, last_incline_height, stone_width), (last_stone_len, 1)) * pos_neg
                height_field_raw[dis_x-last_stone_len//2:dis_x+last_stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
            else:
                heights = np.tile(np.linspace(-incline_height, incline_height, stone_width), (stone_len, 1)) * pos_neg
                height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
            
            goals[i+1] = [dis_x, dis_y]
            goal_heights[i+1] = np.mean(heights.astype(int))

            left_right_flag = 1 - left_right_flag
        final_dis_x = dis_x + 2*np.random.randint(dis_x_min, dis_x_max)
        final_platform_start = dis_x + last_stone_len // 2 + round(0.05 // cfg.horizontal_scale)
        height_field_raw[final_platform_start:, :] = platform_height
        goals[-1] = [final_dis_x, mid_y]
        height_field_raw = padding_height_field_raw(height_field_raw,cfg)
        if cfg.apply_roughness:
            height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
        
        return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale




@parkour_field_to_mesh
def parkour_demo_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourDemoTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
    goals = np.zeros((num_goals, 2))
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    mid_y = length_pixels // 2  # length is actually y width

    height_field_raw = np.zeros((width_pixels, length_pixels))
    goal_heights = np.ones((num_goals)) * round(cfg.platform_height / cfg.vertical_scale)
    platform_length = round(2 / cfg.horizontal_scale)
    hurdle_depth = round(np.random.uniform(0.35, 0.4) / cfg.horizontal_scale)
    hurdle_height = round(np.random.uniform(0.3, 0.36) / cfg.vertical_scale)
    hurdle_width = round(np.random.uniform(1, 1.2) / cfg.horizontal_scale)
    goals[0] = [platform_length + hurdle_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+hurdle_depth, round(mid_y-hurdle_width/2):round(mid_y+hurdle_width/2)] = hurdle_height

    platform_length += round(np.random.uniform(1.5, 2.5) / cfg.horizontal_scale)
    first_step_depth = round(np.random.uniform(0.45, 0.8) / cfg.horizontal_scale)
    first_step_height = round(np.random.uniform(0.35, 0.45) / cfg.vertical_scale)
    first_step_width = round(np.random.uniform(1, 1.2) / cfg.horizontal_scale)
    goals[1] = [platform_length+first_step_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+first_step_depth, round(mid_y-first_step_width/2):round(mid_y+first_step_width/2)] = first_step_height
    goal_heights[1] = first_step_height

    platform_length += first_step_depth
    second_step_depth = round(np.random.uniform(0.45, 0.8) / cfg.horizontal_scale)
    second_step_height = first_step_height
    second_step_width = first_step_width
    goals[2] = [platform_length+second_step_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+second_step_depth, round(mid_y-second_step_width/2):round(mid_y+second_step_width/2)] = second_step_height
    goal_heights[2] = second_step_height

    # gap
    platform_length += second_step_depth
    gap_size = round(np.random.uniform(0.5, 0.8) / cfg.horizontal_scale)
    
    # step down
    platform_length += gap_size
    third_step_depth = round(np.random.uniform(0.25, 0.6) / cfg.horizontal_scale)
    third_step_height = first_step_height
    third_step_width = round(np.random.uniform(1, 1.2) / cfg.horizontal_scale)
    goals[3] = [platform_length+third_step_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+third_step_depth, round(mid_y-third_step_width/2):round(mid_y+third_step_width/2)] = third_step_height
    goal_heights[3] = third_step_height
    
    platform_length += third_step_depth
    forth_step_depth = round(np.random.uniform(0.25, 0.6) / cfg.horizontal_scale)
    forth_step_height = first_step_height
    forth_step_width = third_step_width
    goals[4] = [platform_length+forth_step_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+forth_step_depth, round(mid_y-forth_step_width/2):round(mid_y+forth_step_width/2)] = forth_step_height
    goal_heights[4] = forth_step_height
    
    # parkour
    platform_length += forth_step_depth
    gap_size = round(np.random.uniform(0.1, 0.4) / cfg.horizontal_scale)
    platform_length += gap_size
    
    left_y = mid_y + round(np.random.uniform(0.15, 0.3) / cfg.horizontal_scale)
    right_y = mid_y - round(np.random.uniform(0.15, 0.3) / cfg.horizontal_scale)
    
    slope_height = round(np.random.uniform(0.15, 0.22) / cfg.vertical_scale)
    slope_depth = round(np.random.uniform(0.75, 0.85) / cfg.horizontal_scale)
    slope_width = round(1.0 / cfg.horizontal_scale)
    
    platform_height = slope_height + np.random.randint(0, 0.2 / cfg.vertical_scale)

    goals[5] = [platform_length+slope_depth/2, left_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * 1
    height_field_raw[platform_length:platform_length+slope_depth, left_y-slope_width//2: left_y+slope_width//2] = heights.astype(int) + platform_height
    goal_heights[5] = np.mean(heights.astype(int) + platform_height)
    
    platform_length += slope_depth + gap_size
    goals[6] = [platform_length+slope_depth/2, right_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * -1
    height_field_raw[platform_length:platform_length+slope_depth, right_y-slope_width//2: right_y+slope_width//2] = heights.astype(int) + platform_height
    goal_heights[6] = np.mean(heights.astype(int) + platform_height)
    
    platform_length += slope_depth + gap_size + round(0.4 / cfg.horizontal_scale)
    goals[-1] = [platform_length, left_y]

    height_field_raw = padding_height_field_raw(height_field_raw,cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
    
    return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale


