import numpy as np
import trimesh
import scipy.interpolate as interpolate
import sys
sys.path.append("/home/wang/IsaacLab/Isaaclab_Parkour")
import parkour_isaaclab, os
sys.path.append("/home/wang/IsaacLab/go2_parkour_deploy")
from scripts.utils import load_local_cfg
import core

def random_uniform_terrain(difficulty, cfg, hf):
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
    hf += z_upsampled 
    return hf 


def parkour_demo_terrain_from_yaml(cfg: str):
    size_x, size_y = cfg.size
    hs = cfg.horizontal_scale
    difficulty = 0.7
    apply_rough = cfg.apply_roughness

    width_px  = int(round(size_x / hs))
    length_px = int(round(size_y / hs))
    mid_y = length_px // 2

    hf = np.zeros((width_px, length_px), dtype=int)
    platform_length = round(2 / cfg.horizontal_scale)
    # === hurdle ===
    platform_length = round(2 / cfg.horizontal_scale)
    hurdle_depth = round(np.random.uniform(0.35, 0.4) / cfg.horizontal_scale)
    hurdle_height = round(np.random.uniform(0.3, 0.36) / cfg.vertical_scale)
    hurdle_width = round(np.random.uniform(1, 1.2) / cfg.horizontal_scale)

    hf[platform_length:platform_length+hurdle_depth, round(mid_y-hurdle_width/2):round(mid_y+hurdle_width/2)] = hurdle_height

    # gap to step1
    platform_length += round(np.random.uniform(1.5, 2.5) / cfg.horizontal_scale)
    first_step_depth = round(np.random.uniform(0.45, 0.8) / cfg.horizontal_scale)
    first_step_height = round(np.random.uniform(0.1, 0.1) / cfg.vertical_scale)
    print("gap to step1")
    first_step_width = round(np.random.uniform(1, 1.2) / cfg.horizontal_scale)

    hf[platform_length:platform_length+first_step_depth, round(mid_y-first_step_width/2):round(mid_y+first_step_width/2)] = first_step_height

    platform_length += first_step_depth

    # === step2 === (height/width tied to step1)
    second_step_depth = round(np.random.uniform(0.45, 0.8) / cfg.horizontal_scale)
    second_step_height = first_step_height
    second_step_width = first_step_width
    hf[platform_length:platform_length+second_step_depth, round(mid_y-second_step_width/2):round(mid_y+second_step_width/2)] = second_step_height

    # gap
    platform_length += second_step_depth
    gap_size = round(np.random.uniform(0.5, 0.8) / cfg.horizontal_scale)
    platform_length += gap_size

    # === step3 ===
    third_step_depth = round(np.random.uniform(0.25, 0.6) / cfg.horizontal_scale)
    third_step_height = first_step_height
    third_step_width = round(np.random.uniform(1, 1.2) / cfg.horizontal_scale)
    hf[platform_length:platform_length+third_step_depth, round(mid_y-third_step_width/2):round(mid_y+third_step_width/2)] = third_step_height
    platform_length += third_step_depth
    forth_step_depth = round(np.random.uniform(0.25, 0.6) / cfg.horizontal_scale)
    forth_step_height = first_step_height
    forth_step_width = third_step_width
    hf[platform_length:platform_length+forth_step_depth, round(mid_y-forth_step_width/2):round(mid_y+forth_step_width/2)] = forth_step_height

    # === parkour slopes ===
    platform_length += forth_step_depth
    gap_size = round(np.random.uniform(0.1, 0.4) / cfg.horizontal_scale)
    platform_length += gap_size

    left_y = mid_y + round(np.random.uniform(0.15, 0.3) / cfg.horizontal_scale)
    right_y = mid_y - round(np.random.uniform(0.15, 0.3) / cfg.horizontal_scale)
    
    slope_height = round(np.random.uniform(0.15, 0.22) / cfg.vertical_scale)
    slope_depth = round(np.random.uniform(0.75, 0.85) / cfg.horizontal_scale)
    slope_width = round(1.0 / cfg.horizontal_scale)
    
    platform_height = slope_height + np.random.randint(0, 0.2 / cfg.vertical_scale)

    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * 1
    hf[platform_length:platform_length+slope_depth, left_y-slope_width//2: left_y+slope_width//2] = heights.astype(int) + platform_height
    
    platform_length += slope_depth + gap_size
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * -1
    hf[platform_length:platform_length+slope_depth, right_y-slope_width//2: right_y+slope_width//2] = heights.astype(int) + platform_height
    
    platform_length += slope_depth + gap_size + round(0.4 / cfg.horizontal_scale)

    if apply_rough:
        hf = random_uniform_terrain(difficulty, cfg, hf)

    return hf, cfg

def export_hfield_png(hf, cfg, out_png="terrain.png"):
    vs = cfg.vertical_scale
    hs = cfg.horizontal_scale
    H = hf.astype(np.float32) * vs
    Hmin = float(H.min())
    Hmax = float(H.max())
    Zscale = max(Hmax - Hmin, 1e-6)   
    Hn = (H - Hmin) / Zscale + 1e-6
    from PIL import Image
    Image.fromarray((Hn.T * 65535).astype(np.uint16), mode="I;16").save(out_png)

    nx, ny = hf.shape
    sx = max((nx * hs) / 2.0, 1e-6) 
    sy = max((ny * hs) / 2.0, 1e-6)
    sz = Zscale                     
    
    return nx, ny, sx, sy, sz

def export_mesh_obj(hf, cfg, out_obj="terrain.obj", center_origin=True):
    vs = cfg.vertical_scale
    hs = cfg.horizontal_scale
    nx, ny = hf.shape
    X = np.arange(nx) * hs
    Y = np.arange(ny) * hs
    if center_origin:
        X = X - X.max() / 2.0
        Y = Y - Y.max() / 2.0
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    ZZ = hf.astype(np.float32) * vs
    verts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            v00 = i * ny + j
            v01 = i * ny + (j + 1)
            v10 = (i + 1) * ny + j
            v11 = (i + 1) * ny + (j + 1)
            faces.append([v00, v01, v11])
            faces.append([v00, v11, v10])
    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=False)
    mesh.export(out_obj)
    return out_obj

def main(args):

  logs_path = '/'
  for path in parkour_isaaclab.__path__[0].split('/')[1:-1]:
      logs_path = os.path.join(logs_path, path)
  logs_path = os.path.join(logs_path,'logs',args.rl_lib,args.task, args.expid)
  cfgs_path = os.path.join(logs_path, 'params')
  print(cfgs_path)
  env_cfg = load_local_cfg(cfgs_path, 'env')
  
  parkour_demo_cfg = env_cfg.scene.terrain.terrain_generator.sub_terrains.parkour_demo
  parkour_demo_cfg.horizontal_scale = 0.1
  parkour_demo_cfg.size = (16.,4.0)
  parkour_demo_cfg.noise_range = (0.02, 0.02)
  parkour_demo_cfg.slop_threshold = 0.5
  hf, cfg = parkour_demo_terrain_from_yaml(parkour_demo_cfg)
  _, _, _, _, _ = export_hfield_png(hf, cfg, os.path.join(os.path.join(core.__path__[0],'go2', 'terrain.png')))

if __name__ == "__main__":
    import argparse
    # mp.set_start_method("spawn")
    parser = argparse.ArgumentParser(description='sim_2_sim')
    parser.add_argument("--rl_lib", type=str, default='rsl_rl')
    parser.add_argument("--task", type=str, default='unitree_go2_parkour')
    parser.add_argument("--expid", type=str, default='2025-09-14_13-28-40')
    args = parser.parse_args()
    main(args)
