#!/usr/bin/env python3
"""Simple script to visualize Newton Warp renders."""

import argparse
import numpy as np
import torch
from PIL import Image
import os

from isaaclab.app import AppLauncher

# Add argparse for clean launcher setup
parser = argparse.ArgumentParser(description="Visualize Newton Warp renders")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--task", type=str, default="Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0")
parser.add_argument("--output_dir", type=str, default="newton_renders", help="Output directory")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launcher
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    """Capture and save Newton Warp renders."""
    # Parse environment config
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cuda:0",
        num_envs=args_cli.num_envs,
    )
    
    # Set Newton Warp renderer on the camera
    if hasattr(env_cfg.scene, 'base_camera'):
        env_cfg.scene.base_camera.renderer_type = "newton_warp"
        env_cfg.scene.base_camera.width = 128  # Larger for better visualization
        env_cfg.scene.base_camera.height = 128
    else:
        print("Warning: base_camera not found in scene config")
    
    # Get actual resolution from camera config
    width = env_cfg.scene.base_camera.width if hasattr(env_cfg.scene, 'base_camera') else 64
    height = env_cfg.scene.base_camera.height if hasattr(env_cfg.scene, 'base_camera') else 64
    
    renderer_type = env_cfg.scene.base_camera.renderer_type if hasattr(env_cfg.scene, 'base_camera') else "unknown"
    print(f"\n{'='*80}")
    print(f"Creating environment with Newton Warp renderer")
    print(f"  Task: {args_cli.task}")
    print(f"  Num envs: {args_cli.num_envs}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Renderer: {renderer_type}")
    print(f"{'='*80}\n")
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Create output directory
    os.makedirs(args_cli.output_dir, exist_ok=True)
    
    try:
        # Reset and capture initial state
        print("Resetting environment...")
        obs, _ = env.reset()
        
        # Step a few times to get varied poses
        print(f"Capturing renders...")
        for step_idx in range(5):
            # Get camera data
            camera = env.scene.sensors["base_camera"]
            camera_data = camera.data.output
            
            print(f"\n  Frame {step_idx}:")
            print(f"    Available data: {list(camera_data.keys())}")
            
            # Save RGB
            if "rgb" in camera_data:
                rgb_data = camera_data["rgb"]
                print(f"    RGB shape: {rgb_data.shape}, type: {type(rgb_data)}")
                
                # Convert to numpy
                if hasattr(rgb_data, 'numpy'):  # Warp array
                    rgb_np = rgb_data.numpy()
                elif isinstance(rgb_data, torch.Tensor):
                    rgb_np = rgb_data.cpu().numpy()
                else:
                    rgb_np = np.array(rgb_data)
                
                # Save first 4 environments
                for env_id in range(min(4, args_cli.num_envs)):
                    img = rgb_np[env_id]  # (H, W, 3)
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                    
                    img_pil = Image.fromarray(img, mode='RGB')
                    path = f"{args_cli.output_dir}/step{step_idx:02d}_env{env_id}_rgb.png"
                    img_pil.save(path)
                    if step_idx == 0:  # Only print for first frame
                        print(f"      Saved: {path}")
            
            # Save depth
            if "distance_to_image_plane" in camera_data:
                depth_data = camera_data["distance_to_image_plane"]
                print(f"    Depth shape: {depth_data.shape}")
                
                # Convert to numpy
                if hasattr(depth_data, 'numpy'):
                    depth_np = depth_data.numpy()
                elif isinstance(depth_data, torch.Tensor):
                    depth_np = depth_data.cpu().numpy()
                else:
                    depth_np = np.array(depth_data)
                
                for env_id in range(min(4, args_cli.num_envs)):
                    depth = depth_np[env_id, :, :, 0]  # Remove channel dim
                    
                    # Normalize for visualization
                    d_min, d_max = depth.min(), depth.max()
                    if d_max > d_min:
                        depth_vis = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                    else:
                        depth_vis = np.zeros_like(depth, dtype=np.uint8)
                    
                    img_pil = Image.fromarray(depth_vis, mode='L')
                    path = f"{args_cli.output_dir}/step{step_idx:02d}_env{env_id}_depth.png"
                    img_pil.save(path)
                    if step_idx == 0:
                        print(f"      Saved: {path} (range: {d_min:.3f}-{d_max:.3f}m)")
            
            # Take random action
            action = torch.randn(env.num_envs, env.action_space.shape[0], device=env.device) * 0.05
            obs, *_ = env.step(action)
        
        print(f"\n{'='*80}")
        print(f"✅ Successfully saved renders to: {args_cli.output_dir}/")
        print(f"   Total images: {5 * min(4, args_cli.num_envs) * 2} (5 steps × {min(4, args_cli.num_envs)} envs × 2 types)")
        print(f"{'='*80}\n")
        
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
