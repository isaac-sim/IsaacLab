#!/usr/bin/env python3
"""Visualize Newton Warp renderer output."""

import gymnasium as gym
import numpy as np
import torch
from PIL import Image
import os

# Import Isaac Lab
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def save_renders(env, output_dir="newton_renders", num_frames=5):
    """Capture and save rendered images from the environment.
    
    Args:
        env: The Isaac Lab environment
        output_dir: Directory to save images
        num_frames: Number of frames to capture
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Capturing {num_frames} frames from Newton Warp renderer...")
    print(f"Number of environments: {env.num_envs}")
    
    # Reset environment
    obs, _ = env.reset()
    
    for frame_idx in range(num_frames):
        # Get camera data
        camera = env.scene.sensors["base_camera"]
        camera_data = camera.data.output
        
        print(f"\nFrame {frame_idx + 1}/{num_frames}")
        print(f"Available data types: {list(camera_data.keys())}")
        
        # Save RGB images
        if "rgb" in camera_data:
            rgb_data = camera_data["rgb"]  # Shape: (num_envs, H, W, 3)
            print(f"  RGB shape: {rgb_data.shape}, dtype: {rgb_data.dtype}, device: {rgb_data.device}")
            
            # Convert to numpy and save each environment's view
            if isinstance(rgb_data, torch.Tensor):
                rgb_np = rgb_data.cpu().numpy()
            else:
                # Warp array
                rgb_np = rgb_data.numpy()
            
            for env_id in range(min(4, env.num_envs)):  # Save first 4 envs
                img = rgb_np[env_id]
                # Ensure uint8
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                
                img_pil = Image.fromarray(img)
                save_path = f"{output_dir}/frame{frame_idx:03d}_env{env_id}_rgb.png"
                img_pil.save(save_path)
                print(f"    Saved: {save_path}")
        
        # Save depth images
        if "distance_to_image_plane" in camera_data:
            depth_data = camera_data["distance_to_image_plane"]  # Shape: (num_envs, H, W, 1)
            print(f"  Depth shape: {depth_data.shape}, dtype: {depth_data.dtype}")
            
            # Convert to numpy
            if isinstance(depth_data, torch.Tensor):
                depth_np = depth_data.cpu().numpy()
            else:
                depth_np = depth_data.numpy()
            
            for env_id in range(min(4, env.num_envs)):  # Save first 4 envs
                depth = depth_np[env_id, :, :, 0]
                
                # Normalize depth to 0-255 for visualization
                depth_min, depth_max = depth.min(), depth.max()
                if depth_max > depth_min:
                    depth_norm = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                else:
                    depth_norm = np.zeros_like(depth, dtype=np.uint8)
                
                img_pil = Image.fromarray(depth_norm)
                save_path = f"{output_dir}/frame{frame_idx:03d}_env{env_id}_depth.png"
                img_pil.save(save_path)
                print(f"    Saved: {save_path} (min: {depth_min:.3f}, max: {depth_max:.3f})")
        
        # Take a random action and step
        action = torch.randn(env.num_envs, env.action_space.shape[0], device=env.device) * 0.1
        obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\n✅ Saved {num_frames} frames to {output_dir}/")
    print(f"   View with: eog {output_dir}/ or open the folder")


def main():
    """Main visualization script."""
    # Parse environment config
    env_cfg = parse_env_cfg(
        task="Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0",
        device="cuda:0",
        num_envs=4,
        use_fabric=True,
    )
    
    # Override to use Newton Warp renderer
    print("Setting up environment with Newton Warp renderer...")
    env_cfg.scene.renderer_type = "newton_warp"
    
    # Create environment
    env = gym.make("Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0", cfg=env_cfg)
    
    try:
        # Capture and save renders
        save_renders(env, output_dir="newton_renders_output", num_frames=5)
        
        # Create a grid visualization
        print("\nCreating comparison grid...")
        create_comparison_grid("newton_renders_output")
        
    finally:
        env.close()


def create_comparison_grid(render_dir):
    """Create a grid of renders for easy comparison."""
    import glob
    from PIL import Image, ImageDraw, ImageFont
    
    # Find all RGB images
    rgb_files = sorted(glob.glob(f"{render_dir}/frame000_env*_rgb.png"))
    depth_files = sorted(glob.glob(f"{render_dir}/frame000_env*_depth.png"))
    
    if not rgb_files:
        print("No images found to create grid")
        return
    
    # Load images
    rgb_imgs = [Image.open(f) for f in rgb_files[:4]]
    depth_imgs = [Image.open(f) for f in depth_files[:4]]
    
    # Get dimensions
    img_width, img_height = rgb_imgs[0].size
    
    # Create grid: 2 rows (RGB, Depth) x 4 cols (envs)
    grid_width = img_width * 4 + 30  # Add some padding
    grid_height = img_height * 2 + 50  # Add labels
    
    grid = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid)
    
    # Paste RGB images (top row)
    for i, img in enumerate(rgb_imgs):
        x = i * (img_width + 5) + 10
        y = 20
        grid.paste(img, (x, y))
        # Add label
        draw.text((x + 5, 5), f"Env {i} RGB", fill='black')
    
    # Paste depth images (bottom row)
    for i, img in enumerate(depth_imgs):
        x = i * (img_width + 5) + 10
        y = img_height + 30
        grid.paste(img, (x, y))
        # Add label
        draw.text((x + 5, img_height + 25), f"Env {i} Depth", fill='black')
    
    # Save grid
    grid_path = f"{render_dir}/comparison_grid.png"
    grid.save(grid_path)
    print(f"✅ Created comparison grid: {grid_path}")


if __name__ == "__main__":
    main()
