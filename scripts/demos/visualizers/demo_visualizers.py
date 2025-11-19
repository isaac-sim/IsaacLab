#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quick demo script for testing different visualizer backends.

Usage:
    # Newton OpenGL (default)
    python demo_visualizers.py --viz newton
    
    # Rerun web viewer
    python demo_visualizers.py --viz rerun
    
    # Omniverse viewport
    python demo_visualizers.py --viz ov
    
    # Multiple visualizers
    python demo_visualizers.py --viz newton rerun
    
    # No visualizer
    python demo_visualizers.py --viz none
"""

import argparse
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim.visualizers import NewtonVisualizerCfg, OVVisualizerCfg, RerunVisualizerCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Simple CLI
parser = argparse.ArgumentParser(description="Visualizer Demo")
parser.add_argument("--task", type=str, default="Isaac-Cartpole-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
parser.add_argument("--viz", nargs="+", default=["newton"], 
                    choices=["newton", "rerun", "ov", "none"],
                    help="Visualizer(s) to use")
args = parser.parse_args()

# Load env config
env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)

# Setup visualizers based on CLI args
if "none" not in args.viz:
    viz_cfgs = []
    
    if "newton" in args.viz:
        viz_cfgs.append(NewtonVisualizerCfg(
            window_width=1280,
            window_height=720,
            camera_position=(10.0, 0.0, 3.0),
            train_mode=True,
        ))
    
    if "rerun" in args.viz:
        viz_cfgs.append(RerunVisualizerCfg(
            launch_viewer=True,
        ))
    
    if "ov" in args.viz:
        viz_cfgs.append(OVVisualizerCfg(
            create_viewport=True,
            viewport_name="Demo Viewport",
            window_width=1280,
            window_height=720,
            camera_position=(10.0, 10.0, 3.0),
            camera_target=(0.0, 0.0, 0.0),
        ))
    
    # Set visualizers (single or list)
    env_cfg.sim.visualizer_cfgs = viz_cfgs[0] if len(viz_cfgs) == 1 else viz_cfgs
else:
    env_cfg.sim.visualizer_cfgs = None

print(f"\n{'='*60}")
print(f"Visualizer Demo")
print(f"{'='*60}")
print(f"Task: {args.task}")
print(f"Num Envs: {args.num_envs}")
print(f"Visualizers: {args.viz}")
print(f"{'='*60}\n")

# Create env
import gymnasium as gym
env = gym.make(args.task, cfg=env_cfg)

# Quick test loop
print("Running simulation... (Ctrl+C to exit)")
env.reset()
step = 0

try:
    while env.unwrapped.sim.is_playing():
        with torch.inference_mode():
            # Random actions
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            obs, rew, terminated, truncated, info = env.step(actions)
            
            step += 1
            if step % 100 == 0:
                print(f"Step {step} | Reward: {rew.mean().item():.3f}")
        
except KeyboardInterrupt:
    print("\nInterrupted by user")

env.close()
print("Done!")

