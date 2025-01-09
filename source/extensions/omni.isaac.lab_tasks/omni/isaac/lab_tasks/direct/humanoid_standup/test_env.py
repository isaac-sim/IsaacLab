"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-SigmabanStandUp-Direct-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.isaac.lab_tasks.direct.humanoid_standup.standup_env as standup_env
from omni.isaac.lab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
import torch
import time
import numpy as np

def test_standup_env():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    env.reset(seed=42)

    for i in range(100000):
        action = env.action_space.sample()

        action = torch.tensor(action, dtype=torch.float32, device="cuda")

        action = torch.zeros_like(action, dtype=torch.float32, device="cuda")

        action[:, 4] = np.cos(i/60)


        # action = torch.zeros_like(action, dtype=torch.float32, device="cuda")

        observation, reward, terminated, truncated, info = env.step(action)

        # print(f"Step {i}: Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        # if i ==2:
        #     time.sleep(10)
        # if terminated:
        #     print("Episode finished after {} timesteps".format(_ + 1))
        #     break

    env.close()

if __name__ == "__main__":
    test_standup_env()