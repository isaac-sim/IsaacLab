"""This script tests the RL environment for locomotion tasks with velocity tracking commands.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Experiments on running the locomotion RL environment.")
# parser.add_argument("--video", action="store_true", default=False, help="Record video during training.")
# parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
# parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
# parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
# parser.add_argument("--task", type=str, default="locomotion", help="Name of the task.")
# parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
# parser.add_argument("--headless", type=bool, default=True, help="Run in headless mode.")
# parser.add_argument("--robot", type=str, default="android", help="Name of the robot.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import traceback

import carb

from omni.isaac.orbit.envs import RLTaskEnv
# from omni.isaac.contrib_tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from omni.isaac.contrib_tasks.locomotion.velocity.config.android.flat_env_cfg import AndroidFlatEnvCfg
from omni.isaac.contrib_tasks.locomotion.velocity.config.android.rough_env_cfg import AndroidRoughEnvCfg

def main():
    """"Main function."""
    # parse the arguments
    env_cfg = AndroidFlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = RLTaskEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, reward, done, info = env.step(joint_efforts)
            count += 1
            if count % 100 == 0:
                print(f"[INFO]: Step {count}, Reward: {reward}, Done: {done}, Info: {info}")

    # close the environment
    env.close()


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as err:
        carb.log_error(err)
        traceback.print_exc()
        raise
    finally:
        # close sim app
        simulation_app.close()
