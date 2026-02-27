# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from Stable-Baselines3."""

import argparse
import contextlib
import os
import random
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.dict import print_dict

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation, resolve_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args


def main():
    """Play with stable-baselines agent."""
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    with launch_simulation(env_cfg, args_cli):
        # grab task name for checkpoint path
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)

        # override configurations with non-hydra CLI arguments
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
        env_cfg.seed = agent_cfg["seed"]
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # directory for logging into
        log_root_path = os.path.join("logs", "sb3", train_task_name)
        log_root_path = os.path.abspath(log_root_path)
        # checkpoint and log_dir stuff
        if args_cli.use_pretrained_checkpoint:
            checkpoint_path = get_published_pretrained_checkpoint("sb3", train_task_name)
            if not checkpoint_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint is None:
            if args_cli.use_last_checkpoint:
                checkpoint = "model_.*.zip"
            else:
                checkpoint = "model.zip"
            checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint, sort_alpha=False)
        else:
            checkpoint_path = args_cli.checkpoint
        log_dir = os.path.dirname(checkpoint_path)

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # post-process agent configuration
        agent_cfg = process_sb3_cfg(agent_cfg, env.unwrapped.num_envs)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
        # wrap around environment for stable baselines
        env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

        vec_norm_path = checkpoint_path.replace("/model", "/model_vecnormalize").replace(".zip", ".pkl")
        vec_norm_path = Path(vec_norm_path)

        # normalize environment (if needed)
        if vec_norm_path.exists():
            print(f"Loading saved normalization: {vec_norm_path}")
            env = VecNormalize.load(vec_norm_path, env)
            env.training = False
            env.norm_reward = False
        elif "normalize_input" in agent_cfg:
            env = VecNormalize(
                env,
                training=True,
                norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
                clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            )

        # create agent from stable baselines
        print(f"Loading checkpoint from: {checkpoint_path}")
        agent = PPO.load(checkpoint_path, env, print_system_info=True)

        dt = env.unwrapped.step_dt

        # reset environment
        obs = env.reset()
        timestep = 0
        # simulate environment
        try:
            while True:
                start_time = time.time()
                with torch.inference_mode():
                    actions, _ = agent.predict(obs, deterministic=True)
                    obs, _, _, _ = env.step(actions)
                if args_cli.video:
                    timestep += 1
                    if timestep == args_cli.video_length:
                        break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            # close the simulator
            env.close()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
