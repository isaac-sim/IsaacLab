# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from Stable-Baselines3."""

import argparse
import contextlib
import os
import random
import sys
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.seed import configure_seed

from isaaclab_rl.entrypoints.common import (
    CHECKPOINT_SELECTORS,
    add_frontend_args,
    apply_video_recording,
    create_isaaclab_env,
    resolve_checkpoint_selector,
    resolve_play_task_name,
)
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    get_checkpoint_path,
    resolve_task_config,
    setup_preset_cli,
)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Length of each recorded video clip in env steps. Overrides the value in VideoRecorderCfg.",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=None,
    help="Interval between video clips in env steps. Overrides the value in VideoRecorderCfg.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path, or latest/best.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
parser.add_argument(
    "--train_env_cfg",
    action="store_true",
    default=False,
    help="Play with the training environment configuration as-is, skipping play-mode overrides.",
)
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, hydra_args = setup_preset_cli(parser, agent_library="sb3")
args_cli.task = resolve_play_task_name(args_cli.task)

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args


def main():
    """Play with stable-baselines agent."""
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
    with launch_simulation(env_cfg, args_cli):
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)

        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
        env_cfg.seed = agent_cfg["seed"]
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        log_root_path = os.path.join("logs", "sb3", train_task_name)
        log_root_path = os.path.abspath(log_root_path)
        if args_cli.use_pretrained_checkpoint:
            checkpoint_path = get_published_pretrained_checkpoint("sb3", train_task_name)
            if not checkpoint_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint in CHECKPOINT_SELECTORS:
            checkpoint_path = resolve_checkpoint_selector(
                log_root_path,
                args_cli.checkpoint,
                library="sb3",
                task=train_task_name,
                checkpoint_pattern=r"model(?:_.*)?\.zip",
                preferred_checkpoint_pattern=r"model\.zip",
                metadata={"agent": args_cli.agent},
            )
        elif args_cli.checkpoint is None:
            # prefer the final model (``model.zip``); fall back to the latest periodic checkpoint when it has
            # not been written yet (e.g. short or interrupted runs)
            checkpoint_path = get_checkpoint_path(
                log_root_path, ".*", r"model_.*\.zip", sort_alpha=False, preferred_checkpoint=r"model\.zip"
            )
        else:
            checkpoint_path = args_cli.checkpoint
        log_dir = os.path.dirname(checkpoint_path)

        env_cfg.log_dir = log_dir
        apply_video_recording(env_cfg, log_dir, args_cli, subdir="play")

        env = create_isaaclab_env(
            args_cli.task,
            env_cfg,
            args_cli,
            convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
        )

        agent_cfg = process_sb3_cfg(agent_cfg, env.unwrapped.num_envs)

        env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

        vec_norm_path = checkpoint_path.replace("/model", "/model_vecnormalize").replace(".zip", ".pkl")
        vec_norm_path = Path(vec_norm_path)

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

        print(f"Loading checkpoint from: {checkpoint_path}")
        agent = PPO.load(checkpoint_path, env, print_system_info=True)
        # configure_seed must run after PPO.load so torch determinism does not disturb SB3's initialization
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, torch_deterministic=True)

        dt = env.unwrapped.step_dt

        obs = env.reset()
        timestep = 0
        try:
            while True:
                start_time = time.time()
                with torch.inference_mode():
                    actions, _ = agent.predict(obs, deterministic=True)
                    obs, _, _, _ = env.step(actions)
                if args_cli.video:
                    timestep += 1
                    video_stop = args_cli.video_length
                    if video_stop is None:
                        recorders = getattr(env_cfg, "video_recorders", [])
                        video_stop = recorders[0].video_length if recorders else None
                    if video_stop is not None and timestep >= video_stop:
                        break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            env.close()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
