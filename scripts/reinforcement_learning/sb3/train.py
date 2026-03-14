# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Script to train RL agent with Stable Baselines3."""

import argparse
import contextlib
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import DirectMARLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument(
    "--video",
    nargs="?",
    const="perspective",
    default=None,
    metavar="MODE",
    help="Record videos during training. MODE is 'perspective' (default, wide-angle isometric view) or 'tiled' (camera-sensor tile-grid).",
)
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_interval", type=int, default=100_000, help="Log data every n timesteps.")
parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args


def cleanup_pbar(*args):
    """
    A small helper to stop training and
    cleanup progress bar properly on ctrl+c
    """
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, cleanup_pbar)


def main():
    """Train with stable-baselines agent."""
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    with launch_simulation(env_cfg, args_cli):
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)

        # override configurations with non-hydra CLI arguments
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
        # max iterations for training
        if args_cli.max_iterations is not None:
            agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

        # set the environment seed
        env_cfg.seed = agent_cfg["seed"]
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # directory for logging into
        run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        print(f"Exact experiment name requested from command line: {run_info}")
        log_dir = os.path.join(log_root_path, run_info)
        # dump the configuration into log-directory
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        # save command used to run the script
        command = " ".join(sys.orig_argv)
        (Path(log_dir) / "command.txt").write_text(command)

        # post-process agent configuration
        agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
        # read configurations about the agent-training
        policy_arch = agent_cfg.pop("policy")
        n_timesteps = agent_cfg.pop("n_timesteps")

        # set the IO descriptors export flag if requested
        if isinstance(env_cfg, ManagerBasedRLEnvCfg):
            env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        else:
            logger.warning(
                "IO descriptors are only supported for manager based RL environments."
                " No IO descriptors will be exported."
            )

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        # Forward the video mode ("tiled" / "perspective") to the recorder config before env creation.
        if args_cli.video and hasattr(env_cfg, "video_recorder") and env_cfg.video_recorder is not None:
            env_cfg.video_recorder.video_mode = args_cli.video

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        start_time = time.time()

        # wrap around environment for stable baselines
        env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

        norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
        norm_args = {}
        for key in norm_keys:
            if key in agent_cfg:
                norm_args[key] = agent_cfg.pop(key)

        if norm_args and norm_args.get("normalize_input"):
            print(f"Normalizing input, {norm_args=}")
            env = VecNormalize(
                env,
                training=True,
                norm_obs=norm_args["normalize_input"],
                norm_reward=norm_args.get("normalize_value", False),
                clip_obs=norm_args.get("clip_obs", 100.0),
                gamma=agent_cfg["gamma"],
                clip_reward=np.inf,
            )

        # create agent from stable baselines
        agent = PPO(policy_arch, env, verbose=1, tensorboard_log=log_dir, **agent_cfg)
        if args_cli.checkpoint is not None:
            agent = agent.load(args_cli.checkpoint, env, print_system_info=True)

        # callbacks for agent
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2)
        callbacks = [checkpoint_callback, LogEveryNTimesteps(n_steps=args_cli.log_interval)]

        # train the agent
        with contextlib.suppress(KeyboardInterrupt):
            agent.learn(
                total_timesteps=n_timesteps,
                callback=callbacks,
                progress_bar=True,
                log_interval=None,
            )
        # save the final model
        agent.save(os.path.join(log_dir, "model"))
        print("Saving to:")
        print(os.path.join(log_dir, "model.zip"))

        if isinstance(env, VecNormalize):
            print("Saving normalization")
            env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))

        print(f"Training time: {round(time.time() - start_time, 2)} seconds")

        # close the simulator
        env.close()


if __name__ == "__main__":
    main()
