# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from TorchRL."""

import argparse
import contextlib
import os
import random
import sys
import time

import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed

from isaaclab_rl.entrypoints.common import (
    CHECKPOINT_SELECTORS,
    add_frontend_args,
    apply_video_recording,
    create_isaaclab_env,
    pre_launch_video_config,
    request_determinism,
    resolve_checkpoint_selector,
    resolve_play_task_name,
    show_run_summary,
    startup_screen,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from TorchRL.")
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
    "--agent", type=str, default="torchrl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path, latest, or best.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--train_env_cfg",
    action="store_true",
    default=False,
    help="Play with the training environment configuration as-is, skipping play-mode overrides.",
)
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
args_cli.task = resolve_play_task_name(args_cli.task)

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args


def main():
    """Play with a TorchRL agent."""
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
    pre_launch_video_config(env_cfg, args_cli=args_cli)
    with startup_screen(args_cli, num_stages=3) as screen:
        show_run_summary(screen, args_cli, env_cfg, library="torchrl", action="play")
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli):
            from torchrl.envs import ExplorationType, set_exploration_type

            from isaaclab_rl.torchrl import IsaacLabTorchRLWrapper, make_actor

            task_name = args_cli.task.split(":")[-1]
            train_task_name = task_name.replace("-Play", "")
            if args_cli.seed == -1:
                args_cli.seed = random.randint(0, 10000)

            env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
            request_determinism(args_cli, env_cfg)
            if args_cli.seed is not None:
                agent_cfg.seed = args_cli.seed
            env_cfg.seed = agent_cfg.seed
            env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
            agent_cfg.device = env_cfg.sim.device

            log_root_path = os.path.abspath(os.path.join("logs", "torchrl", agent_cfg.experiment_name))
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            if args_cli.checkpoint in CHECKPOINT_SELECTORS:
                checkpoint_path = resolve_checkpoint_selector(
                    log_root_path,
                    args_cli.checkpoint,
                    library="torchrl",
                    task=train_task_name,
                    checkpoint_pattern=r"model_.*\.pt",
                    metadata={"agent": args_cli.agent},
                )
            elif args_cli.checkpoint and os.path.isdir(args_cli.checkpoint):
                checkpoint_path = get_checkpoint_path(
                    os.path.dirname(args_cli.checkpoint),
                    os.path.basename(args_cli.checkpoint),
                    r"model_.*\.pt",
                    sort_alpha=False,
                )
            elif args_cli.checkpoint:
                checkpoint_path = retrieve_file_path(args_cli.checkpoint)
            else:
                checkpoint_path = get_checkpoint_path(log_root_path, ".*", r"model_.*\.pt", sort_alpha=False)

            log_dir = os.path.dirname(checkpoint_path)
            env_cfg.log_dir = log_dir
            apply_video_recording(env_cfg, log_dir, args_cli, subdir="play", checkpoint_path=checkpoint_path)

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )
            try:
                env = IsaacLabTorchRLWrapper(env, clip_actions=agent_cfg.clip_actions)

                screen.stage("Loading policy")
                print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
                actor = make_actor(env, agent_cfg).to(env.device).eval()
                actor.load_state_dict(torch.load(checkpoint_path, map_location=env.device, weights_only=True))
                if args_cli.deterministic:
                    configure_seed(env_cfg.seed, torch_deterministic=True)

                dt = env.unwrapped.step_dt
                screen.close()
                tensordict = env.reset()
                timestep = 0
                print("[INFO] Policy playback is running, press Ctrl+C to exit...")
                while True:
                    start_time = time.time()
                    with torch.inference_mode(), set_exploration_type(ExplorationType.DETERMINISTIC):
                        tensordict = actor(tensordict)
                        _, tensordict = env.step_and_maybe_reset(tensordict)

                    if args_cli.video:
                        timestep += 1
                        video_stop = args_cli.video_length
                        if video_stop is None:
                            recorders = getattr(env_cfg, "video_recorders", [])
                            video_stop = recorders[0].video_length + recorders[0].step_offset if recorders else None
                        if video_stop is not None and timestep >= video_stop:
                            break

                    sleep_time = dt - (time.time() - start_time)
                    if args_cli.real_time and sleep_time > 0:
                        time.sleep(sleep_time)
            except KeyboardInterrupt:
                pass
            finally:
                env.close()


if __name__ == "__main__":
    main()
