# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

import argparse
import contextlib
import os
import random
import sys
import time

import skrl
import torch
from packaging import version

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.seed import configure_seed

from isaaclab_rl.entrypoints.common import (
    CHECKPOINT_SELECTORS,
    add_frontend_args,
    apply_video_recording,
    create_isaaclab_env,
    preserve_attribute,
    resolve_checkpoint_selector,
    resolve_play_task_name,
)
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

SKRL_VERSION = "2.1.0"

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
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
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path, or latest/best.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--train_env_cfg",
    action="store_true",
    default=False,
    help="Play with the training environment configuration as-is, skipping play-mode overrides.",
)
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, hydra_args = setup_preset_cli(parser, agent_library="skrl")
args_cli.task = resolve_play_task_name(args_cli.task)

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# -- check skrl version ------------------------------------------------------
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def main():
    """Play with SKRL while restoring the caller's global settings."""
    state_context = (
        preserve_attribute(skrl.config.jax, "backend")
        if args_cli.ml_framework.startswith("jax")
        else contextlib.nullcontext()
    )
    with state_context:
        _main()


def _main():
    """Execute SKRL playback."""
    env_cfg, experiment_cfg = resolve_task_config(
        args_cli.task, agent_cfg_entry_point, play_mode=not args_cli.train_env_cfg
    )
    with launch_simulation(env_cfg, args_cli):
        if args_cli.ml_framework.startswith("torch"):
            from skrl.utils.runner.torch import Runner
        elif args_cli.ml_framework.startswith("jax"):
            from skrl.utils.runner.jax import Runner

        from isaaclab_rl.skrl import SkrlVecEnvWrapper

        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # configure the ML framework into the global skrl variable
        if args_cli.ml_framework.startswith("jax"):
            skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)

        experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
        env_cfg.seed = experiment_cfg["seed"]

        log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint in CHECKPOINT_SELECTORS:
            resume_path = resolve_checkpoint_selector(
                log_root_path,
                args_cli.checkpoint,
                library="skrl",
                task=train_task_name,
                checkpoint_pattern=r".*",
                other_dirs=["checkpoints"],
                metadata={
                    "agent": agent_cfg_entry_point,
                    "algorithm": algorithm,
                    "ml_framework": args_cli.ml_framework,
                },
            )
        elif args_cli.checkpoint:
            resume_path = os.path.abspath(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(
                log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
            )
        log_dir = os.path.dirname(os.path.dirname(resume_path))

        env_cfg.log_dir = log_dir
        apply_video_recording(env_cfg, log_dir, args_cli, subdir="play")

        env = create_isaaclab_env(
            args_cli.task,
            env_cfg,
            args_cli,
            convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg) and algorithm in ["ppo"],
        )

        try:
            dt = env.step_dt
        except AttributeError:
            dt = env.unwrapped.step_dt

        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["agent"]["experiment"]["write_interval"] = 0
        experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        runner = Runner(env, experiment_cfg)
        # configure_seed must run after Runner() so torch determinism does not disturb its initialization
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, torch_deterministic=True)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
        runner.agent.enable_training_mode(False, apply_to_models=True)

        obs, _ = env.reset()
        states = env.state()
        timestep = 0
        try:
            while True:
                start_time = time.time()

                with torch.inference_mode():
                    outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
                    if hasattr(env, "possible_agents"):
                        actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                    else:
                        actions = outputs[-1].get("mean_actions", outputs[0])
                    obs, _, _, _, _ = env.step(actions)
                    states = env.state()
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
