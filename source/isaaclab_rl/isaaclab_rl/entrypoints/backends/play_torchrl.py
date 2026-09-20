# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TorchRL playback backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import os

import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

from ..common import (
    CHECKPOINT_SELECTORS,
    add_common_play_args,
    apply_env_overrides,
    apply_video_recording,
    create_isaaclab_env,
    enable_cameras_for_video,
    normalize_task_name,
    pre_launch_video_config,
    resolve_checkpoint_selector,
    resolve_play_task_name,
    resolve_seed,
    run_playback,
    set_hydra_args,
    show_run_summary,
    startup_screen,
)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse TorchRL playback arguments."""
    parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from TorchRL.")
    add_common_play_args(
        parser,
        agent_default="torchrl_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path, latest, or best.")
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    args_cli.task = resolve_play_task_name(args_cli.task)
    enable_cameras_for_video(args_cli)
    set_hydra_args(hydra_args)
    return args_cli


def _resolve_checkpoint(args_cli: argparse.Namespace, log_root_path: str) -> str:
    """Resolve the checkpoint to play."""
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="torchrl",
            task=normalize_task_name(args_cli.task),
            checkpoint_pattern=r"model_.*\.pt",
            metadata={"agent": args_cli.agent},
        )
    if args_cli.checkpoint and os.path.isdir(args_cli.checkpoint):
        return get_checkpoint_path(
            os.path.dirname(args_cli.checkpoint),
            os.path.basename(args_cli.checkpoint),
            r"model_.*\.pt",
            sort_alpha=False,
        )
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, ".*", r"model_.*\.pt", sort_alpha=False)


def run(argv: list[str]) -> None:
    """Play a checkpoint of a TorchRL agent."""
    args_cli = _parse_args(argv)
    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
        pre_launch_video_config(env_cfg, args_cli)
        show_run_summary(screen, args_cli, env_cfg, library="torchrl", action="play")
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli):
            # torchrl is an optional extra; importing it after the task config is resolved lets preset errors
            # surface even when it is not installed
            from torchrl.envs import ExplorationType, set_exploration_type

            from ...torchrl import IsaacLabTorchRLWrapper, make_actor

            apply_env_overrides(args_cli, env_cfg)
            args_cli.seed = resolve_seed(args_cli.seed)
            if args_cli.seed is not None:
                agent_cfg.seed = args_cli.seed
            agent_cfg.device = env_cfg.sim.device
            env_cfg.seed = agent_cfg.seed

            log_root_path = os.path.abspath(os.path.join("logs", "torchrl", agent_cfg.experiment_name))
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            checkpoint_path = _resolve_checkpoint(args_cli, log_root_path)
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

                tensordict = env.reset()

                def step() -> None:
                    nonlocal tensordict
                    with set_exploration_type(ExplorationType.DETERMINISTIC):
                        _, tensordict = env.step_and_maybe_reset(actor(tensordict))

                screen.close()
                run_playback(step, dt=env.unwrapped.step_dt, args_cli=args_cli, env_cfg=env_cfg)
            finally:
                env.close()
