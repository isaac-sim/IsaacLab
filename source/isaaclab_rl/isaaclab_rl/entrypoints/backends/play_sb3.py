# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable-Baselines3 playback backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.seed import configure_seed

import isaaclab_tasks.registry  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

from ...sb3 import Sb3VecEnvWrapper, process_sb3_cfg
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
    resolve_published_checkpoint,
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
    """Parse Stable-Baselines3 playback arguments."""
    parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
    add_common_play_args(
        parser,
        agent_default="sb3_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path, latest/best, or pretrained.")
    parser.add_argument(
        "--keep_all_info",
        action="store_true",
        default=False,
        help="Use a slower SB3 wrapper but keep all the extra training info.",
    )
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    set_hydra_args(hydra_args)
    return args_cli


def _resolve_checkpoint(args_cli: argparse.Namespace, env_cfg: object, log_root_path: str) -> str | None:
    """Resolve the checkpoint to play, or None when no published checkpoint exists."""
    if args_cli.checkpoint == "pretrained":
        return resolve_published_checkpoint("sb3", args_cli.task, env_cfg)
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="sb3",
            task=normalize_task_name(args_cli.task),
            checkpoint_pattern=r"model(?:_.*)?\.zip",
            preferred_checkpoint_pattern=r"model\.zip",
            metadata={"agent": args_cli.agent},
        )
    if args_cli.checkpoint is None:
        # prefer the final model (``model.zip``) and fall back to the latest periodic checkpoint when it has
        # not been written yet (e.g. short or interrupted runs)
        return get_checkpoint_path(
            log_root_path, ".*", r"model_.*\.zip", sort_alpha=False, preferred_checkpoint=r"model\.zip"
        )
    return args_cli.checkpoint


def run(argv: list[str]) -> None:
    """Play a checkpoint of a Stable-Baselines3 agent."""
    args_cli = _parse_args(argv)
    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
        pre_launch_video_config(env_cfg, args_cli)
        show_run_summary(screen, args_cli, env_cfg, library="sb3", action="play")
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli):
            apply_env_overrides(args_cli, env_cfg)
            args_cli.seed = resolve_seed(args_cli.seed)
            if args_cli.seed is not None:
                agent_cfg["seed"] = args_cli.seed
            env_cfg.seed = agent_cfg["seed"]

            log_root_path = os.path.abspath(os.path.join("logs", "sb3", normalize_task_name(args_cli.task)))
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            checkpoint_path = _resolve_checkpoint(args_cli, env_cfg, log_root_path)
            if checkpoint_path is None:
                return
            log_dir = os.path.dirname(checkpoint_path)
            env_cfg.log_dir = log_dir
            apply_video_recording(env_cfg, log_dir, args_cli, subdir="play")

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )
            agent_cfg = process_sb3_cfg(agent_cfg, env.unwrapped.num_envs)

            screen.stage("Loading policy")
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
                    norm_obs=agent_cfg.pop("normalize_input"),
                    clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
                )

            print(f"Loading checkpoint from: {checkpoint_path}")
            agent = PPO.load(checkpoint_path, env, print_system_info=True)
            # configure_seed must run after PPO.load so torch determinism does not disturb SB3's initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)

            obs = env.reset()

            def step() -> None:
                nonlocal obs
                actions, _ = agent.predict(obs, deterministic=True)
                obs, _, _, _ = env.step(actions)

            screen.close()
            run_playback(step, dt=env.unwrapped.step_dt, args_cli=args_cli, env_cfg=env_cfg)
            env.close()
