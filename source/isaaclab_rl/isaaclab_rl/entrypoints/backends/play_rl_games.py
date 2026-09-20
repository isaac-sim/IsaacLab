# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL-Games playback backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import os
import re

from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

from ...rl_games import RlGamesVecEnvWrapper, register_rl_games_env
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
    """Parse RL-Games playback arguments."""
    parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
    add_common_play_args(
        parser,
        agent_default="rl_games_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path, latest/best, or pretrained.")
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    args_cli.task = resolve_play_task_name(args_cli.task)
    enable_cameras_for_video(args_cli)
    set_hydra_args(hydra_args)
    return args_cli


def _resolve_checkpoint(
    args_cli: argparse.Namespace, agent_cfg: dict, env_cfg: object, log_root_path: str
) -> str | None:
    """Resolve the checkpoint to play, or None when no published checkpoint exists."""
    config_name = agent_cfg["params"]["config"]["name"]
    if args_cli.checkpoint == "pretrained":
        return resolve_published_checkpoint("rl_games", args_cli.task, env_cfg)
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="rl_games",
            task=normalize_task_name(args_cli.task),
            checkpoint_pattern=r".*\.pth",
            other_dirs=["nn"],
            preferred_checkpoint_pattern=rf"{re.escape(config_name)}\.pth",
            metadata={"agent": args_cli.agent},
        )
    if args_cli.checkpoint is None:
        # prefer the best-reward checkpoint (``<name>.pth``) and fall back to the latest one when it has not
        # been written yet (e.g. short runs); pass ``--checkpoint latest`` to always use the newest one
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        return get_checkpoint_path(
            log_root_path, run_dir, ".*", other_dirs=["nn"], preferred_checkpoint=f"{config_name}.pth"
        )
    return retrieve_file_path(args_cli.checkpoint)


def run(argv: list[str]) -> None:
    """Play a checkpoint of an RL-Games agent."""
    args_cli = _parse_args(argv)
    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
        if not isinstance(agent_cfg, dict) or not isinstance(agent_cfg.get("params"), dict):
            raise SystemExit(
                f"Invalid RL-Games agent configuration from --agent {args_cli.agent}: expected a dictionary with"
                f" a 'params' dictionary, got {type(agent_cfg).__name__}. Select an RL-Games configuration with"
                " --agent rl_games_cfg_entry_point, or use --rl_library rsl_rl for RSL-RL runner configurations."
            )
        pre_launch_video_config(env_cfg, args_cli)
        show_run_summary(screen, args_cli, env_cfg, library="rl_games", action="play")
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli):
            apply_env_overrides(args_cli, env_cfg)
            params = agent_cfg["params"]
            args_cli.seed = resolve_seed(args_cli.seed)
            if args_cli.seed is not None:
                params["seed"] = args_cli.seed
            env_cfg.seed = params["seed"]

            log_root_path = os.path.abspath(os.path.join("logs", "rl_games", params["config"]["name"]))
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
            if resume_path is None:
                return
            log_dir = os.path.dirname(os.path.dirname(resume_path))
            env_cfg.log_dir = log_dir
            apply_video_recording(env_cfg, log_dir, args_cli, subdir="play")

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )

            screen.stage("Loading policy")
            env = RlGamesVecEnvWrapper.from_agent_cfg(env, agent_cfg)
            register_rl_games_env(env)
            params["load_checkpoint"] = True
            params["load_path"] = resume_path
            params["config"]["num_actors"] = env.unwrapped.num_envs
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner = Runner()
            # configure_seed must run after Runner() so torch determinism does not disturb its initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)
            runner.load(agent_cfg)
            agent: BasePlayer = runner.create_player()
            agent.restore(resume_path)
            agent.reset()

            obs = env.reset()
            if isinstance(obs, dict):
                obs = obs["obs"]
            agent.get_batch_size(obs, 1)
            if agent.is_rnn:
                agent.init_rnn()

            def step() -> None:
                nonlocal obs
                actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=agent.is_deterministic)
                obs, _, dones, _ = env.step(actions)
                # reset recurrent states for episodes that have terminated
                if agent.is_rnn and agent.states is not None and len(dones) > 0:
                    for state in agent.states:
                        state[:, dones, :] = 0.0

            screen.close()
            run_playback(step, dt=env.unwrapped.step_dt, args_cli=args_cli, env_cfg=env_cfg)
            env.close()
