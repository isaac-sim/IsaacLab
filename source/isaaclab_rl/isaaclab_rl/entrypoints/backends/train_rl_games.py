# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL-Games training backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import time
from datetime import datetime
from distutils.util import strtobool

from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.app import add_launcher_args, launch_simulation, report_activity
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
from isaaclab_tasks.utils.training_asset_log import log_training_asset_paths

from ...rl_games import MultiObserver, PbtAlgoObserver, RlGamesVecEnvWrapper, register_rl_games_env
from ..common import (
    CHECKPOINT_SELECTORS,
    add_common_train_args,
    apply_env_overrides,
    apply_video_recording,
    create_isaaclab_env,
    dump_train_configs,
    enable_cameras_for_video,
    pre_launch_video_config,
    resolve_checkpoint_selector,
    resolve_seed,
    set_hydra_args,
    show_run_summary,
    startup_screen,
    validate_distributed_device,
    wrap_sensor_capture,
    write_run_manifest,
)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse RL-Games training arguments."""
    parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
    add_common_train_args(
        parser,
        agent_default="rl_games_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path, or latest/best.")
    parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
    parser.add_argument("--wandb-project-name", type=str, default=None, help="Weights and Biases project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Weights and Biases entity (team).")
    parser.add_argument("--wandb-name", type=str, default=None, help="Weights and Biases run name.")
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Track this experiment with Weights and Biases.",
    )
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    set_hydra_args(hydra_args)
    return args_cli


def _resolve_checkpoint(args_cli: argparse.Namespace, agent_cfg: dict, log_root_path: str) -> str | None:
    """Resolve the checkpoint to resume from, or None when training starts from scratch."""
    if args_cli.checkpoint is None:
        return None
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        config_name = agent_cfg["params"]["config"]["name"]
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="rl_games",
            task=args_cli.task,
            checkpoint_pattern=r".*\.pth",
            other_dirs=["nn"],
            preferred_checkpoint_pattern=rf"{re.escape(config_name)}\.pth",
            metadata={"agent": args_cli.agent},
        )
    return retrieve_file_path(args_cli.checkpoint)


def run(argv: list[str]) -> None:
    """Train an RL-Games agent."""
    args_cli = _parse_args(argv)
    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
        pre_launch_video_config(env_cfg, args_cli)
        show_run_summary(screen, args_cli, env_cfg, library="rl_games", action="train")
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli):
            apply_env_overrides(args_cli, env_cfg)
            validate_distributed_device(args_cli)

            params = agent_cfg["params"]
            config = params["config"]
            args_cli.seed = resolve_seed(args_cli.seed)
            if args_cli.seed is not None:
                params["seed"] = args_cli.seed
            if args_cli.max_iterations is not None:
                config["max_epochs"] = args_cli.max_iterations
            if args_cli.distributed:
                params["seed"] += int(os.getenv("RANK", "0"))
                config["device"] = env_cfg.sim.device
                config["device_name"] = env_cfg.sim.device
                config["multi_gpu"] = True
            env_cfg.seed = params["seed"]

            config_name = config["name"]
            log_root_path = os.path.join("logs", "rl_games", config_name)
            if "pbt" in agent_cfg and agent_cfg["pbt"]["directory"] != ".":
                log_root_path = os.path.join(agent_cfg["pbt"]["directory"], log_root_path)
            else:
                log_root_path = os.path.abspath(log_root_path)
            print(f"[INFO] Logging experiment in directory: {log_root_path}")

            resume_path = _resolve_checkpoint(args_cli, agent_cfg, log_root_path)
            if resume_path is not None:
                params["load_checkpoint"] = True
                params["load_path"] = resume_path
                print(f"[INFO]: Loading model checkpoint from: {resume_path}")

            run_name = config.get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            config["train_dir"] = log_root_path
            config["full_experiment_name"] = run_name
            log_dir = os.path.join(log_root_path, run_name)
            write_run_manifest(log_dir, library="rl_games", task=args_cli.task, metadata={"agent": args_cli.agent})
            dump_train_configs(log_dir, env_cfg, agent_cfg)
            print(f"Exact experiment name requested from command line: {log_dir}")

            env_cfg.log_dir = log_dir
            apply_video_recording(env_cfg, log_dir, args_cli)

            log_training_asset_paths(args_cli.task, env_cfg, "training start (before environment creation)")

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )
            env = wrap_sensor_capture(env, log_dir, args_cli)

            screen.stage("Preparing agent")
            start_time = time.time()
            report_activity("Wrapping environment")
            env = RlGamesVecEnvWrapper.from_agent_cfg(env, agent_cfg)
            register_rl_games_env(env)
            config["num_actors"] = env.unwrapped.num_envs
            report_activity(None)

            report_activity("Building policy")
            if "pbt" in agent_cfg and agent_cfg["pbt"]["enabled"]:
                runner = Runner(MultiObserver([IsaacAlgoObserver(), PbtAlgoObserver(agent_cfg, args_cli)]))
            else:
                runner = Runner(IsaacAlgoObserver())
            report_activity(None)

            # configure_seed must run after Runner() so torch determinism does not disturb its initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)
            runner.load(agent_cfg)
            runner.reset()

            if args_cli.track and int(os.getenv("RANK", "0")) == 0:
                if args_cli.wandb_entity is None:
                    raise ValueError("Weights and Biases entity must be specified for tracking.")
                # wandb is an optional dependency of experiment tracking
                import wandb

                wandb.init(
                    project=args_cli.wandb_project_name or config_name,
                    entity=args_cli.wandb_entity,
                    name=args_cli.wandb_name or run_name,
                    sync_tensorboard=True,
                    monitor_gym=True,
                    save_code=True,
                )
                if not wandb.run.resumed:
                    wandb.config.update({"env_cfg": env_cfg.to_dict()})
                    wandb.config.update({"agent_cfg": agent_cfg})

            train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None
            run_args = {"train": True, "play": False, "sigma": train_sigma}
            if resume_path is not None:
                run_args["checkpoint"] = resume_path

            screen.close()
            try:
                with contextlib.suppress(KeyboardInterrupt):
                    runner.run(run_args)
                    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
            finally:
                log_training_asset_paths(args_cli.task, env_cfg, "training end (after training loop)")
                env.close()
