# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""skrl training backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from datetime import datetime

import skrl

from isaaclab.app import add_launcher_args, launch_simulation, report_activity
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
from isaaclab_tasks.utils.training_asset_log import log_training_asset_paths

from ...skrl import (
    SkrlVecEnvWrapper,
    check_skrl_version,
    import_skrl_runner,
    resolve_skrl_agent_cfg_entry_point,
    resolve_skrl_algorithm,
)
from ..common import (
    CHECKPOINT_SELECTORS,
    add_common_train_args,
    apply_env_overrides,
    apply_video_recording,
    create_isaaclab_env,
    dump_train_configs,
    enable_cameras_for_video,
    pre_launch_video_config,
    preserve_attribute,
    resolve_checkpoint_selector,
    resolve_seed,
    set_hydra_args,
    show_run_summary,
    startup_screen,
    validate_distributed_device,
    wrap_sensor_capture,
    write_run_manifest,
)
from . import cli_args_skrl as cli_args

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse skrl training arguments."""
    parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
    add_common_train_args(
        parser,
        agent_default=None,
        agent_help="Agent configuration entry point (default: the task's canonical SKRL configuration).",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path, or latest/best.")
    cli_args.add_skrl_args(parser)
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    set_hydra_args(hydra_args)
    return args_cli


def _distributed_rank(args_cli: argparse.Namespace) -> int:
    """Return the global distributed rank for the selected skrl ML framework."""
    if args_cli.ml_framework == "jax":
        return int(os.getenv("JAX_RANK", "0"))
    return int(os.getenv("RANK", "0"))


def run(argv: list[str]) -> None:
    """Train a skrl agent while restoring the caller's global skrl settings."""
    args_cli = _parse_args(argv)
    with contextlib.ExitStack() as cleanup:
        if args_cli.ml_framework == "jax":
            cleanup.enter_context(preserve_attribute(skrl.config.jax, "backend"))
            skrl.config.jax.backend = "jax"
        _run(args_cli)


def _run(args_cli: argparse.Namespace) -> None:
    """Execute skrl training with parsed arguments."""
    check_skrl_version()
    agent_cfg_entry_point = resolve_skrl_agent_cfg_entry_point(args_cli.agent, args_cli.algorithm)
    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, agent_cfg_entry_point)
        algorithm = resolve_skrl_algorithm(agent_cfg, args_cli.algorithm)
        pre_launch_video_config(env_cfg, args_cli)
        show_run_summary(screen, args_cli, env_cfg, library="skrl", action="train")
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli):
            runner_cls = import_skrl_runner(args_cli.ml_framework)
            apply_env_overrides(args_cli, env_cfg)
            validate_distributed_device(args_cli)

            if args_cli.max_iterations:
                agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
            agent_cfg["trainer"]["close_environment_at_exit"] = False
            args_cli.seed = resolve_seed(args_cli.seed)
            if args_cli.seed is not None:
                agent_cfg["seed"] = args_cli.seed
            if args_cli.distributed:
                agent_cfg["seed"] += _distributed_rank(args_cli)
            env_cfg.seed = agent_cfg["seed"]

            experiment_cfg = agent_cfg["agent"]["experiment"]
            log_root_path = os.path.abspath(os.path.join("logs", "skrl", experiment_cfg["directory"]))
            print(f"[INFO] Logging experiment in directory: {log_root_path}")
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
            print(f"Exact experiment name requested from command line: {run_name}")
            if experiment_cfg["experiment_name"]:
                run_name += f"_{experiment_cfg['experiment_name']}"
            experiment_cfg["directory"] = log_root_path
            experiment_cfg["experiment_name"] = run_name
            log_dir = os.path.join(log_root_path, run_name)
            manifest_metadata = {
                "agent": agent_cfg_entry_point,
                "algorithm": algorithm,
                "ml_framework": args_cli.ml_framework,
            }
            write_run_manifest(log_dir, library="skrl", task=args_cli.task, metadata=manifest_metadata)
            dump_train_configs(log_dir, env_cfg, agent_cfg)

            if args_cli.checkpoint in CHECKPOINT_SELECTORS:
                resume_path = resolve_checkpoint_selector(
                    log_root_path,
                    args_cli.checkpoint,
                    library="skrl",
                    task=args_cli.task,
                    checkpoint_pattern=r".*",
                    other_dirs=["checkpoints"],
                    metadata=manifest_metadata,
                )
            else:
                resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

            env_cfg.log_dir = log_dir
            apply_video_recording(env_cfg, log_dir, args_cli)

            log_training_asset_paths(args_cli.task, env_cfg, "training start (before environment creation)")

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg) and algorithm == "ppo",
            )
            env = wrap_sensor_capture(env, log_dir, args_cli)

            screen.stage("Preparing agent")
            start_time = time.time()
            report_activity("Wrapping environment")
            env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
            report_activity(None)
            report_activity("Building policy")
            runner = runner_cls(env, agent_cfg)
            report_activity(None)

            # configure_seed must run after Runner() so torch determinism does not disturb its initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)
            if resume_path:
                print(f"[INFO] Loading model checkpoint from: {resume_path}")
                runner.agent.load(resume_path)

            screen.close()
            try:
                with contextlib.suppress(KeyboardInterrupt):
                    runner.run()
                    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
                    total_timesteps = agent_cfg["trainer"]["timesteps"]
                    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
                    runner.agent.write_checkpoint(timestep=total_timesteps, timesteps=total_timesteps)
                    print(f"[INFO] Saved final agent checkpoint to: {log_dir}/checkpoints")
            finally:
                log_training_asset_paths(args_cli.task, env_cfg, "training end (after training loop)")
                env.close()
