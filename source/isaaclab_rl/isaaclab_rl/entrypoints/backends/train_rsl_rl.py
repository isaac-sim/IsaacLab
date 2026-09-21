# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL training backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from datetime import datetime

from isaaclab.app import add_launcher_args, launch_simulation, report_activity
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli
from isaaclab_tasks.utils.training_asset_log import log_training_asset_paths

from ...rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)
from ...utils.wandb import announce_new_run, is_wandb_checkpoint, resolve_wandb_checkpoint, resolve_wandb_entity
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
    scoped_torch_backend_flags,
    set_hydra_args,
    show_run_summary,
    startup_screen,
    validate_distributed_device,
    wrap_sensor_capture,
    write_run_manifest,
)
from . import cli_args_rsl_rl as cli_args

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse RSL-RL training arguments."""
    parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
    add_common_train_args(
        parser,
        agent_default="rsl_rl_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument(
        "--external_callback", default=None, help="Fully qualified path to an externally defined callback."
    )
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    remaining_args_env_registration = cli_args.register_external_tasks(argv)
    args_cli, remaining_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    set_hydra_args(list_intersection(remaining_args, remaining_args_env_registration))
    return args_cli


def _resolve_checkpoint(args_cli: argparse.Namespace, agent_cfg: RslRlBaseRunnerCfg, log_root_path: str) -> str | None:
    """Resolve the checkpoint to resume from, or None when training starts from scratch."""
    if args_cli.checkpoint and is_wandb_checkpoint(args_cli.checkpoint):
        return resolve_wandb_checkpoint(args_cli.checkpoint)
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="rsl_rl",
            task=args_cli.task,
            checkpoint_pattern=r"model_.*\.pt",
            metadata={"agent": args_cli.agent},
        )
    if args_cli.checkpoint and os.path.isdir(args_cli.checkpoint):
        return get_checkpoint_path(
            os.path.dirname(args_cli.checkpoint), os.path.basename(args_cli.checkpoint), agent_cfg.load_checkpoint
        )
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    if agent_cfg.algorithm.class_name == "Distillation":
        raise ValueError("Distillation training requires --checkpoint.")
    return None


def run(argv: list[str]) -> None:
    """Train an RSL-RL agent while restoring the caller's Torch backend settings."""
    args_cli = _parse_args(argv)
    with scoped_torch_backend_flags(
        cuda_matmul_allow_tf32=True,
        cudnn_allow_tf32=True,
        cudnn_deterministic=False,
        cudnn_benchmark=False,
    ):
        _run(args_cli)


def _run(args_cli: argparse.Namespace) -> None:
    """Execute RSL-RL training with parsed arguments."""
    installed_version = check_rsl_rl_version()

    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
        pre_launch_video_config(env_cfg, args_cli)
        show_run_summary(screen, args_cli, env_cfg, library="rsl_rl", action="train")
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli):
            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
            apply_env_overrides(args_cli, env_cfg)
            validate_distributed_device(args_cli)
            if args_cli.max_iterations is not None:
                agent_cfg.max_iterations = args_cli.max_iterations
            if args_cli.distributed:
                agent_cfg.device = env_cfg.sim.device
                agent_cfg.seed += int(os.getenv("RANK", "0"))
            env_cfg.seed = agent_cfg.seed

            log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            print(f"[INFO] Logging experiment in directory: {log_root_path}")
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            print(f"Exact experiment name requested from command line: {run_name}")
            if agent_cfg.run_name:
                run_name += f"_{agent_cfg.run_name}"
            log_dir = os.path.join(log_root_path, run_name)
            write_run_manifest(log_dir, library="rsl_rl", task=args_cli.task, metadata={"agent": args_cli.agent})

            resume_path = _resolve_checkpoint(args_cli, agent_cfg, log_root_path)
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
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            report_activity(None)
            report_activity("Building policy")
            runner = create_rsl_rl_runner(env, agent_cfg, log_dir=log_dir)
            report_activity(None)

            # configure_seed must run after runner construction so torch determinism does not disturb its initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)

            runner.add_git_repo_to_log(__file__)
            if resume_path is not None:
                print(f"[INFO]: Loading model checkpoint from: {resume_path}")
                runner.load(resume_path)
            dump_train_configs(log_dir, env_cfg, agent_cfg)

            if agent_cfg.logger == "wandb":
                announce_new_run(agent_cfg.wandb_project, resolve_wandb_entity())

            screen.close()
            try:
                with contextlib.suppress(KeyboardInterrupt):
                    runner.learn(
                        num_learning_iterations=agent_cfg.max_iterations,
                        init_at_random_ep_len=agent_cfg.init_at_random_ep_len,
                    )
                    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
            finally:
                log_training_asset_paths(args_cli.task, env_cfg, "training end (after training loop)")
                env.close()
