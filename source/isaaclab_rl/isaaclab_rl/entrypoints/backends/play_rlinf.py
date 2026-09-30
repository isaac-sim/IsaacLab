# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RLinf evaluation backend of the unified reinforcement learning entrypoint.

Evaluation runs on RLinf's distributed infrastructure, which VLA model inference requires since the
models are too large to run on a single GPU without FSDP.

Usage:
    # Evaluate a trained checkpoint (config YAML discovered in the isaaclab_tasks package)
    uv run isaaclab play --rl_library rlinf --config_name isaaclab_ppo_gr00t_assemble_trocar \\
        --model_path /path/to/checkpoint

    # Evaluate with a config YAML in a custom directory
    uv run isaaclab play --rl_library rlinf --config_path /path/to/config/dir \\
        --config_name isaaclab_ppo_gr00t_assemble_trocar --model_path /path/to/checkpoint

    # Evaluate with video recording
    uv run isaaclab play --rl_library rlinf --config_name isaaclab_ppo_gr00t_assemble_trocar \\
        --model_path /path/to/checkpoint --video
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from . import cli_args_rlinf as cli_args


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse RLinf evaluation arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained RLinf agent.")
    cli_args.add_rlinf_args(parser)
    parser.add_argument(
        "--num_episodes", type=int, default=None, help="Number of evaluation episodes (overrides the config if set)."
    )
    parser.add_argument("--video", action="store_true", default=False, help="Enable video recording.")
    args_cli = parser.parse_args(argv)
    if not args_cli.config_name:
        parser.error("--config_name is required (e.g. --config_name isaaclab_ppo_gr00t_assemble_trocar)")
    return args_cli


def run(argv: list[str]) -> None:
    """Launch RLinf evaluation."""
    args_cli = _parse_args(argv)
    config_name = args_cli.config_name
    config_dir = cli_args.configure_rlinf_environment(config_name, args_cli.config_path)

    # rlinf reads RLINF_EXT_MODULE and RLINF_CONFIG_FILE at import time, so it is imported after the setup above
    import rlinf  # noqa: F401
    import torch.multiprocessing as mp
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import open_dict
    from rlinf.config import validate_cfg
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    mp.set_start_method("spawn", force=True)

    print(f"[INFO] Using config: {config_name}")
    print(f"[INFO] Config path: {config_dir}")
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_dir, version_base="1.1")
    cfg = compose(config_name=config_name)

    task_id = cfg.env.eval.init_params.id
    print(f"[INFO] Task: {task_id}")
    # hyphens instead of colons in the time stamp; colons are invalid in Windows paths
    timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    log_dir = Path("logs") / "rlinf" / "eval" / f"{timestamp}-{task_id.replace('/', '_')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    with open_dict(cfg):
        cfg.runner.only_eval = True
        cfg.runner.logger.log_path = str(log_dir)
        if args_cli.model_path:
            cfg.rollout.model.model_path = args_cli.model_path
        if args_cli.checkpoint:
            cfg.runner.eval_policy_path = cli_args.resolve_rlinf_checkpoint(
                args_cli.checkpoint,
                log_root_path=str(Path("logs") / "rlinf"),
                task=args_cli.task or task_id,
                config_name=config_name,
            )
        if args_cli.video:
            cfg.env.eval.video_cfg.save_video = True
            cfg.env.eval.video_cfg.video_base_dir = str(log_dir / "videos")
        if args_cli.task:
            cfg.env.eval.init_params.id = args_cli.task
            cfg.env.train.init_params.id = args_cli.task
        if args_cli.num_envs is not None:
            cfg.env.eval.total_num_envs = args_cli.num_envs
        if args_cli.seed is not None:
            cfg.actor.seed = args_cli.seed
        if args_cli.num_episodes is not None:
            cfg.algorithm.eval_rollout_epoch = args_cli.num_episodes

    cfg = validate_cfg(cfg)
    fields = {
        "Task": cfg.env.eval.init_params.id,
        "Num envs": cfg.env.eval.total_num_envs,
        "Model": cfg.rollout.model.model_path,
        "Checkpoint": cfg.runner.eval_policy_path,
        "Videos": cfg.env.eval.video_cfg.save_video,
    }
    if cfg.env.eval.video_cfg.save_video:
        fields["Video dir"] = cfg.env.eval.video_cfg.video_base_dir
    fields["Log dir"] = log_dir
    cli_args.print_rlinf_banner("RLinf Evaluation Configuration", fields)

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=component_placement.get_strategy("rollout")
    )
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=component_placement.get_strategy("env")
    )

    runner = EmbodiedEvalRunner(cfg=cfg, rollout=rollout_group, env=env_group)
    runner.init_workers()
    print("[INFO] Policy playback is running, press Ctrl+C to exit...")
    runner.run()
