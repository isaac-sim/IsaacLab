# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RLinf training logic for the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

from common import import_local_module

logger = logging.getLogger(__name__)

RL_ROOT = Path(__file__).resolve().parents[1]
RLINF_DIR = RL_ROOT / "rlinf"
CLI_ARGS = import_local_module("isaaclab_rlinf_cli_args", RLINF_DIR / "cli_args.py")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse RLinf training arguments."""
    parser = argparse.ArgumentParser(description="Train an RL agent with RLinf.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed used for the environment (overrides config if set)",
    )
    parser.add_argument("--max_epochs", type=int, default=None, help="RL Policy training iterations.")
    parser.add_argument("--list_tasks", action="store_true", default=False, help="List all available tasks and exit.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model checkpoint (required).")
    CLI_ARGS.add_rlinf_args(parser)
    args_cli = parser.parse_args(argv)
    if not args_cli.list_tasks and not args_cli.config_name:
        parser.error("--config_name is required (e.g. --config_name isaaclab_ppo_gr00t_assemble_trocar)")
    return args_cli


def _list_tasks() -> None:
    """List available RLinf tasks."""
    print("\n" + "=" * 60)
    print("Available RLinf Tasks")
    print("=" * 60)

    print("\n[RLinf Registered Tasks]")
    try:
        from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

        for task_id in sorted(REGISTER_ISAACLAB_ENVS.keys()):
            print(f"  - {task_id}")
    except ImportError:
        print("  (Could not import RLinf registry)")

    print("\n" + "=" * 60)


def run(argv: list[str]) -> None:
    """Launch RLinf training."""
    os.environ.setdefault("RLINF_EXT_MODULE", "isaaclab_contrib.rl.rlinf.extension")
    args_cli = _parse_args(argv)

    if args_cli.list_tasks:
        _list_tasks()
        return

    config_name = args_cli.config_name
    config_dir = CLI_ARGS.resolve_config_dir(config_name, args_cli.config_path)
    os.environ["RLINF_CONFIG_FILE"] = str(Path(config_dir) / f"{config_name}.yaml")

    if config_dir not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = config_dir + os.pathsep + os.environ.get("PYTHONPATH", "")

    import rlinf  # noqa: F401
    import torch.multiprocessing as mp
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import open_dict
    from rlinf.config import validate_cfg
    from rlinf.runners.embodied_runner import EmbodiedRunner
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

    task_id = cfg.env.train.init_params.id
    print(f"[INFO] Task: {task_id}")

    # Use hyphens instead of colons in time — colons are invalid in Windows paths.
    timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    log_dir = RLINF_DIR / "logs" / "rlinf" / f"{timestamp}-{task_id.replace('/', '_')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    with open_dict(cfg):
        cfg.runner.logger.log_path = str(log_dir)

        if args_cli.task:
            cfg.env.train.init_params.id = args_cli.task
            cfg.env.eval.init_params.id = args_cli.task

        if args_cli.num_envs is not None:
            cfg.env.train.total_num_envs = args_cli.num_envs
            cfg.env.eval.total_num_envs = args_cli.num_envs
        if args_cli.seed is not None:
            cfg.actor.seed = args_cli.seed
        if args_cli.max_epochs is not None:
            cfg.runner.max_epochs = args_cli.max_epochs
        if args_cli.model_path is not None:
            cfg.actor.model.model_path = args_cli.model_path
            cfg.rollout.model.model_path = args_cli.model_path
        if args_cli.only_eval:
            cfg.runner.only_eval = True
        if args_cli.resume_dir:
            cfg.runner.resume_dir = args_cli.resume_dir

    cfg = validate_cfg(cfg)

    print("\n" + "=" * 60)
    print("RLinf Training Configuration")
    print("=" * 60)
    print(f"  Task: {cfg.env.train.init_params.id}")
    print(f"  Num envs: {cfg.env.train.total_num_envs}")
    print(f"  Max epochs: {cfg.runner.max_epochs}")
    print(f"  Model: {cfg.actor.model.model_path}")
    print(f"  Algorithm: {cfg.algorithm.loss_type}")
    print(f"  Log dir: {log_dir}")
    print("=" * 60 + "\n")

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    actor_placement = component_placement.get_strategy("actor")
    if cfg.algorithm.loss_type == "embodied_sac":
        from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

        actor_worker_cls = EmbodiedSACFSDPPolicy
    else:
        from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

        actor_worker_cls = EmbodiedFSDPActor

    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(cluster, name=cfg.env.group_name, placement_strategy=env_placement)

    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()
