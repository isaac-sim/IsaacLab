# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RLinf training backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from ..common import write_run_manifest
from . import cli_args_rlinf as cli_args


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse RLinf training arguments."""
    parser = argparse.ArgumentParser(description="Train an RL agent with RLinf.")
    cli_args.add_rlinf_args(parser)
    parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
    parser.add_argument("--list_tasks", action="store_true", default=False, help="List all available tasks and exit.")
    args_cli = parser.parse_args(argv)
    if not args_cli.list_tasks and not args_cli.config_name:
        parser.error("--config_name is required (e.g. --config_name isaaclab_ppo_gr00t_assemble_trocar)")
    return args_cli


def _list_tasks() -> None:
    """List the tasks registered with RLinf."""
    rule = "=" * 60
    print(f"\n{rule}\nAvailable RLinf Tasks\n{rule}\n\n[RLinf Registered Tasks]")
    try:
        # rlinf reads RLINF_EXT_MODULE at import time and is only needed for the listing
        from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS
    except ImportError:
        print("  (Could not import RLinf registry)")
    else:
        for task_id in sorted(REGISTER_ISAACLAB_ENVS):
            print(f"  - {task_id}")
    print(f"\n{rule}")


def run(argv: list[str]) -> None:
    """Launch RLinf training."""
    # Ray 2.47+ turns the current project into a ``working_dir`` runtime environment when the driver is
    # launched through ``uv run``. An Isaac Lab checkout commonly contains a large ``.venv`` and local
    # model checkpoints, which exceed Ray's 500 MiB upload limit. RLinf already selects the Python
    # executable for each worker, so this upload is neither needed nor desirable.
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    # required for RLinf to register Isaac Lab tasks and converters
    os.environ.setdefault("RLINF_EXT_MODULE", "isaaclab_contrib.rl.rlinf.extension")
    args_cli = _parse_args(argv)
    if args_cli.list_tasks:
        _list_tasks()
        return

    config_name = args_cli.config_name
    config_dir = cli_args.configure_rlinf_environment(config_name, args_cli.config_path)

    # rlinf reads RLINF_EXT_MODULE and RLINF_CONFIG_FILE at import time, so it is imported after the setup above
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
    # hyphens instead of colons in the time stamp; colons are invalid in Windows paths
    timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    log_dir = Path("logs") / "rlinf" / f"{timestamp}-{task_id.replace('/', '_')}"
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
        if args_cli.max_iterations is not None:
            cfg.runner.max_epochs = args_cli.max_iterations
        if args_cli.model_path is not None:
            cfg.actor.model.model_path = args_cli.model_path
            cfg.rollout.model.model_path = args_cli.model_path
        if args_cli.only_eval:
            cfg.runner.only_eval = True
        if args_cli.checkpoint:
            checkpoint_path = cli_args.resolve_rlinf_checkpoint(
                args_cli.checkpoint,
                log_root_path=str(Path("logs") / "rlinf"),
                task=args_cli.task or task_id,
                config_name=config_name,
            )
            cfg.runner.resume_dir = str(Path(checkpoint_path).parent)

    write_run_manifest(
        str(log_dir), library="rlinf", task=args_cli.task or task_id, metadata={"config_name": config_name}
    )
    cfg = validate_cfg(cfg)
    cli_args.print_rlinf_banner(
        "RLinf Training Configuration",
        {
            "Task": cfg.env.train.init_params.id,
            "Num envs": cfg.env.train.total_num_envs,
            "Max iterations": cfg.runner.max_epochs,
            "Model": cfg.actor.model.model_path,
            "Algorithm": cfg.algorithm.loss_type,
            "Log dir": log_dir,
        },
    )

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    if cfg.algorithm.loss_type == "embodied_sac":
        from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy as actor_worker_cls
    else:
        from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor as actor_worker_cls
    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=component_placement.get_strategy("actor")
    )
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=component_placement.get_strategy("rollout")
    )
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=component_placement.get_strategy("env")
    )

    runner = EmbodiedRunner(cfg=cfg, actor=actor_group, rollout=rollout_group, env=env_group)
    runner.init_workers()
    runner.run()
