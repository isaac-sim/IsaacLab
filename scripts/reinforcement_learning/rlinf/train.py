# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RLinf.

This script launches RLinf distributed training for IsaacLab tasks.
Tasks can be either:
1. Registered in IsaacLab with `rlinf_cfg_entry_point` - will be auto-registered into RLinf
2. Already registered in RLinf's REGISTER_ISAACLAB_ENVS

Usage:
    # Train an IsaacLab task (config YAML in the same directory as train.py)
    python train.py --config_name isaaclab_ppo_gr00t_assemble_trocar

    # Train with config YAML in a custom directory
    python train.py --config_path /path/to/config/dir \\
        --config_name isaaclab_ppo_gr00t_assemble_trocar

    # Train with task override and custom settings
    python train.py --config_name isaaclab_ppo_gr00t_assemble_trocar \\
        --task Isaac-Assemble-Trocar-G129-Dex3-RLinf-v0 --num_envs 64 --max_epochs 1000

Note:
    RLinf training requires a pretrained VLA model (e.g., GR00T, OpenVLA).
    The model_path should point to a HuggingFace format checkpoint directory.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
# required for RLinf to register IsaacLab tasks and converters
os.environ.setdefault("RLINF_EXT_MODULE", "isaaclab_contrib.rl.rlinf.extension")

# local imports
import cli_args  # noqa: E402  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RLinf.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment (overrides config if set)")
parser.add_argument("--max_epochs", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--list_tasks", action="store_true", default=False, help="List all available tasks and exit.")
parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model checkpoint (required).")

# append RLinf cli arguments
cli_args.add_rlinf_args(parser)
args_cli = parser.parse_args()

# Resolve config path and name from CLI args
if not args_cli.config_name:
    parser.error("--config_name is required (e.g. --config_name isaaclab_ppo_gr00t_assemble_trocar)")
config_dir = args_cli.config_path or str(SCRIPT_DIR)
config_name = args_cli.config_name
os.environ["RLINF_CONFIG_FILE"] = str(Path(config_dir) / f"{config_name}.yaml")

# Add config dir to PYTHONPATH so that Ray rollout workers can resolve
# data_config_class references like "gr00t_config:IsaacLabDataConfig"
if config_dir not in os.environ.get("PYTHONPATH", ""):
    os.environ["PYTHONPATH"] = config_dir + os.pathsep + os.environ.get("PYTHONPATH", "")

# Handle --list_tasks before any heavy imports
if args_cli.list_tasks:
    print("\n" + "=" * 60)
    print("Available RLinf Tasks")
    print("=" * 60)

    # List RLinf registered tasks
    print("\n[RLinf Registered Tasks]")
    try:
        from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

        for task_id in sorted(REGISTER_ISAACLAB_ENVS.keys()):
            print(f"  - {task_id}")
    except ImportError:
        print("  (Could not import RLinf registry)")

    print("\n" + "=" * 60)
    sys.exit(0)

"""Rest of the script - launch RLinf training."""
import rlinf  # noqa: F401
import torch.multiprocessing as mp  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from omegaconf import open_dict  # noqa: E402
from rlinf.config import validate_cfg  # noqa: E402
from rlinf.runners.embodied_runner import EmbodiedRunner  # noqa: E402
from rlinf.scheduler import Cluster  # noqa: E402
from rlinf.utils.placement import HybridComponentPlacement  # noqa: E402
from rlinf.workers.env.env_worker import EnvWorker  # noqa: E402
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker  # noqa: E402

logger = logging.getLogger(__name__)

mp.set_start_method("spawn", force=True)


def main():
    """Launch RLinf training."""
    print(f"[INFO] Using config: {config_name}")
    print(f"[INFO] Config path: {config_dir}")

    # Initialize Hydra and load config
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_dir, version_base="1.1")
    cfg = compose(config_name=config_name)

    # Get task_id from config
    task_id = cfg.env.train.init_params.id
    print(f"[INFO] Task: {task_id}")

    # Setup logging directory
    timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    log_dir = SCRIPT_DIR / "logs" / "rlinf" / f"{timestamp}-{task_id.replace('/', '_')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    # Apply runtime overrides from CLI arguments
    with open_dict(cfg):
        cfg.runner.logger.log_path = str(log_dir)

        # Override task if provided via CLI
        if args_cli.task:
            cfg.env.train.init_params.id = args_cli.task
            cfg.env.eval.init_params.id = args_cli.task

        # Override from CLI if provided
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

    # Validate config
    cfg = validate_cfg(cfg)

    # Print config summary
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

    # Create cluster and component placement
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create actor worker
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

    # Create rollout worker
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # Create env worker
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(cluster, name=cfg.env.group_name, placement_strategy=env_placement)

    # Create and run training
    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
