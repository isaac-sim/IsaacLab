# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate a trained RLinf agent.

This script runs evaluation using RLinf's distributed infrastructure,
which is required for VLA model inference.

Usage:
    # Evaluate a trained checkpoint (config YAML in the same directory as play.py)
    python play.py --config_name isaaclab_ppo_gr00t_assemble_trocar \\
        --model_path /path/to/checkpoint

    # Evaluate with config YAML in a custom directory
    python play.py --config_path /path/to/config/dir \\
        --config_name isaaclab_ppo_gr00t_assemble_trocar --model_path /path/to/checkpoint

    # Evaluate with video recording
    python play.py --config_name isaaclab_ppo_gr00t_assemble_trocar \\
        --model_path /path/to/checkpoint --video

Note:
    Evaluation requires the full RLinf infrastructure since VLA models
    are too large to run on a single GPU without FSDP.
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
# required for RLinf to register IsaacLab tasks and converters
os.environ.setdefault("RLINF_EXT_MODULE", "isaaclab_contrib.rl.rlinf.extension")

# local imports
import cli_args  # noqa: E402  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a trained RLinf agent.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task (overrides YAML config if set).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment (overrides config if set)")
parser.add_argument(
    "--model_path", type=str, default=None, help="Path to the model checkpoint (optional, can be set in config)."
)
parser.add_argument(
    "--num_episodes", type=int, default=None, help="Number of evaluation episodes (overrides config if set)."
)
parser.add_argument("--video", action="store_true", default=False, help="Enable video recording.")
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


"""launch RLinf evaluation."""
import rlinf  # noqa: F401
import torch.multiprocessing as mp  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from omegaconf import open_dict  # noqa: E402
from rlinf.config import validate_cfg  # noqa: E402
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner  # noqa: E402
from rlinf.scheduler import Cluster  # noqa: E402
from rlinf.utils.placement import HybridComponentPlacement  # noqa: E402
from rlinf.workers.env.env_worker import EnvWorker  # noqa: E402
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker  # noqa: E402

logger = logging.getLogger(__name__)

mp.set_start_method("spawn", force=True)


def main():
    """Launch RLinf evaluation."""
    print(f"[INFO] Using config: {config_name}")
    print(f"[INFO] Config path: {config_dir}")

    # Initialize Hydra and load config
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_dir, version_base="1.1")
    cfg = compose(config_name=config_name)

    # Get task_id from config (eval task)
    task_id = cfg.env.eval.init_params.id
    print(f"[INFO] Task: {task_id}")

    # Setup logging directory
    timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    log_dir = SCRIPT_DIR / "logs" / "rlinf" / "eval" / f"{timestamp}-{task_id.replace('/', '_')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    # Apply runtime overrides
    with open_dict(cfg):
        # Set evaluation mode
        cfg.runner.only_eval = True
        # Set logging
        cfg.runner.logger.log_path = str(log_dir)

        # Override checkpoint if provided via CLI
        if args_cli.model_path:
            cfg.rollout.model.model_path = args_cli.model_path

        # Enable video saving if requested
        if args_cli.video:
            cfg.env.eval.video_cfg.save_video = True
            cfg.env.eval.video_cfg.video_base_dir = str(log_dir / "videos")

        # Override task if provided via CLI
        if args_cli.task:
            cfg.env.eval.init_params.id = args_cli.task
            cfg.env.train.init_params.id = args_cli.task

        # Apply CLI args
        if args_cli.num_envs is not None:
            cfg.env.eval.total_num_envs = args_cli.num_envs
        if args_cli.seed is not None:
            cfg.actor.seed = args_cli.seed
        if args_cli.num_episodes is not None:
            cfg.algorithm.eval_rollout_epoch = args_cli.num_episodes

    # Validate config
    cfg = validate_cfg(cfg)

    # Print config summary
    print("\n" + "=" * 60)
    print("RLinf Evaluation Configuration")
    print("=" * 60)
    print(f"  Task: {cfg.env.eval.init_params.id}")
    print(f"  Num envs: {cfg.env.eval.total_num_envs}")
    print(f"  Model: {cfg.rollout.model.model_path}")
    print(f"  Videos: {cfg.env.eval.video_cfg.save_video}")
    if cfg.env.eval.video_cfg.save_video:
        print(f"  Video dir: {cfg.env.eval.video_cfg.video_base_dir}")
    print(f"  Log dir: {log_dir}")
    print("=" * 60 + "\n")

    # Create cluster and workers
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create rollout worker
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # Create env worker
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(cluster, name=cfg.env.group_name, placement_strategy=env_placement)

    # Run evaluation
    runner = EmbodiedEvalRunner(
        cfg=cfg,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
