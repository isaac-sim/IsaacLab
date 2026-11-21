# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the reference model-based actor-critic on a native motion-tracking task."""

import argparse
import json
import os
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import to_dict

from isaaclab_rl.model_based.rollout import PhysicalRollout

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

from ...model_based.runner import ModelBasedRunner
from ..common import (
    add_common_train_args,
    apply_env_overrides,
    apply_video_recording,
    create_isaaclab_env,
    dump_train_configs,
    enable_cameras_for_video,
    pre_launch_video_config,
    scoped_torch_backend_flags,
    wrap_sensor_capture,
    write_run_manifest,
)


def run(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_train_args(
        parser,
        agent_default="model_based_cfg_entry_point",
        agent_help="Model-based learning configuration.",
    )
    parser.add_argument("--log_dir", help="Directory for checkpoints and metrics.jsonl.")
    parser.add_argument("--motion_dir", help="Reference clips at the task's configured source frame rate.")
    parser.add_argument("--asset_path", help="Matching USD articulation.")
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser, argv)
    if args.frontend != "torch":
        parser.error("The model-based learner requires the Torch environment frontend.")
    env_cfg, cfg = resolve_task_config(args.task, args.agent, overrides=overrides)
    apply_env_overrides(args, env_cfg)
    if args.seed is not None:
        cfg.seed = args.seed
    env_cfg.seed = cfg.seed
    if args.max_iterations is not None:
        cfg.max_iterations = args.max_iterations
    if args.motion_dir:
        env_cfg.commands.motion.motion_dir = args.motion_dir
    if args.asset_path:
        env_cfg.scene.robot.spawn.usd_path = args.asset_path
    env_cfg.commands.motion.sampling_mode = "external" if cfg.sampling_mode == "return" else cfg.sampling_mode
    if cfg.max_iterations < 1 or min(cfg.log_interval, cfg.save_interval, cfg.wm.horizon) < 1:
        parser.error("Iteration counts, logging intervals, and the model horizon must be positive.")
    with _distributed_context(args, env_cfg, cfg) as (rank, world_size):
        _train(args, env_cfg, cfg, rank, world_size)


@contextmanager
def _distributed_context(args, env_cfg, cfg):
    """Own the process group and keep collection batches and motion shards local."""
    if not args.distributed:
        yield 0, 1
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    args.device = env_cfg.sim.device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    cfg.seed += rank
    env_cfg.seed = cfg.seed
    env_cfg.commands.motion.motion_shard_index = rank
    env_cfg.commands.motion.motion_shard_count = world_size
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=torch.device(args.device))
    try:
        yield rank, world_size
    finally:
        dist.destroy_process_group()


def _train(args, env_cfg, cfg, rank: int, world_size: int) -> None:
    timestamp = args.run_timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output = Path(args.log_dir or Path("logs/model_based") / timestamp)
    if rank:
        output /= f"rank_{rank}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"task": args.task, "agent": args.agent, "env_cfg": to_dict(env_cfg), "agent_cfg": to_dict(cfg)}
    metadata["distributed"] = {
        "world_size": world_size,
        "num_envs_per_rank": env_cfg.scene.num_envs,
        "wm_batch_size_per_rank": cfg.wm.batch_size,
        "policy_batch_size_per_rank": cfg.policy.batch_size,
    }
    dump_train_configs(str(output), env_cfg, cfg)
    enable_cameras_for_video(args)
    pre_launch_video_config(env_cfg, args)
    apply_video_recording(env_cfg, str(output), args)
    write_run_manifest(str(output), library="model_based", task=args.task, metadata={"agent": args.agent})
    logging = nullcontext(None)
    if cfg.wandb_project and rank == 0:
        import wandb

        logging = wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, dir=str(output), config=metadata)
    torch.manual_seed(cfg.seed)
    with (
        logging as run,
        scoped_torch_backend_flags(
            cuda_matmul_allow_tf32=cfg.allow_tf32,
            cudnn_allow_tf32=True,
            cudnn_deterministic=False,
            cudnn_benchmark=False,
        ),
        launch_simulation(env_cfg, args),
    ):
        env = create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
        env = wrap_sensor_capture(env, str(output), args)
        runner = None
        try:
            adapter = PhysicalRollout(env, cfg)
            runner = ModelBasedRunner(adapter, cfg)
            adapter.reset()
            start = time.monotonic()
            with (output / "metrics.jsonl").open("a") as metrics:
                for _ in range(cfg.max_iterations):
                    logs = runner.step()
                    if runner.iteration == 1 or runner.iteration % cfg.log_interval == 0:
                        if world_size > 1:
                            logs = _average_logs(logs)
                        row = {key: float(value.detach()) for key, value in logs.items()}
                        row.update(
                            iteration=runner.iteration,
                            learning_updates=runner.learning_updates,
                            transitions=runner.iteration * adapter.num_envs * world_size,
                            world_size=world_size,
                            rank=rank,
                            seconds=time.monotonic() - start,
                            learning_rate=float(runner.world_model_learner.optimizers[0].param_groups[0]["lr"]),
                        )
                        if torch.device(adapter.device).type == "cuda":
                            row["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(adapter.device)
                        metrics.write(json.dumps(row) + "\n")
                        metrics.flush()
                        if rank == 0:
                            print(row, flush=True)
                        if run:
                            run.log(row, step=runner.iteration)
                    if rank == 0 and (
                        runner.iteration % cfg.save_interval == 0 or runner.iteration == cfg.max_iterations
                    ):
                        destination = output / f"model_{runner.iteration:06d}.pt"
                        checkpoint = runner.checkpoint() | metadata
                        # Playback evaluates the complete reference corpus, independently of training rank.
                        checkpoint["env_cfg"]["commands"]["motion"].update(motion_shard_index=0, motion_shard_count=1)
                        torch.save(checkpoint, destination.with_suffix(".tmp"))
                        destination.with_suffix(".tmp").replace(destination)
        finally:
            if runner is not None:
                runner.close()
            env.close()


def _average_logs(logs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reduce diagnostics without modifying tensors retained by captured updates."""
    keys = sorted(logs)
    values = torch.stack([logs[key].detach().float() for key in keys])
    episodes = keys.index("Train/completed_episodes")
    returns = keys.index("Train/normalized_episode_return")
    fallback = values[returns].clone()
    values[returns] *= values[episodes]
    dist.all_reduce(values)
    dist.all_reduce(fallback)
    values[returns] = torch.where(
        values[episodes] > 0, values[returns] / values[episodes].clamp_min(1), fallback / dist.get_world_size()
    )
    for index, key in enumerate(keys):
        if index not in (episodes, returns):
            values[index] /= dist.get_world_size()
    return dict(zip(keys, values, strict=True))
