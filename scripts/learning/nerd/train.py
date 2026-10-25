# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fit temporal NeRD to native HDF5 episodes, with resumable checkpoints and optional W&B uploads."""

import argparse
import json
import os
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from isaaclab.utils import to_dict

from isaaclab_rl.nerd import NeRDTrainCfg, OfflineNeRD

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="JSON overrides of NeRDTrainCfg, including network geometry.")
    parser.add_argument("--max_updates", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, help="Resume a native checkpoint.")
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--save_seconds", type=float, default=120.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--wandb_project")
    parser.add_argument("--wandb_entity")
    parser.add_argument("--wandb_name")
    args = parser.parse_args()
    if min(args.save_interval, args.save_seconds, args.log_interval) <= 0:
        parser.error("Checkpoint and logging intervals must be positive.")
    task = json.loads(args.dataset[0].with_suffix(".json").read_text())["task"]
    task_cfg = load_cfg_from_registry(task, "nerd_cfg_entry_point")
    cfg = NeRDTrainCfg() if isinstance(task_cfg, dict) else task_cfg
    if isinstance(task_cfg, dict):
        cfg.from_dict(task_cfg)
    if args.config:
        cfg.from_dict(json.loads(args.config.read_text()))
    if args.max_updates is not None:
        cfg.max_updates = args.max_updates
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "last.pt"
    if target.exists() and not args.checkpoint:
        parser.error("Output contains last.pt; use --checkpoint to resume or choose another output.")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True) if args.checkpoint else None
    settings = to_dict(cfg)
    if saved:
        previous = saved["training_cfg"]
        if any(previous[key] != value for key, value in settings.items() if key not in ("max_updates", "compile")):
            parser.error("Resume requires unchanged optimization settings, split and model geometry.")
    learner = OfflineNeRD(args.dataset, cfg, args.device, checkpoint=saved)
    if learner.updates >= cfg.max_updates:
        parser.error("The requested update limit has already been reached.")
    logging = nullcontext(None)
    if args.wandb_project:
        import wandb

        identity = saved.get("wandb") if saved else None
        resume_identity = (
            identity
            and identity["project"] == args.wandb_project
            and (args.wandb_entity is None or identity["entity"] == args.wandb_entity)
        )
        logging = wandb.init(
            project=args.wandb_project,
            entity=identity["entity"] if resume_identity else args.wandb_entity,
            name=args.wandb_name,
            id=identity["id"] if resume_identity else None,
            resume="allow",
            config=settings,
            allow_val_change=True,
            dir=str(args.output),
        )
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    handlers = {signum: signal.signal(signum, stop) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        with logging as run, (args.output / "metrics.jsonl").open("a") as metrics:
            if run:
                run.define_metric("update")
                run.define_metric("*", step_metric="update")
            last_save = time.monotonic()
            while learner.updates < cfg.max_updates and not stopped:
                values = learner.update()
                checkpoint_due = (
                    learner.updates % args.save_interval == 0 or time.monotonic() - last_save >= args.save_seconds
                )
                finished = learner.updates == cfg.max_updates or stopped
                if learner.updates % args.log_interval == 0 or finished or checkpoint_due:
                    values |= learner.validate()
                    values["update"] = learner.updates
                    metrics.write(json.dumps(values) + "\n")
                    metrics.flush()
                    print(json.dumps(values), flush=True)
                    if run:
                        run.log(values)
                if checkpoint_due or finished:
                    checkpoint = learner.checkpoint()
                    checkpoint["training_cfg"] = settings
                    checkpoint["wandb"] = {"id": run.id, "project": run.project, "entity": run.entity} if run else None
                    temporary = target.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        torch.save(checkpoint, stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, target)
                    if run:
                        artifact = wandb.Artifact(f"nerd-{run.id}", type="model", metadata={"update": learner.updates})
                        artifact.add_file(str(target), name="last.pt")
                        logged = run.log_artifact(artifact, aliases=["latest", f"update-{learner.updates}"])
                        if run.settings.mode != "offline":
                            logged.wait(timeout=180)
                        run.log({"update": learner.updates, "checkpoint/durable_update": learner.updates})
                    last_save = time.monotonic()
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
