# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fit the task's model definition to a recorded dataset.

The model is built on the robot that the dataset describes, so one dataset fits any architecture.
``model.<field>`` overrides configure the model definition and ``agent.<field>`` overrides configure
NeRDTrainingCfg. Pass --dataset and --checkpoint explicitly.
"""

# ruff: noqa: E402
# Configure Warp before importing simulation modules.

import argparse
import itertools
import time
from pathlib import Path

import warp as wp

wp.config.enable_backward = False

import torch
from isaaclab_neural import NeRDTrainingCfg
from isaaclab_neural.nerd import build_model, default_compile
from newton_neural.data import TrajectoryDataset, TrajectoryWindows
from newton_neural.evaluation import rollout_error
from newton_neural.models import NeRDBase
from newton_neural.training import LossOneStep, create_optimizer

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

MODEL_ENTRY_POINT = "nerd_cfg_entry_point"
TRAINING_ENTRY_POINT = "nerd_training_cfg_entry_point"
LOG_INTERVAL = 500  # Optimizer steps between two printed losses


def check_report(nerd: NeRDBase, held_out: TrajectoryWindows, cfg: NeRDTrainingCfg) -> None:
    """Check before the training that ``held_out`` can give the report that follows it.

    Raises:
        ValueError: If the model rolls out alone and ``cfg.horizons`` is empty, holds a negative frame count or is
            longer than the recorded episodes.
    """
    if not nerd.prediction.integrates:
        return
    if not cfg.horizons or min(cfg.horizons) < 0:
        raise ValueError(
            f"the report rolls the model out over the frame counts in agent.horizons, got {tuple(cfg.horizons)}. "
            "Name at least one frame count of zero or more, such as agent.horizons=[1,10]"
        )
    history = 0 if nerd.receptive_field is None else nerd.receptive_field - 1
    frames = history + max(cfg.horizons)
    if len(held_out.with_length(frames)) == 0:
        raise ValueError(
            f"the report rolls the model out over {max(cfg.horizons)} frames after {history} frames of history, and "
            f"the held out robots have no episode of {frames + 1} frames. Set shorter agent.horizons, or "
            "record longer episodes with env.actions.joint_targets.episode_frames of record.py"
        )


def fit(nerd: NeRDBase, windows: TrajectoryWindows, cfg: NeRDTrainingCfg, compiled: bool) -> None:
    """Fit the normalization of ``nerd`` to ``windows`` and train it for ``cfg.max_steps`` optimizer steps.

    With ``compiled``, the loss runs through ``torch.compile``.
    """
    nerd.fit_normalization(windows.loader(cfg.batch_size))
    loss = LossOneStep(noise=cfg.noise)
    if compiled:
        loss.compile()  # The first step compiles the loss, which pays off over thousands of steps
    optimizer = create_optimizer(nerd, lr=cfg.learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.max_steps)
    first_step = " The first one compiles the loss and takes much longer." if compiled else ""
    print(f"Training for {cfg.max_steps} steps.{first_step}")
    start = time.perf_counter()
    batches = itertools.islice(windows.loader(cfg.batch_size, epochs=None), cfg.max_steps)
    for step, batch in enumerate(batches, start=1):
        optimizer.zero_grad()
        error = loss(nerd, batch)
        error.backward()
        optimizer.step()
        schedule.step()
        if step % LOG_INTERVAL == 0 or step == cfg.max_steps:
            print(f"Step {step} of {cfg.max_steps}, loss {error.item():.4f}, {time.perf_counter() - start:.0f} s.")


def report(nerd: NeRDBase, held_out: TrajectoryWindows, cfg: NeRDTrainingCfg, compiled: bool) -> None:
    """Print the error of ``nerd`` on the robots that it was not trained on.

    With ``compiled``, the rollouts of the model run through ``torch.compile``.
    """
    if not nerd.prediction.integrates:
        # A model inside a solver gives no next state to roll out, so its one step error is what there is to report.
        loss = LossOneStep()
        batches = held_out.loader(min(cfg.batch_size, len(held_out)), shuffle=False)
        with torch.no_grad():
            error = torch.stack([loss(nerd, batch) for batch in batches]).mean()
        print(f"On held out samples the prediction is off by {error:.2%} of its spread on average.")
        return
    # Alone, the model starts either in the middle of an episode, with its history, or at a reset, without one.
    # A model with unbounded history cannot be given the frames before a rollout, so it starts at resets only.
    starts = {"in the middle of an episode": False, "at a reset": True}
    if nerd.receptive_field is None:
        del starts["in the middle of an episode"]
    for where, from_episode_start in starts.items():
        errors = rollout_error(
            nerd, held_out, horizons=cfg.horizons, from_episode_start=from_episode_start, compile=compiled
        )
        for horizon, error in errors.items():
            print(
                f"Started {where}, after {horizon * held_out.frame_dt:.2f} s alone the joint coordinates are off by "
                f"{error.p50['joint_q']:.4f} m or rad at the median, {error.p90['joint_q']:.4f} at the 90th "
                f"percentile and {error.p99['joint_q']:.4f} at the 99th."
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="Task whose robot the model stands for.")
    parser.add_argument("--dataset", type=Path, required=True, help="Recorded dataset folder.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Output checkpoint file.")
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Optimizer steps, the task's training configuration by default."
    )
    parser.add_argument("--device", default=None, help="Device to train on, a GPU when there is one.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the training.")
    args, overrides = setup_preset_cli(parser, argv)

    # The task resolver composes one configuration in its agent slot, so each definition is resolved on its own.
    model_overrides = [f"agent.{token.removeprefix('model.')}" for token in overrides if token.startswith("model.")]
    training_overrides = [token for token in overrides if not token.startswith("model.")]
    _, cfg = resolve_task_config(args.task, TRAINING_ENTRY_POINT, overrides=training_overrides)
    _, model_cfg = resolve_task_config(args.task, MODEL_ENTRY_POINT, overrides=model_overrides)
    checkpoint, dataset = args.checkpoint, args.dataset
    if not (dataset / "meta.json").is_file():
        parser.error(f"{dataset} holds no recorded dataset; record it first")
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps

    torch.set_float32_matmul_precision("high")  # TF32 matrix products on GPUs with tensor cores
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device is None else args.device
    windows, held_out = TrajectoryWindows(dataset, cfg.window, device=device).split()
    try:
        nerd = build_model(TrajectoryDataset(dataset), model_cfg).to(device)
        check_report(nerd, held_out, cfg)
    except ValueError as error:
        parser.error(str(error))
    compiled = default_compile(nerd.device)  # On a GPU, where PyTorch has the Triton that torch.compile needs
    fit(nerd, windows, cfg, compiled)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    nerd.save(checkpoint)  # Save before reporting so a report failure cannot lose trained weights.
    report(nerd, held_out, cfg, compiled)
    print(f"Saved {checkpoint.resolve()}")


if __name__ == "__main__":
    main()
