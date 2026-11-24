# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the neural robot dynamics model of a task on the dataset that ``record.py`` wrote.

The training settings come from the settings that the task names as ``nerd_recipe_cfg_entry_point``, an
:class:`isaaclab_neural.NeRDRecipeCfg`. The trained model is saved where the task's physics preset loads it, so the
task runs on it afterwards, for example with ``physics=newton_nerd``. No simulation starts.

Settings of the training can be set on the command line as ``agent.train.<name>=value``, for example
``agent.train.batch_size=256``. The other settings of the recipe shape the model and its samples, which ``record.py``
made, so the script stops when they differ from those the dataset was recorded with.

Run from the repository root:

    uv run --extra newton-neural python scripts/neural/train.py --task Isaac-Reach-Franka
"""

import argparse
import itertools
import time
from pathlib import Path

import gymnasium as gym
import torch
import yaml
from isaaclab_neural import NeRDRecipeCfg, NeRDTrainCfg
from isaaclab_neural.physics import MJWarpNeRDSolverCfg, NeRDSolverCfg
from newton_neural.data import TrajectoryWindows
from newton_neural.evaluation import rollout_error
from newton_neural.models import NeRDBase
from newton_neural.training import LossOneStep, create_optimizer

from isaaclab.utils import to_dict
from isaaclab.utils.io import load_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

RECIPE_ENTRY_POINT = "nerd_recipe_cfg_entry_point"
LOG_INTERVAL = 500  # Optimizer steps between two printed losses


def changed_settings(recipe_cfg: NeRDRecipeCfg, dataset: Path) -> list[str]:
    """Name the settings of ``recipe_cfg`` that differ from those ``dataset`` was recorded with.

    The sizes of the recording, the physics it came from and the settings of the training are left out. The
    recording decides the first two, and none of them shapes the model.
    """

    def plain(value: object) -> object:
        # A tuple and a list of the same values are the same setting.
        if isinstance(value, dict):
            return {key: plain(entry) for key, entry in value.items()}
        return [plain(entry) for entry in value] if isinstance(value, (list, tuple)) else value

    recorded = plain(load_yaml(str(dataset / "params" / "recipe.yaml")))
    # The settings pass through YAML as the recorded ones did, so that both sides name classes the same way.
    settings = plain(yaml.full_load(yaml.dump(to_dict(recipe_cfg))))
    return [
        name
        for name, value in settings.items()
        if name not in ("record", "reference_preset", "train") and recorded.get(name) != value
    ]


def check_report(nerd: NeRDBase, held_out: TrajectoryWindows, cfg: NeRDTrainCfg) -> None:
    """Check before the training that ``held_out`` can give the report that follows it.

    Raises:
        ValueError: If the model rolls out alone and ``cfg.horizons`` is empty, holds a negative frame count or is
            longer than the recorded episodes.
    """
    if not nerd.prediction.integrates:
        return
    if not cfg.horizons or min(cfg.horizons) < 0:
        raise ValueError(
            f"the report rolls the model out over the frame counts in train.horizons, got {tuple(cfg.horizons)}. "
            "Name at least one frame count of zero or more, such as agent.train.horizons=[1,10]"
        )
    history = 0 if nerd.receptive_field is None else nerd.receptive_field - 1
    frames = history + max(cfg.horizons)
    if len(held_out.with_length(frames)) == 0:
        raise ValueError(
            f"the report rolls the model out over {max(cfg.horizons)} frames after {history} frames of history, and "
            f"the held out robots have no episode of {frames + 1} frames. Set shorter agent.train.horizons, or "
            "record longer episodes with agent.record.episode_frames of record.py"
        )


def fit(nerd: NeRDBase, windows: TrajectoryWindows, cfg: NeRDTrainCfg) -> None:
    """Fit the normalization of ``nerd`` to ``windows`` and train it for ``cfg.max_steps`` optimizer steps."""
    nerd.fit_normalization(windows.loader(cfg.batch_size))
    loss = LossOneStep(noise=cfg.noise)
    if nerd.device.type == "cuda":
        loss.compile()  # The first step compiles the loss, which pays off over thousands of steps
    optimizer = create_optimizer(nerd, lr=cfg.learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.max_steps)
    print(f"Training for {cfg.max_steps} steps. On a GPU the first one compiles the loss and takes much longer.")
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


def report(nerd: NeRDBase, held_out: TrajectoryWindows, cfg: NeRDTrainCfg) -> None:
    """Print the error of ``nerd`` on the robots that it was not trained on."""
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
        errors = rollout_error(nerd, held_out, horizons=cfg.horizons, from_episode_start=from_episode_start)
        for horizon, error in errors.items():
            print(
                f"Started {where}, after {horizon * held_out.frame_dt:.2f} s alone the joint coordinates are off by "
                f"{error.p50['joint_q']:.4f} m or rad at the median, {error.p90['joint_q']:.4f} at the 90th "
                f"percentile and {error.p99['joint_q']:.4f} at the 99th."
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="Task whose robot the model stands for.")
    parser.add_argument("--dataset", type=Path, default=None, help="Dataset folder, next to the model by default.")
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="File of the trained model, the preset's by default."
    )
    parser.add_argument("--max_steps", type=int, default=None, help="Optimizer steps, the recipe's by default.")
    parser.add_argument("--device", default=None, help="Device to train on, a GPU when there is one.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the training.")
    args, overrides = setup_preset_cli(parser, argv)

    if RECIPE_ENTRY_POINT not in gym.spec(args.task).kwargs:
        parser.error(
            f"the task {args.task} registers no recipe for a neural robot dynamics model. Add an "
            f"isaaclab_neural.NeRDRecipeCfg to its gym registration as {RECIPE_ENTRY_POINT}"
        )
    try:
        _, recipe_cfg = resolve_task_config(args.task, RECIPE_ENTRY_POINT, overrides=overrides)
        target, _ = resolve_task_config(args.task, None, overrides=[f"physics={recipe_cfg.physics_preset}"])
    except ValueError as error:
        parser.error(str(error))
    solver_cfg = getattr(target.sim.physics, "solver_cfg", None)
    if not isinstance(solver_cfg, NeRDSolverCfg | MJWarpNeRDSolverCfg):
        parser.error(
            f"the recipe of {args.task} names the physics preset {recipe_cfg.physics_preset!r}, whose solver is "
            f"configured by a {type(solver_cfg).__name__}. Name a preset that runs a model, with a NeRDSolverCfg or a "
            "MJWarpNeRDSolverCfg"
        )
    preset_checkpoint = Path(solver_cfg.checkpoint)
    checkpoint = preset_checkpoint if args.checkpoint is None else args.checkpoint
    dataset = preset_checkpoint.parent / "dataset" if args.dataset is None else args.dataset
    untrained = dataset / preset_checkpoint.name
    if not untrained.is_file() or not (dataset / "params" / "recipe.yaml").is_file():
        parser.error(
            f"{dataset} holds no dataset of record.py, which comes with the untrained model {untrained.name} and "
            f"the settings params/recipe.yaml. Record it first, with scripts/neural/record.py --task {args.task}, "
            "or name its folder with --dataset"
        )
    changed = changed_settings(recipe_cfg, dataset)
    if changed:
        parser.error(
            f"the dataset in {dataset} was recorded with other values of {', '.join(changed)} than this run names. "
            "The model and its data follow these settings of the recipe, so record again with scripts/neural/"
            f"record.py --task {args.task}, or name the values that the dataset was recorded with"
        )
    cfg = recipe_cfg.train
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps

    torch.set_float32_matmul_precision("high")  # TF32 matrix products on GPUs with tensor cores
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device is None else args.device
    nerd = NeRDBase.load(untrained, device=device)
    windows, held_out = TrajectoryWindows(dataset, cfg.window, device=device).split()
    try:
        check_report(nerd, held_out, cfg)
    except ValueError as error:
        parser.error(str(error))
    fit(nerd, windows, cfg)
    nerd.save(checkpoint)  # Before the report, so that nothing after the training can lose the model
    report(nerd, held_out, cfg)
    print(f"The model is in {checkpoint.resolve()}. Train a policy on it with physics={recipe_cfg.physics_preset}.")


if __name__ == "__main__":
    main()
