# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the FlashNeRD CartPole rollout error of an Isaac Lab Cartpole model with the study's protocol.

The run follows the CartPole configuration of the FlashNeRD main-results study
(PhysicsNeMo Robotics tag ``mnabian/flashnerd-benchmark-v1``):

1. ``data``: 10,000 training trajectories of 100 transitions and 2,048 held-out trajectories of 509
   transitions, recorded with MuJoCo Warp at 1/60 s from uniform initial states and uniform random
   efforts resampled every frame;
2. ``train``: ParallelNeRD with the frozen CartPole geometry, ten frames of history, global batch 4,096,
   and a linear 1e-3 to 1e-4 learning rate;
3. ``evaluate``: the prefix-pooled rollout NRMSE ``E_H`` over the held-out cohort, normalized by the
   training-state standard deviations, for H = 1, 15, 50, 100, 200, and 500.

The study trains for five million updates; ``--max-updates`` sets the budget of this run.
"""

import argparse
import logging
from pathlib import Path

import torch
from isaaclab_physicsnemo import (
    CONTROL_KEY,
    TrainingSettings,
    TrajectoryDatasetRecorder,
    UniformRandomActions,
    evaluate_rollout_nrmse,
    train_dynamics_artifact,
)
from isaaclab_physicsnemo.tasks import CARTPOLE_FLASHNERD_MODEL_CONFIG, CartpoleFlashNeRDBenchmarkEnvCfg
from physicsnemo_robotics.data import NewtonTrajectoryZarrReader
from physicsnemo_robotics.dynamics import ControlFieldEncoder, DynamicsArtifact, newton_state_codec

from isaaclab.sim import SimulationContext

from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv

HORIZONS = (1, 15, 50, 100, 200, 500)
# FlashNeRD in Table I of the paper; the study adds H = 50 and 500.
PAPER_FLASHNERD = {1: 4.4e-5, 15: 1.7e-4, 100: 3.2e-3, 200: 6.0e-2}
# The cart effort is 100 N per action unit and saturates at 400 N.
EFFORT_BOUND = 4.0


def record(path: Path, *, num_envs: int, transitions: int, episodes_per_env: int, seed: int, device: str) -> None:
    """Record ``num_envs * episodes_per_env`` episodes of ``transitions`` transitions each."""
    cfg = CartpoleFlashNeRDBenchmarkEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.seed = seed
    cfg.episode_length_s = (transitions + 0.5) * cfg.sim.dt
    env = CartpoleEnv(cfg=cfg)
    try:
        env.reset(seed=seed)
        actions = UniformRandomActions(num_envs, cfg.action_space, high=EFFORT_BOUND, device=env.device, seed=seed)
        dones = None
        with TrajectoryDatasetRecorder(
            env, path, scenario_id="flashnerd-cartpole", seed=seed, replace=True
        ) as recorder:
            with torch.inference_mode():
                for _ in range((transitions + 1) * episodes_per_env):
                    _, _, terminated, truncated, _ = env.step(actions(dones))
                    dones = terminated | truncated
                    recorder.record(dones)
        print(f"Wrote {recorder.episodes_written} episodes to {path}")
    finally:
        env.close()
    SimulationContext.clear_instance()


def scene_model(device: str):
    """Return the Newton model of the benchmark scene."""
    cfg = CartpoleFlashNeRDBenchmarkEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = device
    env = CartpoleEnv(cfg=cfg)
    return env, env.sim.physics_manager.get_model()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Directory of datasets, training, and results.")
    parser.add_argument(
        "--stages", nargs="+", default=["data", "train", "evaluate"], choices=["data", "train", "evaluate"]
    )
    parser.add_argument("--max-updates", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    train_data, evaluation_data = args.output / "train", args.output / "evaluation"

    if "data" in args.stages:
        record(train_data, num_envs=2000, transitions=100, episodes_per_env=5, seed=7, device=args.device)
        record(evaluation_data, num_envs=2048, transitions=509, episodes_per_env=1, seed=8, device=args.device)

    if "train" in args.stages:
        env, model = scene_model(args.device)
        try:
            summary = train_dynamics_artifact(
                train_data,
                model,
                args.output / "training",
                settings=TrainingSettings(
                    history=10,
                    max_updates=args.max_updates,
                    batch_size=args.batch_size,
                    learning_rate=1.0e-3,
                    final_learning_rate=1.0e-4,
                    validation_fraction=0.1,
                    log_every=1000,
                    seed=20260813,
                    architecture="parallel_nerd",
                    model_kwargs=CARTPOLE_FLASHNERD_MODEL_CONFIG,
                    rollout_horizons=(),
                ),
                device=args.device,
            )
        finally:
            env.close()
        SimulationContext.clear_instance()
        print(f"Trained {summary.parameter_updates} updates, validation {summary.validation}")

    if "evaluate" in args.stages:
        env, model = scene_model(args.device)
        try:
            codec = newton_state_codec(model, "joint")
            features = ControlFieldEncoder("joint_f", per_world_shape=(codec.layout.dof_qd,), key=CONTROL_KEY)
            artifact = DynamicsArtifact.load(args.output / "training" / "artifact")
            with (
                NewtonTrajectoryZarrReader(train_data) as train_reader,
                NewtonTrajectoryZarrReader(evaluation_data) as evaluation_reader,
            ):
                errors = evaluate_rollout_nrmse(
                    artifact,
                    evaluation_reader,
                    codec,
                    features,
                    evaluation_indices=range(len(evaluation_reader)),
                    reference_indices=range(len(train_reader)),
                    reference_reader=train_reader,
                    horizons=HORIZONS,
                    device=args.device,
                )
        finally:
            env.close()
        print("horizon  E_H (this run)  E_H (FlashNeRD paper)  trajectories  non-finite")
        for error in errors:
            paper = PAPER_FLASHNERD.get(error.horizon)
            paper_text = f"{paper:.2g}" if paper is not None else "-"
            print(
                f"{error.horizon:7d}  {error.nrmse:14.3g}  {paper_text:>21}  {error.trajectories:12d}"
                f"  {error.nonfinite_trajectories:10d}"
            )


if __name__ == "__main__":
    main()
