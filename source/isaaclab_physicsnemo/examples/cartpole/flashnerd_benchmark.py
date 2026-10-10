# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the FlashNeRD CartPole rollout error with the study's protocol, on Isaac Lab or the study's own teacher.

The run follows the CartPole configuration of the FlashNeRD main-results study
(PhysicsNeMo Robotics tag ``mnabian/flashnerd-benchmark-v1``):

1. ``data``: 10,000 training trajectories of 100 transitions and 2,048 held-out trajectories of 509
   transitions at 1/60 s from uniform initial states and uniform random efforts resampled every frame.
   ``--teacher isaaclab`` records the Isaac Lab Cartpole with MuJoCo Warp; ``--teacher physicsnemo``
   generates the study's own Cartpole with Newton's Featherstone solver through the PhysicsNeMo Robotics
   example code in ``--physicsnemo-root``.
   The Isaac Lab cart has 2 kg of moving mass against the study's 52.5 kg, so its default forces of up to
   400 N accelerate it ten times harder; ``--force-range 57.1`` matches the study's cart acceleration.
2. ``train``: ParallelNeRD with the frozen CartPole geometry, ten frames of history, global batch 4,096,
   a linear 1e-3 to 1e-4 learning rate, resident windows, and FP16 autocast. Every checkpoint appends the
   held-out ``E_H`` to ``rollout_nrmse.jsonl``, and a rerun resumes from the last checkpoint. Launch it
   with ``torchrun --nproc-per-node 2`` for the study's two data-parallel ranks.
3. ``evaluate``: the prefix-pooled rollout NRMSE ``E_H`` over the held-out cohort, normalized by the
   training-state standard deviations, for H = 1, 15, 50, 100, 200, and 500.

The study trains for five million updates; ``--max-updates`` sets the budget of this run.
"""

import argparse
import json
import logging
import os
import sys
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
# The Isaac Lab cart actuator saturates at 400 N.
ISAACLAB_FORCE_RANGE = 400.0
# The study drives its 52.5 kg Cartpole with up to 1500 N; the same 28.6 m/s^2 of the Isaac Lab Cartpole's
# 2 kg of moving mass needs 57 N.
ISAACLAB_MATCHED_FORCE_RANGE = 1500.0 / 52.5 * 2.0
# Data controls of the study's CartPole teacher.
STUDY_DATA = {
    "history": 10,
    "validation_fraction": 0.1,
    "cart_position_range": 1.0,
    "pole_angle_range": 3.141592653589793,
    "velocity_range": 1.0,
    "force_scale": 1500.0,
}


def record_isaaclab(
    path: Path,
    *,
    num_envs: int,
    transitions: int,
    episodes_per_env: int,
    seed: int,
    device: str,
    force_range: float = ISAACLAB_FORCE_RANGE,
):
    """Record ``num_envs * episodes_per_env`` Isaac Lab episodes of ``transitions`` transitions each.

    Each frame applies a cart force drawn uniformly within ``force_range`` [N].
    """
    cfg = CartpoleFlashNeRDBenchmarkEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.seed = seed
    cfg.episode_length_s = (transitions + 0.5) * cfg.sim.dt
    env = CartpoleEnv(cfg=cfg)
    try:
        env.reset(seed=seed)
        high = force_range / cfg.action_scale
        actions = UniformRandomActions(num_envs, cfg.action_space, high=high, device=env.device, seed=seed)
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


def physicsnemo_example(root: Path):
    """Import the PhysicsNeMo Robotics CartPole example modules from a source checkout."""
    sys.path.insert(0, str(root / "examples" / "learned_dynamics" / "cartpole"))
    import data
    import scene

    return data, scene


def generate_physicsnemo(root: Path, path: Path, *, trajectories: int, world_batch: int, transitions: int, seed: int):
    """Generate the study's Featherstone CartPole trajectories with the PhysicsNeMo Robotics example."""
    from omegaconf import OmegaConf

    data, _ = physicsnemo_example(root)
    config = OmegaConf.create(
        STUDY_DATA
        | {"seed": seed, "trajectories": trajectories, "world_batch_size": world_batch, "frames": transitions}
    )
    storage = OmegaConf.create({"backend": "disk", "path": str(path), "regenerate": True})
    teacher = data.collect_cartpole_teacher_data(config, torch.device("cuda:0"), storage=storage)
    print(f"Wrote {len(teacher.reader)} trajectories to {path}")


def dataset(path: Path) -> Path:
    """Return the Zarr dataset directory, which the PhysicsNeMo Robotics generator writes as one shard."""
    shard = path / "shard-000"
    return shard if shard.is_dir() else path


class Scene:
    """Newton model of the teacher scene, which defines the state codec and the artifact identity."""

    def __init__(self, teacher: str, physicsnemo_root: Path | None, device: str) -> None:
        self.env = None
        if teacher == "isaaclab":
            cfg = CartpoleFlashNeRDBenchmarkEnvCfg()
            cfg.scene.num_envs = 1
            cfg.sim.device = device
            self.env = CartpoleEnv(cfg=cfg)
            self.model = self.env.sim.physics_manager.get_model()
        else:
            _, scene = physicsnemo_example(physicsnemo_root)
            self.model = scene.make_cartpole(device)[0].model
        self.codec = newton_state_codec(self.model, "joint")
        self.features = ControlFieldEncoder("joint_f", per_world_shape=(self.codec.layout.dof_qd,), key=CONTROL_KEY)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            SimulationContext.clear_instance()


def rollout_errors(scene: Scene, artifact: Path, train_data: Path, evaluation_data: Path, device: str):
    """Return the held-out ``E_H`` of an artifact, normalized by the training-state spread."""
    with (
        NewtonTrajectoryZarrReader(dataset(train_data)) as train_reader,
        NewtonTrajectoryZarrReader(dataset(evaluation_data)) as evaluation_reader,
    ):
        return evaluate_rollout_nrmse(
            DynamicsArtifact.load(artifact),
            evaluation_reader,
            scene.codec,
            scene.features,
            evaluation_indices=range(len(evaluation_reader)),
            reference_indices=range(len(train_reader)),
            reference_reader=train_reader,
            horizons=HORIZONS,
            device=device,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Directory of datasets, training, and results.")
    parser.add_argument(
        "--stages", nargs="+", default=["data", "train", "evaluate"], choices=["data", "train", "evaluate"]
    )
    parser.add_argument("--teacher", choices=["isaaclab", "physicsnemo"], default="isaaclab")
    parser.add_argument(
        "--force-range",
        type=float,
        default=ISAACLAB_FORCE_RANGE,
        help=(
            "Bound [N] of the uniform Isaac Lab cart forces. The default is the actuator limit; "
            f"{ISAACLAB_MATCHED_FORCE_RANGE:.1f} N matches the study's cart acceleration."
        ),
    )
    parser.add_argument(
        "--physicsnemo-root", type=Path, help="PhysicsNeMo Robotics checkout for --teacher physicsnemo."
    )
    parser.add_argument("--max-updates", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4096, help="Global batch over all ranks.")
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--fp32", action="store_true", help="Train without FP16 autocast.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.teacher == "physicsnemo" and args.physicsnemo_root is None:
        parser.error("--teacher physicsnemo requires --physicsnemo-root")
    logging.basicConfig(level=logging.INFO)
    device = f"cuda:{os.environ['LOCAL_RANK']}" if "LOCAL_RANK" in os.environ else args.device
    train_data, evaluation_data = args.output / "train", args.output / "evaluation"
    curve = args.output / "rollout_nrmse.jsonl"

    if "data" in args.stages:
        if args.teacher == "isaaclab":
            for path, num_envs, transitions, episodes_per_env, seed in (
                (train_data, 2000, 100, 5, 7),
                (evaluation_data, 2048, 509, 1, 8),
            ):
                record_isaaclab(
                    path,
                    num_envs=num_envs,
                    transitions=transitions,
                    episodes_per_env=episodes_per_env,
                    seed=seed,
                    device=device,
                    force_range=args.force_range,
                )
        else:
            generate_physicsnemo(
                args.physicsnemo_root, train_data, trajectories=10_000, world_batch=2_000, transitions=100, seed=7
            )
            generate_physicsnemo(
                args.physicsnemo_root, evaluation_data, trajectories=2_048, world_batch=1_024, transitions=509, seed=8
            )

    scene = Scene(args.teacher, args.physicsnemo_root, device)
    try:
        if "train" in args.stages:

            def track(updates: int, artifact: Path) -> None:
                errors = rollout_errors(scene, artifact, train_data, evaluation_data, device)
                record = {"updates": updates, **{f"E{error.horizon}": error.nrmse for error in errors}}
                print(f"rollout NRMSE after {updates} updates: {record}")
                with curve.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")

            summary = train_dynamics_artifact(
                dataset(train_data),
                scene.model,
                args.output / "training",
                codec=scene.codec,
                features=scene.features,
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
                    resident=True,
                    mixed_precision=not args.fp32,
                    compile=True,
                    checkpoint_every=args.checkpoint_every,
                    validate_every=args.checkpoint_every,
                    rollout_horizons=(),
                ),
                device=device,
                on_checkpoint=track,
            )
            print(f"Trained {summary.parameter_updates} updates, validation {summary.validation}")

        if "evaluate" in args.stages and int(os.environ.get("RANK", "0")) == 0:
            errors = rollout_errors(scene, args.output / "training" / "artifact", train_data, evaluation_data, device)
            print("horizon  E_H (this run)  E_H (FlashNeRD paper)  trajectories  non-finite")
            for error in errors:
                paper = PAPER_FLASHNERD.get(error.horizon)
                paper_text = f"{paper:.2g}" if paper is not None else "-"
                print(
                    f"{error.horizon:7d}  {error.nrmse:14.3g}  {paper_text:>21}  {error.trajectories:12d}"
                    f"  {error.nonfinite_trajectories:10d}"
                )
    finally:
        scene.close()


if __name__ == "__main__":
    main()
