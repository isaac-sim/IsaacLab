# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a Cartpole NeRD dynamics artifact on a recorded dataset."""

import argparse
import logging
from pathlib import Path

from isaaclab_physicsnemo import TrainingSettings, train_dynamics_artifact
from isaaclab_physicsnemo.tasks import CARTPOLE_REMOVE_ROOT_TRANSLATION, CartpoleTeacherEnvCfg

from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset written by record.py.")
    parser.add_argument("--output", type=Path, required=True, help="Training output directory.")
    parser.add_argument("--max-updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # The state codec is built from the finalized model of the recording scene.
    cfg = CartpoleTeacherEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env = CartpoleEnv(cfg=cfg)
    try:
        summary = train_dynamics_artifact(
            args.dataset,
            env.sim.physics_manager.get_model(),
            args.output,
            remove_root_translation=CARTPOLE_REMOVE_ROOT_TRANSLATION,
            settings=TrainingSettings(
                history=args.history, max_updates=args.max_updates, batch_size=args.batch_size, seed=args.seed
            ),
            device=args.device,
        )
    finally:
        env.close()
    print(
        f"Trained {summary.parameter_updates} updates on {len(summary.train_trajectory_ids)} trajectories,"
        f" validated on {len(summary.validation_trajectory_ids)}"
    )
    for name, value in summary.validation.items():
        print(f"  validation {name}: {value:.6g}")
    for error in summary.rollout_nrmse:
        print(f"  rollout NRMSE E{error.horizon}: {error.nrmse:.4g} over {error.trajectories} trajectories")
    print(f"Artifact: {summary.artifact_path}")


if __name__ == "__main__":
    main()
