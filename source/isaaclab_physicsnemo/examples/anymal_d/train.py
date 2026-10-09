# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an ANYmal-D ParallelNeRD (FlashNeRD) dynamics artifact on a recorded dataset.

The model follows the input recipe of the PhysicsNeMo Robotics G1 application: the state in the base
heading frame, conditioned on the applied joint efforts, base-frame gravity, and the four fixed foot-probe
features, with the ParallelNeRD geometry of the FlashNeRD main-results G1 model. FlashNeRD trains its
ANYmal model for 60,000 updates with a 5 % trajectory validation split.
"""

import argparse
import logging
from pathlib import Path

from isaaclab_physicsnemo import TrainingSettings, train_dynamics_artifact
from isaaclab_physicsnemo.tasks import (
    ANYMAL_D_MODEL_CONFIG,
    AnymalDFlatTeacherEnvCfg,
    anymal_d_dynamics_features,
    anymal_d_foot_contacts,
    anymal_d_state_codec,
)

from isaaclab.envs import ManagerBasedRLEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset written by record.py.")
    parser.add_argument("--output", type=Path, required=True, help="Training output directory.")
    parser.add_argument("--max-updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--history", type=int, default=10, help="Frames observed; the receptive field of the model.")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # The state codec and probes are built from the finalized model of the recording scene.
    cfg = AnymalDFlatTeacherEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        model = env.sim.physics_manager.get_model()
        codec = anymal_d_state_codec(model)
        features = anymal_d_dynamics_features(codec, anymal_d_foot_contacts(model))
        summary = train_dynamics_artifact(
            args.dataset,
            model,
            args.output,
            codec=codec,
            features=features,
            settings=TrainingSettings(
                history=args.history,
                max_updates=args.max_updates,
                batch_size=args.batch_size,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                architecture="parallel_nerd",
                model_kwargs=ANYMAL_D_MODEL_CONFIG,
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
    print(f"Artifact: {summary.artifact_path}")


if __name__ == "__main__":
    main()
