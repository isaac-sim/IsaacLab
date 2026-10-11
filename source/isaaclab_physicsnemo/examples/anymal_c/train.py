# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the FlashNeRD ANYmal-C locomotion model on a dataset recorded by record.py.

The model sees the state in the base heading frame and is conditioned on the joint-position targets,
base-frame gravity, and the four fixed foot probes. FlashNeRD trains it for 60,000 updates with a 13-frame
receptive field and a 5 % trajectory validation split; this run uses resident windows, FP16 autocast,
global batch 4,096, and the ParallelNeRD geometry of the study's G1 model. Launch it under ``torchrun``
for several data-parallel ranks.
"""

import argparse
import logging
import os
from pathlib import Path

from anymal_c_env_cfg import (
    ANYMAL_MODEL_CONFIG,
    AnymalCTargetControlEnvCfg,
    anymal_dynamics_features,
    anymal_foot_contacts,
)
from isaaclab_physicsnemo import HeadingJointStateCodec, TrainingSettings, train_dynamics_artifact

from isaaclab.envs import ManagerBasedRLEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset", type=Path, nargs="+", required=True, help="Datasets written by record.py, trained on together."
    )
    parser.add_argument("--output", type=Path, required=True, help="Training output directory.")
    parser.add_argument("--max-updates", type=int, default=60_000)
    parser.add_argument("--batch-size", type=int, default=4096, help="Global batch over all ranks.")
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    device = f"cuda:{os.environ['LOCAL_RANK']}" if "LOCAL_RANK" in os.environ else args.device

    # The codec and probes are built from the deployment scene, whose Newton model equals the teacher's.
    cfg = AnymalCTargetControlEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = device
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        model = env.sim.physics_manager.get_model()
        codec = HeadingJointStateCodec(model)
        summary = train_dynamics_artifact(
            args.dataset,
            model,
            args.output,
            codec=codec,
            features=anymal_dynamics_features(codec, anymal_foot_contacts(model)),
            settings=TrainingSettings(
                history=13,
                max_updates=args.max_updates,
                batch_size=args.batch_size,
                validation_fraction=0.05,
                log_every=1000,
                seed=args.seed,
                architecture="parallel_nerd",
                model_kwargs=ANYMAL_MODEL_CONFIG,
                mixed_precision=True,
                compile=True,
                checkpoint_every=args.checkpoint_every,
                validate_every=args.checkpoint_every,
            ),
            device=device,
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
