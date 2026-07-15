# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate an oversampled candidate pool for the Franka Pour reset dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.franka_pour.pour_env import FrankaPourResetSamplerEnv
from isaaclab_tasks.contrib.franka_pour.reset_dataset_generator import (
    FRANKA_POUR_RESET_DATASET_TASK_ID,
    FrankaPourResetDatasetGenerator,
    FrankaPourResetDatasetGeneratorCfg,
    save_reset_dataset,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT = _REPO_ROOT / "datasets/franka_pour/reset_dataset_candidates.pt"
_CANDIDATE_GRASPING_COUNT = 14_000
_CANDIDATE_NON_GRASPING_COUNT = 12_000
_CANDIDATE_NEAR_POUR_COUNT = 2_000


def main() -> None:
    """Launch one task world, sample exact category quotas, and atomically save them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=FRANKA_POUR_RESET_DATASET_TASK_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_attempt_multiplier", type=int, default=100)
    parser.add_argument("--grasping_count", type=int, default=_CANDIDATE_GRASPING_COUNT)
    parser.add_argument("--non_grasping_count", type=int, default=_CANDIDATE_NON_GRASPING_COUNT)
    parser.add_argument("--near_pour_grasp_count", type=int, default=_CANDIDATE_NEAR_POUR_COUNT)
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Candidate dataset written before dynamic validation.",
    )
    add_launcher_args(parser)
    args = parser.parse_args()

    # One scene world is sufficient: IK and collision candidates are replicated in sampler-owned
    # Newton models.  This avoids allocating an MPM training batch merely to generate reset data.
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    with launch_simulation(env_cfg, args):
        env = FrankaPourResetSamplerEnv(env_cfg)
        try:
            sampler_cfg = FrankaPourResetDatasetGeneratorCfg(
                grasping_count=args.grasping_count,
                non_grasping_count=args.non_grasping_count,
                near_pour_grasp_count=args.near_pour_grasp_count,
                batch_size=args.batch_size,
                seed=args.seed,
                max_attempt_multiplier=args.max_attempt_multiplier,
            )
            payload = FrankaPourResetDatasetGenerator(env, sampler_cfg).generate()
            save_reset_dataset(payload, args.output)
            print(f"[INFO] Wrote {payload['metadata']['state_count']} candidate states to {args.output.resolve()}.")
            print(f"[INFO] Contract SHA-256: {payload['contract_sha256']}")
            print(f"[INFO] Content SHA-256:  {payload['content_sha256']}")
            print("[INFO] Next: validate_franka_pour_reset_dataset.py")
        finally:
            env.close()


if __name__ == "__main__":
    main()
