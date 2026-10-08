# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record task-independent ANYmal-D episodes stepped by MuJoCo Warp as a PhysicsNeMo trajectory dataset.

Episodes follow the FlashNeRD locomotion data: randomized initial states, uniform joint-position targets
within 0.75 rad of the default pose resampled every 20 ms, at most 2 s, and falls kept. Each 5 ms physics
step is one recorded frame, and each transition also stores the four fixed foot-probe tokens of its
starting state.
"""

import argparse
from pathlib import Path

import torch
from actions import ACTION_BOUND, make_action_source
from isaaclab_physicsnemo import TrajectoryDatasetRecorder
from isaaclab_physicsnemo.tasks import (
    ANYMAL_D_REMOVE_ROOT_TRANSLATION,
    AnymalDExplorationEnvCfg,
    anymal_d_foot_contacts,
)

from isaaclab.envs import ManagerBasedRLEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Dataset directory.")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3000, help="Physics steps (5 ms) per environment.")
    parser.add_argument("--policy", type=Path, default=None, help="Optional exported TorchScript policy.")
    parser.add_argument("--action-bound", type=float, default=ACTION_BOUND, help="Bound of the uniform actions.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--replace", action="store_true", help="Replace an existing dataset.")
    args = parser.parse_args()

    cfg = AnymalDExplorationEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        env.reset(seed=args.seed)
        # Targets change every 20 ms control period while each environment step is one 5 ms physics step.
        hold_steps = round(0.02 / cfg.sim.dt)
        actions = make_action_source(
            env, policy_path=args.policy, high=args.action_bound, seed=args.seed, hold_steps=hold_steps
        )
        dones = None
        with TrajectoryDatasetRecorder(
            env,
            args.output,
            scenario_id="isaaclab-anymal-d-flat-mjwarp-task-independent",
            remove_root_translation=ANYMAL_D_REMOVE_ROOT_TRANSLATION,
            recorded_features=(anymal_d_foot_contacts(env.sim.physics_manager.get_model()),),
            seed=args.seed,
            replace=args.replace,
        ) as recorder:
            with torch.inference_mode():
                for _ in range(args.steps):
                    _, _, terminated, truncated, _ = env.step(actions(dones))
                    dones = terminated | truncated
                    recorder.record(dones)
        print(f"Wrote {recorder.episodes_written} episodes to {args.output.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
