# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record the FlashNeRD ANYmal-C locomotion data: task-independent 20 ms control periods stepped by MuJoCo Warp.

Episodes start from randomized base poses, base velocities, and joint configurations, receive uniform
joint-position targets within 0.75 rad of the default pose every 20 ms, last at most 2 s, and keep falls.
Each frame is one control period of four 5 ms physics steps, in which the ANYdrive actuator network turns
the targets into torques, and each transition stores the targets and the four foot-probe tokens of its
starting state.
"""

import argparse
from pathlib import Path

import torch
from anymal_c_env_cfg import AnymalCExplorationEnvCfg, anymal_foot_contacts
from isaaclab_physicsnemo import HeadingJointStateCodec, TrajectoryDatasetRecorder, UniformRandomActions

from isaaclab.envs import ManagerBasedRLEnv

# The joint-position action scales by 0.5 rad, so 1.5 gives targets within 0.75 rad of the default pose.
ACTION_BOUND = 1.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Dataset directory.")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500, help="Control periods (20 ms) per environment.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--replace", action="store_true", help="Replace an existing dataset.")
    parser.add_argument(
        "--shard-episodes", type=int, default=None, help="Write datasets of at most this many episodes under --output."
    )
    args = parser.parse_args()

    cfg = AnymalCExplorationEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        env.reset(seed=args.seed)
        model = env.sim.physics_manager.get_model()
        actions = UniformRandomActions(
            env.num_envs, env.action_manager.total_action_dim, high=ACTION_BOUND, device=env.device, seed=args.seed
        )
        dones = None
        with TrajectoryDatasetRecorder(
            env,
            args.output,
            scenario_id="isaaclab-anymal-c-flat-mjwarp-flashnerd",
            codec=HeadingJointStateCodec(model),
            recorded_features=(anymal_foot_contacts(model),),
            control_targets=env.scene["robot"],
            seed=args.seed,
            replace=args.replace,
            shard_episodes=args.shard_episodes,
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
