# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the FlashNeRD ANYmal-C model as the Newton solver and compare it with MuJoCo Warp and its actuator network."""

import argparse
from pathlib import Path

from anymal_c_env_cfg import ANYMAL_CONTINUITY, AnymalCFlatTeacherEnvCfg, AnymalCTargetControlEnvCfg
from isaaclab_physicsnemo import (
    UniformRandomActions,
    apply_learned_physics,
    compare_rollouts,
    preflight_learned_physics,
    record_joint_rollout,
)

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import SimulationContext

ACTION_BOUND = 1.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="Artifact directory written by train.py.")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100, help="Control periods (20 ms) of each rollout.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--inference-dtype",
        choices=("float32", "float16"),
        default="float32",
        help="Precision of the learned model's projections.",
    )
    args = parser.parse_args()

    def rollout(cfg):
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            env.reset(seed=args.seed)
            report = None
            if isinstance(cfg, AnymalCTargetControlEnvCfg):
                report = preflight_learned_physics(env.sim.physics_manager, artifact)
                print(report)
                if not report.passed:
                    raise RuntimeError("The learned ANYmal-C failed deployment preflight.")
            # Both rollouts start from one reset with the same seed, so they share initial states and targets.
            actions = UniformRandomActions(
                env.num_envs, env.action_manager.total_action_dim, high=ACTION_BOUND, device=env.device, seed=args.seed
            )
            return record_joint_rollout(env, actions, args.steps)
        finally:
            env.close()
            SimulationContext.clear_instance()

    teacher = rollout(AnymalCFlatTeacherEnvCfg())
    cfg = AnymalCTargetControlEnvCfg()
    artifact = apply_learned_physics(
        cfg, args.artifact, continuity=ANYMAL_CONTINUITY, inference_dtype=args.inference_dtype
    )
    learned = rollout(cfg)

    if not (learned.joint_q.isfinite().all() and learned.joint_qd.isfinite().all()):
        raise RuntimeError("The learned ANYmal-C produced a non-finite state.")
    error = compare_rollouts(teacher, learned)
    base_height_error = (learned.joint_q[..., 2] - teacher.joint_q[..., 2]).abs()
    print("step  active  joint_q_rmse  joint_qd_rmse  mean_base_height_error[m]")
    for step in range(0, args.steps + 1, max(1, args.steps // 10)):
        active = teacher.active[step] & learned.active[step]
        print(
            f"{step:4d}  {int(error.active_envs[step]):6d}  {error.joint_q_rmse[step].item():12.5f}"
            f"  {error.joint_qd_rmse[step].item():13.5f}  {base_height_error[step][active].mean().item():25.4f}"
        )


if __name__ == "__main__":
    main()
