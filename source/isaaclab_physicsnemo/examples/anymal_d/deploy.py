# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run an ANYmal-D dynamics artifact as the Newton solver and compare it with MuJoCo Warp."""

import argparse
from pathlib import Path

from actions import ACTION_BOUND, make_action_source
from isaaclab_physicsnemo import (
    apply_learned_physics,
    compare_rollouts,
    preflight_learned_physics,
    record_joint_rollout,
)
from isaaclab_physicsnemo.tasks import ANYMAL_D_CONTINUITY, AnymalDFlatTeacherEnvCfg

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import SimulationContext


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="Artifact directory written by train.py.")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100, help="Environment steps of each rollout.")
    parser.add_argument("--policy", type=Path, default=None, help="Optional exported TorchScript policy.")
    parser.add_argument("--action-bound", type=float, default=ACTION_BOUND, help="Bound of the uniform actions.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    def make_cfg() -> AnymalDFlatTeacherEnvCfg:
        cfg = AnymalDFlatTeacherEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        return cfg

    # Both rollouts start from one reset with the same seed, so they share initial states and actions.
    def rollout(env: ManagerBasedRLEnv):
        actions = make_action_source(env, policy_path=args.policy, high=args.action_bound, seed=args.seed)
        return record_joint_rollout(env, actions, args.steps)

    # teacher reference rollout
    env = ManagerBasedRLEnv(cfg=make_cfg())
    try:
        env.reset(seed=args.seed)
        teacher = rollout(env)
    finally:
        env.close()
    SimulationContext.clear_instance()

    # learned rollout from the same seed and actions
    cfg = make_cfg()
    artifact = apply_learned_physics(cfg, args.artifact, continuity=ANYMAL_D_CONTINUITY)
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        env.reset(seed=args.seed)
        report = preflight_learned_physics(env.sim.physics_manager, artifact)
        print(report)
        if not report.passed:
            raise RuntimeError("The learned ANYmal-D failed deployment preflight.")
        learned = rollout(env)
    finally:
        env.close()

    if not (learned.joint_q.isfinite().all() and learned.joint_qd.isfinite().all()):
        raise RuntimeError("The learned ANYmal-D produced a non-finite state.")
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
