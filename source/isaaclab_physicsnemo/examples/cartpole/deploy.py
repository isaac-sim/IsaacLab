# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a Cartpole dynamics artifact as the Newton solver and compare it with MuJoCo Warp."""

import argparse
from pathlib import Path

from cartpole_env_cfg import CartpoleTeacherEnvCfg
from isaaclab_physicsnemo import (
    SmoothRandomActions,
    apply_learned_physics,
    compare_rollouts,
    preflight_learned_physics,
    record_joint_rollout,
)

from isaaclab.sim import SimulationContext

from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="Artifact directory written by train.py.")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=120, help="Environment steps of each rollout.")
    parser.add_argument("--action-scale", type=float, default=0.5, help="Must match the recorded action scale.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    def make_cfg() -> CartpoleTeacherEnvCfg:
        cfg = CartpoleTeacherEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        return cfg

    # Both rollouts start from one reset with the same seed, so they share initial states and actions.
    def rollout(env: CartpoleEnv):
        actions = SmoothRandomActions(
            env.num_envs, env.cfg.action_space, scale=args.action_scale, device=env.device, seed=args.seed
        )
        return record_joint_rollout(env, actions, args.steps)

    # teacher reference rollout
    env = CartpoleEnv(cfg=make_cfg())
    try:
        env.reset(seed=args.seed)
        teacher = rollout(env)
    finally:
        env.close()
    SimulationContext.clear_instance()

    # learned rollout from the same seed and actions
    cfg = make_cfg()
    artifact = apply_learned_physics(cfg, args.artifact)
    env = CartpoleEnv(cfg=cfg)
    try:
        env.reset(seed=args.seed)
        report = preflight_learned_physics(env.sim.physics_manager, artifact)
        print(report)
        if not report.passed:
            raise RuntimeError("The learned Cartpole failed deployment preflight.")
        learned = rollout(env)
    finally:
        env.close()

    if not (learned.joint_q.isfinite().all() and learned.joint_qd.isfinite().all()):
        raise RuntimeError("The learned Cartpole produced a non-finite state.")
    error = compare_rollouts(teacher, learned)
    print("step  active  joint_q_rmse  joint_qd_rmse")
    for step in range(0, args.steps + 1, max(1, args.steps // 10)):
        print(
            f"{step:4d}  {int(error.active_envs[step]):6d}  {error.joint_q_rmse[step].item():12.5f}"
            f"  {error.joint_qd_rmse[step].item():13.5f}"
        )


if __name__ == "__main__":
    main()
