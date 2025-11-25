# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight smoke test for MP-based envs with the RslRlMPVecEnvWrapper."""

import argparse
import torch

from isaaclab.app import AppLauncher


def resolve_class(path: str):
    mod_name, cls_name = path.split(":")
    mod = __import__(mod_name, fromlist=[cls_name])
    return getattr(mod, cls_name)


def main():
    parser = argparse.ArgumentParser(description="Smoke-test MP env masking and RSL-RL wrapper.")
    parser.add_argument(
        "--base_id",
        type=str,
        default="Isaac-Box-Pushing-Dense-step-Franka-v0",
        help="Base step env id (e.g., Isaac-Box-Pushing-Dense-step-Franka-v0).",
    )
    parser.add_argument("--mp_id", type=str, default=None, help="MP env id to register/use.")
    parser.add_argument(
        "--mp_wrapper",
        type=str,
        default="isaaclab_tasks.manager_based.box_pushing.mp_wrapper:BoxPushingMPWrapper",
        help="MP wrapper class import path.",
    )
    parser.add_argument("--mp_type", type=str, default="ProDMP", choices=["ProDMP", "ProMP", "DMP"], help="MP backend.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device.")
    parser.add_argument("--steps", type=int, default=2, help="How many env steps to run.")
    parser.add_argument("--gui", dest="headless", action="store_false", help="Run with GUI.")
    parser.set_defaults(headless=True)
    args_cli, _ = parser.parse_known_args()

    # launch app early
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym

    from isaaclab_rl import RslRlMPVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.mp import upgrade

    mp_wrapper_cls = resolve_class(args_cli.mp_wrapper)
    mp_id = args_cli.mp_id or f"Isaac_MP/{args_cli.base_id.split('/')[-1]}-{args_cli.mp_type}"
    mp_id = upgrade(
        mp_id=mp_id,
        base_id=args_cli.base_id,
        mp_wrapper_cls=mp_wrapper_cls,
        mp_type=args_cli.mp_type,
        device=args_cli.device,
    )

    env_cfg = parse_env_cfg(
        args_cli.base_id, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.headless
    )
    base_env = gym.make(mp_id, cfg=env_cfg)
    vec_env = RslRlMPVecEnvWrapper(base_env)

    space = vec_env.observation_space["policy"]
    policy_dim = space.shape[-1]
    action_dim = vec_env.num_actions
    print(f"[smoke] policy dim={policy_dim}, action dim={action_dim}")

    obs, _ = vec_env.reset()
    policy_obs = obs["policy"]
    assert policy_obs.shape[-1] == policy_dim, f"Reset obs dim {policy_obs.shape[-1]} != space dim {policy_dim}"

    actions = torch.zeros((vec_env.num_envs, action_dim), device=vec_env.device, dtype=torch.float32)
    for _ in range(args_cli.steps):
        obs, rew, done, _ = vec_env.step(actions)
        policy_obs = obs["policy"]
        assert policy_obs.shape[-1] == policy_dim, f"Step obs dim {policy_obs.shape[-1]} != space dim {policy_dim}"
        assert rew.shape[0] == vec_env.num_envs
        if torch.as_tensor(done).any():
            obs, _ = vec_env.reset()
            policy_obs = obs["policy"]
            assert policy_obs.shape[-1] == policy_dim, "Obs dim mismatch after reset."

    print(
        f"Smoke test passed: policy dim={policy_dim}, obs shape={policy_obs.shape}, "
        f"action shape={(vec_env.num_envs, action_dim)}, num_envs={vec_env.num_envs}"
    )

    vec_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
