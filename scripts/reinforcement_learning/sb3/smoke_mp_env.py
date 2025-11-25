# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight smoke test for MP-based envs with Sb3VecEnvWrapper and VecNormalize.

Run this inside the Isaac Lab conda env to ensure the masked policy observation space
matches the actual observations SB3 sees (catches shape/broadcast issues).
"""

import argparse
import numpy as np

from isaaclab.app import AppLauncher


def resolve_class(path: str):
    mod_name, cls_name = path.split(":")
    mod = __import__(mod_name, fromlist=[cls_name])
    return getattr(mod, cls_name)


def main():
    parser = argparse.ArgumentParser(description="Smoke-test MP env masking and SB3 wrappers.")
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
    parser.add_argument("--norm_reward", action="store_true", help="Also normalize rewards.")
    parser.add_argument("--no_norm_obs", action="store_true", help="Disable observation normalization.")
    parser.add_argument("--gui", dest="headless", action="store_false", help="Run with GUI.")
    parser.set_defaults(headless=True)
    args_cli, _ = parser.parse_known_args()

    # launch app early
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym

    from stable_baselines3.common.vec_env import VecNormalize

    from isaaclab_rl import Sb3MPVecEnvWrapper

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
    vec_env = Sb3MPVecEnvWrapper(base_env)
    print(f"[smoke] vec_env observation_space: {vec_env.observation_space}")
    print(f"[smoke] vec_env single_observation_space: {getattr(vec_env.env, 'single_observation_space', None)}")

    norm_obs = not args_cli.no_norm_obs
    if norm_obs or args_cli.norm_reward:
        vec_env = VecNormalize(
            vec_env,
            training=True,
            norm_obs=norm_obs,
            norm_reward=args_cli.norm_reward,
            gamma=0.99,
        )

    obs = vec_env.reset()
    print(f"[smoke] reset obs shape: {obs.shape}")
    policy_dim = vec_env.observation_space.shape[-1]
    assert obs.shape[-1] == policy_dim, f"Obs dim {obs.shape[-1]} != space dim {policy_dim}"

    actions = np.zeros((vec_env.num_envs,) + vec_env.action_space.shape, dtype=np.float32)
    for _ in range(args_cli.steps):
        obs, rew, done, infos = vec_env.step(actions)
        # simple sanity: shapes should stay consistent across steps/resets
        assert obs.shape[-1] == policy_dim, f"Step obs dim {obs.shape[-1]} != space dim {policy_dim}"
        if done.any():
            obs = vec_env.reset()
            assert obs.shape[-1] == policy_dim, "Obs dim mismatch after reset."

    print(
        f"Smoke test passed: policy dim={policy_dim}, "
        f"obs shape={obs.shape}, action shape={vec_env.action_space.shape}, num_envs={vec_env.num_envs}"
    )

    vec_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
