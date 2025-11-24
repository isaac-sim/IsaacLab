# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import yaml
from pathlib import Path

from parse_sb3_hp import process_sb3_cfg
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize


def main():
    task_name = "fancy/BoxPushingDense-v0"

    cfg_path = Path(__file__).resolve().parent / "sb3_ppo_cfg.yaml"
    agent_cfg = yaml.safe_load(cfg_path.read_text())
    agent_cfg = process_sb3_cfg(agent_cfg)
    seed = agent_cfg.pop("seed")
    n_envs = agent_cfg.pop("n_envs")
    n_steps_per_update = agent_cfg.pop("n_steps_per_update")
    if n_steps_per_update % n_envs != 0:
        raise ValueError(f"n_steps_per_update: {n_steps_per_update} not dividable by n_envs: {n_envs}")
    n_steps = int(n_steps_per_update / n_envs)
    agent_cfg["n_steps"] = n_steps
    n_minibatches = agent_cfg.pop("n_minibatches")
    agent_cfg["batch_size"] = int(n_envs * n_steps / n_minibatches)

    vec_env = make_vec_env(task_name, vec_env_cls=SubprocVecEnv, n_envs=n_envs)
    vec_env.seed(seed)

    if "normalize_input" in agent_cfg:
        vec_env = VecNormalize(
            vec_env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward="clip_rew" in agent_cfg and agent_cfg.pop("clip_rew"),
        )

    model = PPO.load("box_pushing")

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, _, _, _ = vec_env.step(action)
        vec_env.render("human")


if __name__ == "__main__":
    main()
