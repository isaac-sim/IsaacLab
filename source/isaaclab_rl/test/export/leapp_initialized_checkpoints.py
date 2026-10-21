# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create native initialized checkpoints from fixed test spaces, without starting simulation."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from leapp_test_tasks import MANAGER_TASK
from tensordict import TensorDict

from isaaclab.utils import to_dict

# These are the real tensor scene's policy terms. Joint velocity has two frames
# in concatenated groups and one frame in SB3's dictionary group.
_POLICY_WIDTHS = {"joint_pos": 10, "joint_vel": 10, "previous": 10}


class _CheckpointEnv(gym.Env):
    """Only the spaces and samples needed to initialize the test policies on CPU."""

    def __init__(self, task: str, backend: str):
        self.device = "cpu"
        self.num_envs = 1
        self.cfg = {}
        self.render_mode = None
        self.state_space = None
        if task == MANAGER_TASK:
            self.num_actions = 10
            shapes = {"policy": (40,), "proprio": (2,), "critic": (2,), "vision": (3, 4, 4)}
            self.state_space = gym.spaces.Box(-np.inf, np.inf, shapes["critic"], dtype=np.float32)
        elif task == "Isaac-Reach-UR10":
            self.num_actions = 6
            shapes = {"policy": (24,)}  # Six positions, two velocity frames, and six previous actions.
        else:
            raise ValueError(f"No checkpoint spaces defined for {task}")
        self._observations = TensorDict({key: torch.zeros(1, *shape) for key, shape in shapes.items()}, batch_size=[1])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shapes["policy"], dtype=np.float32)
        if backend == "sb3":
            self.observation_space = gym.spaces.Dict(
                {
                    key: gym.spaces.Box(-np.inf, np.inf, (width,), dtype=np.float32)
                    for key, width in _POLICY_WIDTHS.items()
                }
            )
        action_bound = 1.0 if backend in ("rl_games", "sb3") else np.inf
        self.action_space = gym.spaces.Box(-action_bound, action_bound, (self.num_actions,), dtype=np.float32)

    def get_observations(self) -> TensorDict:
        """Return batched samples for RSL-RL's model construction."""
        return self._observations.clone()

    def reset(self, *, seed=None, options=None):
        """Supply SB3 normalization samples without stepping physics."""
        super().reset(seed=seed)
        return {
            key: np.full(space.shape, 0.25, dtype=space.dtype) for key, space in self.observation_space.spaces.items()
        }, {}

    def step(self, action):
        """Fail if checkpoint preparation accidentally starts collecting rollouts."""
        raise RuntimeError("Checkpoint creation must not step an environment")


def _create_rsl_rl_checkpoint(env: _CheckpointEnv, cfg, directory: Path) -> Path:
    """Initialize and save the same runner used by the task, without learning."""
    from rsl_rl.runners import OnPolicyRunner

    cfg.device = "cpu"
    runner = OnPolicyRunner(env, to_dict(cfg), log_dir=str(directory), device="cpu")
    # Preserve nontrivial normalization coverage with deterministic synthetic samples.
    observations = env.get_observations()
    runner.alg.actor.update_normalization(observations.apply(lambda value: value + 0.2))
    runner.alg.actor.update_normalization(observations.apply(lambda value: value - 0.4))
    path = directory / "model_0.pt"
    runner.save(str(path), infos={"initialized_for_leapp_export_test": True})
    return path


def _create_rl_games_checkpoint(env: _CheckpointEnv, cfg: dict, directory: Path) -> Path:
    """Use RL-Games' env_info path, which does not construct a vector environment."""
    from rl_games.torch_runner import Runner

    config = cfg["params"]["config"]
    config.update(
        device="cpu",
        device_name="cpu",
        train_dir=str(directory),
        full_experiment_name="initialized",
        num_actors=1,
        minibatch_size=config["horizon_length"],
        env_info={"observation_space": env.observation_space, "action_space": env.action_space, "agents": 1},
    )
    runner = Runner()
    runner.load(cfg)
    agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
    agent.save(str(directory / "model"))
    return directory / "model.pth"


def _create_skrl_checkpoint(env: _CheckpointEnv, cfg: dict, directory: Path) -> Path:
    """Construct skrl's configured models and agent from spaces only."""
    from skrl.utils.runner.torch import Runner

    cfg["trainer"]["close_environment_at_exit"] = False
    cfg["agent"]["experiment"].update(
        directory=str(directory), experiment_name="initialized", write_interval=0, checkpoint_interval=0
    )
    runner = Runner(env, cfg)
    path = directory / "agent.pt"
    runner.agent.save(str(path))
    return path


def _create_sb3_checkpoint(env: _CheckpointEnv, cfg: dict, directory: Path) -> Path:
    """Save PPO and VecNormalize through their standard serialization APIs."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from isaaclab_rl.sb3 import process_sb3_cfg

    cfg = process_sb3_cfg(cfg, num_envs=1)
    policy = cfg.pop("policy")
    cfg.pop("n_timesteps")
    normalize_input = cfg.pop("normalize_input")
    normalize_value = cfg.pop("normalize_value", False)
    clip_obs = cfg.pop("clip_obs", 100.0)
    assert normalize_input  # This case protects saved per-key normalization.
    wrapped = VecNormalize(
        DummyVecEnv([lambda: env]),
        norm_obs=True,
        norm_reward=normalize_value,
        clip_obs=clip_obs,
        gamma=cfg["gamma"],
        clip_reward=np.inf,
    )
    try:
        cfg["device"] = "cpu"
        agent = PPO(policy, wrapped, verbose=0, **cfg)
        wrapped.reset()
        for name, stats in wrapped.obs_rms.items():
            observations = wrapped.get_original_obs()[name]
            stats.update(np.concatenate((observations - 1.0, observations + 1.0)))
        agent.save(str(directory / "model"))
        wrapped.save(str(directory / "model_vecnormalize.pkl"))
    finally:
        wrapped.close()
    return directory / "model.zip"
