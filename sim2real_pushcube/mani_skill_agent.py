"""ManiSkill PushCube PPO (RGB) network + checkpoint loader.

Faithful re-implementation of the ``Agent`` / ``NatureCNN`` defined in
``ManiSkill/examples/baselines/ppo/ppo_rgb.py`` so that a checkpoint saved via
``torch.save(agent.state_dict(), path)`` can be loaded back with
``load_state_dict``. The module hierarchy (and therefore the state_dict keys)
is kept identical to the original.

Contract (derived from ManiSkill PushCube-v1 + ppo_rgb.py):
- obs is a dict: ``{"rgb": uint8[N,128,128,3] (HWC), "state": float32[N,35]}``
- rgb is normalized by /255 inside the network.
- state order: [qpos(9), qvel(9), tcp_pose(7), goal_pos(3), obj_pose(7)] = 35.
- action: float32[N,8] = [arm_delta(7), gripper_abs_target(1)].
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal weight init (matches ppo_rgb.py). Init values are irrelevant once
    the state_dict is loaded, but the module structure must match exactly."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    """RGB(+state) encoder. Keys: ``extractors.rgb.0.*`` (cnn), ``extractors.rgb.1.*``
    (fc), ``extractors.state.*`` (linear)."""

    def __init__(self, sample_obs):
        super().__init__()
        extractors = {}
        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgb"].shape[-1]

        cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # figure out flattened dim with a dummy forward (NHWC -> NCHW)
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0, 3, 1, 2).cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded = []
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0, 3, 1, 2)
                obs = obs / 255.0
            encoded.append(extractor(obs))
        return torch.cat(encoded, dim=1)


class Agent(nn.Module):
    """Actor-critic. Keys: ``feature_net.*``, ``actor_mean.{0,2}.*``, ``actor_logstd``,
    ``critic.{0,2}.*``. action_dim=8 for PushCube-v1 with pd_joint_delta_pos."""

    def __init__(self, sample_obs, action_dim: int):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        latent_size = self.feature_net.out_features  # 512 when rgb+state
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_features(self, x):
        return self.feature_net(x)

    def get_value(self, x):
        return self.critic(self.feature_net(x))

    @torch.no_grad()
    def get_action(self, x, deterministic: bool = True):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std).sample()

    def get_action_and_value(self, x, action=None):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def build_agent(
    ckpt_path: str,
    num_envs: int,
    device: torch.device,
    rgb_shape=(128, 128, 3),
    state_dim: int = 35,
    action_dim: int = 8,
) -> Agent:
    """Build the Agent with dummy sample obs and load the ManiSkill state_dict.

    Args:
        ckpt_path: Path to a ``.pt`` file saved by ppo_rgb.py
            (``torch.save(agent.state_dict(), ...)``).
        num_envs: Number of parallel envs (sets the leading dim of sample obs;
            only affects dummy init, not the loaded weights).
        device: Device to place the network on.
        rgb_shape: (H, W, C) of the policy's rgb input. Default (128,128,3).
        state_dim: Length of the state vector. Default 35 for PushCube-v1.
        action_dim: Action dimension. Default 8 for pd_joint_delta_pos on Panda.

    Returns:
        The Agent in eval mode, with weights loaded and ready for
        ``agent.get_action(obs, deterministic=True)``.
    """
    sample_obs = {
        "rgb": torch.zeros((num_envs, *rgb_shape), dtype=torch.uint8),
        "state": torch.zeros((num_envs, state_dim), dtype=torch.float32),
    }
    agent = Agent(sample_obs=sample_obs, action_dim=action_dim).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent
