# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""skrl environment wrapper bridging full-chain actions to seven physical controls."""

from __future__ import annotations

import socket
import time

import gymnasium as gym
import numpy as np
import torch
from skrl.envs.wrappers.torch import Wrapper

from .policy import ChainPolicy, FrozenEncoder
from .protocol import Request, RpcConnection, RunConfig, Transition


class RemoteEnvironment(Wrapper):
    """Single auto-reset environment with frozen-feature observations and state-only critic inputs."""

    def __init__(self, cfg: RunConfig, encoder: FrozenEncoder, policy: ChainPolicy):
        # The remote interface itself owns the skrl spaces; no local Gym environment is built.
        self._device = encoder.device
        self._observation_space = policy.observation_space
        self._action_space = policy.action_space
        self._state_space = gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)
        self.encoder = encoder
        self.policy = policy
        self._state = None
        self._observations = None
        self._autoreset_pending = False
        self._closed = False
        self.metrics: list[dict] = []
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(cfg.rpc_timeout)
        deadline = time.monotonic() + cfg.rpc_timeout
        try:
            while True:
                try:
                    connection.connect(cfg.socket_path)
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Simulation did not accept a model connection") from None
                    time.sleep(0.1)
        except BaseException:
            connection.close()
            raise
        self.rpc = RpcConnection(connection, cfg.rpc_timeout)

    @property
    def num_envs(self) -> int:
        """The runner intentionally uses one physical environment."""
        return 1

    @property
    def num_agents(self) -> int:
        """One native PPO agent controls the robot."""
        return 1

    @property
    def observation_space(self) -> gym.Space:
        """Fixed frozen-feature observation space."""
        return self._observation_space

    @property
    def state_space(self) -> gym.Space:
        """Eight-dimensional physical critic state."""
        return self._state_space

    @property
    def action_space(self) -> gym.Space:
        """Complete stochastic generation-chain space stored by native memory."""
        return self._action_space

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reuse the received auto-reset observation when native trainer requests reset."""
        if self._autoreset_pending:
            self._autoreset_pending = False
            return self._observations, {}
        reply = self._request(Request("reset"))
        return self._encode(reply), {}

    def state(self) -> torch.Tensor:
        """Return the normal, possibly reset-after-terminal critic state."""
        return self._state

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Decode only x_K's first action; preserve final critic state independently."""
        expected = (1, self.policy.num_actions)
        if actions.shape != expected or not torch.isfinite(actions).all():
            raise ValueError(f"Expected finite full-chain actions of shape {expected}")
        chain = actions.reshape(1, self.policy.generation_steps + 1, self.policy.horizon, self.policy.action_dim)
        command = self.encoder.decode(chain[:, -1])
        reply = self._request(Request("step", command))
        if not np.array_equal(reply.executed_action, command):
            raise RuntimeError("Simulator did not execute the requested seven-dimensional command")
        next_observations = self._encode(reply)
        infos = {}
        if reply.final_state is not None:
            if reply.final_state.shape != (8,) or not np.isfinite(reply.final_state).all():
                raise ValueError("Invalid terminal critic state")
            infos["final_state"] = torch.as_tensor(reply.final_state, device=self.device).reshape(1, 8)
        self._autoreset_pending = reply.terminated or reply.truncated
        self.metrics.append(
            {
                "command": command.tolist(),
                "motion_m": reply.motion,
                "reward": reply.reward,
                "terminated": reply.terminated,
                "truncated": reply.truncated,
                "table_rgb_std": float(reply.observation.table_rgb.std()),
                "wrist_rgb_std": float(reply.observation.wrist_rgb.std()),
            }
        )
        tensors = [
            torch.tensor([[value]], device=self.device) for value in (reply.reward, reply.terminated, reply.truncated)
        ]
        return next_observations, *tensors, infos

    def render(self) -> None:
        """Cameras are rendered inside the simulation child."""

    def close(self) -> None:
        """Close gracefully when possible; always release the local socket."""
        if not self._closed:
            self._closed = True
            try:
                self.rpc.send(Request("close"))
            except OSError:
                pass
            finally:
                self.rpc.close()

    def _request(self, request: Request) -> Transition:
        request.validate()
        self.rpc.send(request)
        reply = self.rpc.receive()
        if isinstance(reply, str):
            raise RuntimeError(f"Simulation failed: {reply}")
        if not isinstance(reply, Transition) or not np.isfinite(reply.reward):
            raise ValueError("Invalid simulation reply")
        return reply

    def _encode(self, reply: Transition) -> torch.Tensor:
        self._state = torch.as_tensor(reply.observation.state, device=self.device).reshape(1, 8)
        self._observations = self.encoder.encode(reply.observation)
        return self._observations
