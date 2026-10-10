# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts unique to the GR00T adapter; no Kit is launched by the parent test process."""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from skrl.agents.torch import ExperimentCfg
from skrl.agents.torch.ppo import PPO_CFG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch.sequential import SequentialTrainerCfg
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from scripts.reinforcement_learning.gr00t_skrl.agent import Gr00tPPO, RunState, validate_checkpoint
from scripts.reinforcement_learning.gr00t_skrl.policy import (
    ChainPolicy,
    FeatureLayout,
    chain_log_probability,
    module_digest,
)
from scripts.reinforcement_learning.gr00t_skrl.protocol import Request, RpcConnection, Transition


class ScalarPolicy(GaussianMixin, Model):
    """Small genuine skrl Gaussian policy used for independent timeout return calculations."""

    def __init__(self):
        space = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        Model.__init__(self, observation_space=space, action_space=space, device="cpu")
        GaussianMixin.__init__(self, clip_actions=False)
        self.mean = nn.Parameter(torch.zeros(1))
        self.log_std = nn.Parameter(torch.zeros(1))

    def compute(self, inputs: dict, role: str = "") -> tuple:
        return self.mean.expand(inputs["observations"].shape[0], 1), {"log_std": self.log_std}


class ScalarValue(DeterministicMixin, Model):
    """V(s)=s for hand-calculated returns, with a trainable value scale."""

    def __init__(self):
        space = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        Model.__init__(self, observation_space=space, state_space=space, action_space=space, device="cpu")
        DeterministicMixin.__init__(self, clip_actions=False)
        self.scale = nn.Parameter(torch.ones(1))

    def compute(self, inputs: dict, role: str = "") -> tuple:
        return inputs["states"] * self.scale, {}


def make_agent(directory: Path) -> Gr00tPPO:
    """Construct native PPO with adapter lifecycle hooks on a small CPU fixture."""
    policy = ScalarPolicy()
    value = ScalarValue()
    cfg = PPO_CFG(
        rollouts=2,
        learning_epochs=2,
        mini_batches=2,
        discount_factor=0.5,
        learning_rate=0.001,
        time_limit_bootstrap=True,
        experiment=ExperimentCfg(
            directory=str(directory), experiment_name="native", write_interval=0, checkpoint_interval=0
        ),
    )
    agent = Gr00tPPO(
        encoder=None,
        run_state=RunState({"policy_version": "contract", "generation_steps": 2}),
        models={"policy": policy, "value": value},
        memory=RandomMemory(memory_size=2, num_envs=1, device="cpu"),
        observation_space=policy.observation_space,
        state_space=value.state_space,
        action_space=policy.action_space,
        cfg=cfg,
        device="cpu",
    )
    agent.init(trainer_cfg=SequentialTrainerCfg(timesteps=2))
    agent.enable_training_mode(True)
    return agent


def fill_episode_boundaries(agent: Gr00tPPO) -> None:
    """One timeout then simultaneous true termination+timeout, with deliberately different reset states."""
    for index, (state, reset_state, terminal, reward, terminated) in enumerate(
        ((1.0, 100.0, 7.0, 10.0, False), (100.0, 200.0, 9.0, 20.0, True))
    ):
        state = torch.tensor([[state]])
        with torch.no_grad():
            action, _ = agent.act(torch.zeros(1, 1), state, timestep=index, timesteps=2)
        agent.record_transition(
            observations=torch.zeros(1, 1),
            states=state,
            actions=action,
            rewards=torch.tensor([[reward]]),
            next_observations=torch.zeros(1, 1),
            next_states=torch.tensor([[reset_state]]),
            terminated=torch.tensor([[terminated]]),
            truncated=torch.tensor([[True]]),
            infos={"final_state": torch.tensor([[terminal]])},
            timestep=index,
            timesteps=2,
        )


def test_full_chain_probability_and_native_memory() -> None:
    """Native memory retains all chain coordinates, and likelihood sums every Gaussian term."""
    chain = torch.tensor([[[[0.0, 1.0]], [[0.5, 2.0]], [[2.0, 3.0]]]])
    means = torch.tensor([[[[0.0, 1.0]], [[1.0, 2.0]]]], requires_grad=True)
    memory = RandomMemory(memory_size=1, num_envs=1, device="cpu")
    memory.create_tensor(name="actions", size=6, dtype=torch.float32)
    memory.add_samples(actions=chain.flatten(1))
    saved = next(iter(memory.sample(names=["actions"], batch_size=1, mini_batches=1)))[0].reshape(1, 3, 1, 2)
    # Four scalar residuals: 0.5, 1, 1, 1; normalizing constant counted four times.
    expected = -4 * math.log(0.5 * math.sqrt(2 * math.pi)) - (0.25 + 1 + 1 + 1) / (2 * 0.5**2)
    probability = chain_log_probability(saved, means, 0.5)
    assert probability.item() == pytest.approx(expected, abs=1e-6)
    probability.backward()
    torch.testing.assert_close(means.grad.flatten(), torch.tensor([2.0, 4.0, 4.0, 4.0]))
    # Including x_0 contributes exactly the same parameter-independent density to both policies.
    changed = means.detach() + 0.125
    old_density = probability.detach()
    new_density = chain_log_probability(saved, changed, 0.5)
    initial_constant = -math.log(2 * math.pi) - 0.5
    torch.testing.assert_close(
        new_density - old_density, (new_density + initial_constant) - (old_density + initial_constant)
    )
    with pytest.raises(ValueError, match="complete"):
        chain_log_probability(saved[:, 1:], means, 0.5)


def test_original_head_recompute_and_trainable_features() -> None:
    """A small original GR00T head retains gradients through projectors and has no dropout drift."""
    torch.manual_seed(4)
    config = Gr00tN1d7Config(
        hidden_size=8,
        input_embedding_dim=8,
        backbone_embedding_dim=8,
        max_state_dim=2,
        max_action_dim=2,
        action_horizon=2,
        max_num_embodiments=1,
        max_seq_len=8,
        use_vlln=True,
        vl_self_attention_cfg={"num_layers": 1, "num_attention_heads": 1, "attention_head_dim": 8, "dropout": 0.2},
        diffusion_model_cfg={
            "num_layers": 2,
            "num_attention_heads": 1,
            "attention_head_dim": 8,
            "output_dim": 8,
            "dropout": 0.2,
            "interleave_self_attention": True,
        },
    )
    head = Gr00tN1d7ActionHead(config)
    layout = FeatureLayout(4, 8, 2)
    output = BatchFeature(
        data={
            "backbone_features": torch.randn(2, 3, 8),
            "backbone_attention_mask": torch.ones(2, 3, dtype=torch.bool),
            "image_mask": torch.tensor([[True, False, False]]).expand(2, -1),
        }
    )
    observations = layout.pack(output, torch.randn(2, 1, 2), torch.zeros(2, dtype=torch.long))
    policy = ChainPolicy(head, layout, 2, 0.05, "cpu")
    policy.train(True)
    with torch.no_grad():
        chain, sampled = policy.act({"observations": observations})
    _, recomputed = policy.act({"observations": observations, "taken_actions": chain})
    assert chain.shape == (2, 12)
    torch.testing.assert_close(sampled["log_prob"], recomputed["log_prob"], atol=0.01, rtol=0)
    assert not any(module.training for module in policy.modules())
    recomputed["log_prob"].sum().backward()
    for module in (head.state_encoder, head.vlln, head.vl_self_attention, head.model, head.action_decoder):
        assert any(parameter.grad is not None and parameter.grad.abs().max() > 0 for parameter in module.parameters())
    with pytest.raises(ValueError, match="entire saved"):
        policy.act({"observations": observations, "taken_actions": chain[:, 4:]})


def test_terminal_bootstrap_and_new_process_resume(tmp_path: Path) -> None:
    """Native returns isolate resets, and a new process restores models, moments, counters and RNG."""
    torch.manual_seed(7)
    agent = make_agent(tmp_path)
    fill_episode_boundaries(agent)
    agent.post_interaction(timestep=0, timesteps=2)
    agent.post_interaction(timestep=1, timesteps=2)
    # Timeout: 10 + gamma*7. True termination: 20, even with truncated=True.
    torch.testing.assert_close(agent.memory.get_tensor_by_name("returns").flatten(), torch.tensor([13.5, 20.0]))
    assert agent.run_state.optimizer_steps == 4
    checkpoint_path = Path(agent.experiment_dir) / "checkpoints/agent_2.pt"
    before = module_digest(agent.policy)
    # Restore the saved RNG to compute what should follow the checkpoint, without relying on current state.
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    torch.set_rng_state(saved["run_state"]["rng"]["torch"])
    expected_rng = torch.rand(3).tolist()
    first_state = next(iter(saved["optimizer"]["state"].values()))
    expected_moment = first_state["exp_avg"].flatten().tolist()
    child = subprocess.run(
        [sys.executable, str(Path(__file__).absolute()), str(checkpoint_path), str(tmp_path / "resume")],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).absolute().parents[4]), "NO_ALBUMENTATIONS_UPDATE": "1"},
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stderr
    result = json.loads(child.stdout.strip().splitlines()[-1])
    assert result["digest"] == before
    assert result["moment"] == expected_moment
    assert result["random"] == expected_rng
    assert result["optimizer_steps"] == 8
    assert result["environment_steps"] == 4
    with pytest.raises(ValueError, match="Incompatible"):
        validate_checkpoint(str(checkpoint_path), {"policy_version": "contract", "generation_steps": 3})
    with pytest.raises(ValueError, match="requires.*final_state"):
        agent.record_transition(
            observations=torch.zeros(1, 1),
            states=torch.zeros(1, 1),
            actions=torch.zeros(1, 1),
            rewards=torch.zeros(1, 1),
            next_observations=torch.zeros(1, 1),
            next_states=torch.zeros(1, 1),
            terminated=torch.tensor([[False]]),
            truncated=torch.tensor([[True]]),
            infos={},
            timestep=2,
            timesteps=4,
        )


def test_numpy_versions_and_rpc_deadline(tmp_path: Path) -> None:
    """Real NumPy 2 simulator-side arrays arrive intact in NumPy 1 model-side RPC, and EOF cannot hang."""
    repo = Path(__file__).absolute().parents[4]
    socket_path = str(tmp_path / "transport.sock")
    script = """
import socket
import numpy as np
from scripts.reinforcement_learning.gr00t_skrl.protocol import Observation, RpcConnection, Transition
connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
connection.connect(SOCKET_PATH)
rpc = RpcConnection(connection, 15)
assert np.__version__.startswith("2.")
rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
state = np.arange(8, dtype=np.float32)
rpc.send(Transition(Observation(rgb, rgb.copy(), state), truncated=True, final_state=state + 7))
request = rpc.receive()
request.validate()
np.testing.assert_array_equal(request.action, np.array([0.1, -0.1, 0, 0, 0, 0, 1], np.float32))
rpc.close()
""".replace("SOCKET_PATH", repr(socket_path))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(socket_path)
        server.listen(1)
        server.settimeout(15)
        child = subprocess.Popen(
            ["uv", "run", "--project", str(repo), "--no-sync", "python", "-c", script],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            connection, _ = server.accept()
            rpc = RpcConnection(connection, 15)
            reply = rpc.receive()
            assert isinstance(reply, Transition)
            reply.observation.validate()
            np.testing.assert_array_equal(reply.observation.table_rgb, np.arange(18, dtype=np.uint8).reshape(2, 3, 3))
            np.testing.assert_array_equal(reply.final_state, np.arange(8, dtype=np.float32) + 7)
            rpc.send(Request("step", np.array([0.1, -0.1, 0, 0, 0, 0, 1], np.float32)))
            _, stderr = child.communicate(timeout=15)
            assert child.returncode == 0, stderr
            with pytest.raises(ConnectionError, match="disconnected"):
                rpc.receive()
            rpc.close()
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=15)
    receiver, sender = socket.socketpair()
    with receiver, sender:
        rpc = RpcConnection(receiver, 0.05)
        with pytest.raises(TimeoutError):
            rpc.receive()


if __name__ == "__main__":
    checkpoint_path, output_dir = sys.argv[1:]
    agent = make_agent(Path(output_dir))
    agent.resume(checkpoint_path)
    digest = module_digest(agent.policy)
    moment = next(iter(agent.optimizer.state.values()))["exp_avg"].flatten().tolist()
    agent.run_state.restore_rng()
    restored_random = torch.rand(3).tolist()
    fill_episode_boundaries(agent)
    agent.post_interaction(timestep=0, timesteps=2)
    agent.post_interaction(timestep=1, timesteps=2)
    print(
        json.dumps(
            {
                "digest": digest,
                "moment": moment,
                "random": restored_random,
                "optimizer_steps": agent.run_state.optimizer_steps,
                "environment_steps": agent.run_state.environment_steps,
            }
        )
    )
