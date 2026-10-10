# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Original N1.7 head exposed as a stochastic full-chain skrl policy."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import partial

import gr00t.model  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
from skrl.models.torch import DeterministicMixin, Model
from torch import nn
from torch.distributions import Normal
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModel, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature

from .protocol import Observation, RunConfig

STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def module_digest(module: nn.Module) -> str:
    """Fingerprint every parameter without keeping a second model on the GPU."""
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def chain_log_probability(chain: torch.Tensor, means: torch.Tensor, sigma: float) -> torch.Tensor:
    """Joint FP32 transition log density, omitting the parameter-independent x_0 density."""
    if chain.ndim != 4 or chain.shape[1] != means.shape[1] + 1 or chain[:, 1:].shape != means.shape:
        raise ValueError("Expected a complete [batch, K+1, H, D] chain and K conditional means")
    return Normal(means.float(), sigma).log_prob(chain[:, 1:].float()).sum(dim=(1, 2, 3)).unsqueeze(-1)


@dataclass(frozen=True)
class FeatureLayout:
    """Fixed packing of raw frozen features, masks and normalized processor state."""

    max_tokens: int
    embedding_dim: int
    state_dim: int

    @property
    def size(self) -> int:
        """Total number of scalars in the skrl observation."""
        return self.max_tokens * (self.embedding_dim + 2) + self.state_dim + 1

    def pack(self, output: BatchFeature, state: torch.Tensor, embodiment: torch.Tensor) -> torch.Tensor:
        """Pad frozen features; reject overflow rather than silently truncate tokens."""
        features = output["backbone_features"]
        batch, tokens, width = features.shape
        if tokens > self.max_tokens or width != self.embedding_dim or state[0].numel() != self.state_dim:
            raise ValueError(f"Features exceed or disagree with fixed layout: {features.shape}, {state.shape}")
        packed = torch.zeros(batch, self.size, device=features.device, dtype=torch.float32)
        offset = self.max_tokens * self.embedding_dim
        packed[:, :offset].view(batch, self.max_tokens, self.embedding_dim)[:, :tokens] = features.float()
        packed[:, offset : offset + tokens] = output["backbone_attention_mask"].float()
        packed[:, offset + self.max_tokens : offset + self.max_tokens + tokens] = output["image_mask"].float()
        packed[:, offset + 2 * self.max_tokens : -1] = state.reshape(batch, -1).float()
        packed[:, -1] = embodiment.flatten().float()
        return packed.detach()

    def unpack(self, observations: torch.Tensor, dtype: torch.dtype) -> tuple[BatchFeature, BatchFeature]:
        """Recover raw backbone output; trainable projectors run afterwards."""
        if observations.ndim != 2 or observations.shape[-1] != self.size:
            raise ValueError("Observation disagrees with fixed feature layout")
        batch = observations.shape[0]
        offset = self.max_tokens * self.embedding_dim
        mask = observations[:, offset : offset + self.max_tokens].bool()
        # Retain the longest real sequence in the minibatch, including internal padding.
        positions = torch.arange(1, self.max_tokens + 1, device=observations.device)
        tokens = int((positions * mask).max().item())
        if not tokens:
            raise ValueError("Empty backbone attention mask")
        backbone = BatchFeature(
            data={
                "backbone_features": observations[:, :offset]
                .reshape(batch, self.max_tokens, self.embedding_dim)[:, :tokens]
                .to(dtype),
                "backbone_attention_mask": mask[:, :tokens],
                "image_mask": observations[:, offset + self.max_tokens : offset + 2 * self.max_tokens][
                    :, :tokens
                ].bool(),
            }
        )
        action = BatchFeature(
            data={
                "state": observations[:, offset + 2 * self.max_tokens : -1].reshape(batch, 1, self.state_dim).to(dtype),
                "embodiment_id": observations[:, -1].long(),
            }
        )
        return backbone, action


class FrozenEncoder:
    """Own the frozen backbone and processor outside policy/optimizer/checkpoint modules."""

    def __init__(self, cfg: RunConfig, device: str = "cuda:0"):
        config = AutoConfig.from_pretrained(cfg.model_path, local_files_only=True)
        config.model_name = cfg.backbone_path
        model: Gr00tN1d7 = AutoModel.from_pretrained(cfg.model_path, config=config, local_files_only=True)
        model.to(device=device, dtype=torch.bfloat16).eval()
        model.backbone.requires_grad_(False)
        model.action_head.set_trainable_parameters(True, True, True)
        self.model = model
        self.backbone = model.backbone
        self.head = model.action_head
        self.processor = AutoProcessor.from_pretrained(cfg.model_path, local_files_only=True)
        self.processor.eval()
        self.embodiment = EmbodimentTag.resolve("libero_sim")
        self.layout = FeatureLayout(
            cfg.max_tokens, config.backbone_embedding_dim, config.max_state_dim * config.state_history_length
        )
        if config.state_history_length != 1:
            raise ValueError("This runner requires the micro-SFT single-state-history processor")
        self.device = torch.device(device)
        self.instruction = cfg.instruction
        self.last_states: dict[str, np.ndarray] = {}
        for transformer in (self.head.model, self.head.vl_self_attention):
            for block in transformer.transformer_blocks:
                # Preserve upstream parameters and state_dict names; checkpoint the original block forward.
                block.forward = partial(checkpoint, block.forward, use_reentrant=False)

    def encode(self, observation: Observation) -> torch.Tensor:
        """Process RGB/state and cache only frozen backbone outputs (no autograd graph)."""
        observation.validate()
        self.last_states = {key: observation.state[i : i + 1].reshape(1, 1, 1) for i, key in enumerate(STATE_KEYS[:-1])}
        self.last_states["gripper"] = observation.state[6:].reshape(1, 1, 2)
        step = VLAStepData(
            images={"image": observation.table_rgb[None], "wrist_image": observation.wrist_rgb[None]},
            states={key: value[0] for key, value in self.last_states.items()},
            actions={},
            text=self.instruction,
            embodiment=self.embodiment,
        )
        processed = self.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
        collated = self.processor.collator([processed])
        with torch.no_grad():
            backbone_input, action_input = self.model.prepare_input(collated["inputs"])
            output = self.backbone(backbone_input)
            return self.layout.pack(output, action_input["state"], action_input["embodiment_id"])

    def decode(self, action: torch.Tensor) -> np.ndarray:
        """Decode x_K and execute the first relative IK command [m, rad] with binary gripper."""
        decoded = self.processor.decode_action(action.detach().float().cpu().numpy(), self.embodiment, self.last_states)
        command = np.array([decoded[key][0, 0, 0] for key in STATE_KEYS], dtype=np.float32)
        command[:6] = np.clip(command[:6], -0.1, 0.1)
        command[6] = 1.0 if command[6] > 0 else -1.0
        if not np.isfinite(command).all():
            raise ValueError("Processor decoded a non-finite robot command")
        return command

    def offload(self) -> None:
        """Free GPU backbone allocations before native PPO updates or checkpoint loading."""
        self.backbone.to("cpu")
        torch.cuda.empty_cache()

    def restore(self) -> None:
        """Return the frozen backbone to the inference device."""
        self.backbone.to(self.device)


class ChainPolicy(Model):
    """Full original head, with full-chain actions and teacher-forced likelihood."""

    def __init__(
        self, head: Gr00tN1d7ActionHead, layout: FeatureLayout, generation_steps: int, sigma: float, device: str
    ):
        self.horizon = head.config.action_horizon
        self.action_dim = head.config.max_action_dim
        self.generation_steps = generation_steps
        self.sigma = sigma
        observation_space = gym.spaces.Box(-np.inf, np.inf, (layout.size,), dtype=np.float32)
        action_space = gym.spaces.Box(
            -np.inf, np.inf, ((generation_steps + 1) * self.horizon * self.action_dim,), dtype=np.float32
        )
        super().__init__(observation_space=observation_space, action_space=action_space, device=device)
        self.head = head
        self.layout = layout
        self.recompute_count = 0
        self.recomputed_versions: set[int] = set()
        self.train(False)

    def train(self, mode: bool = True) -> ChainPolicy:
        """Disable all dropout in sampling and updates, while retaining autograd."""
        super().train(False)
        return self

    def compute(self, inputs: dict[str, torch.Tensor], role: str = "") -> tuple[torch.Tensor, dict]:
        """Implement the skrl model computation contract."""
        return self.act(inputs, role=role)

    def act(self, inputs: dict[str, torch.Tensor], role: str = "") -> tuple[torch.Tensor, dict]:
        """Sample a complete chain, or recompute the likelihood of supplied taken_actions."""
        backbone, action = self.layout.unpack(inputs["observations"], next(self.head.parameters()).dtype)
        features = self.head._encode_features(backbone, action)
        taken = inputs.get("taken_actions")
        batch = inputs["observations"].shape[0]
        if taken is not None:
            if taken.shape != (batch, self.num_actions):
                raise ValueError("taken_actions must contain the entire saved generation chain")
            chain = taken.reshape(batch, self.generation_steps + 1, self.horizon, self.action_dim)
            self.recompute_count += 1
            self.recomputed_versions.add(next(self.head.action_decoder.parameters())._version)
        else:
            chain = torch.randn(batch, 1, self.horizon, self.action_dim, device=self.device, dtype=torch.float32)
        means = []
        for index in range(self.generation_steps):
            current = chain[:, index]
            timesteps = torch.full(
                (batch,),
                int(index / self.generation_steps * self.head.num_timestep_buckets),
                device=self.device,
                dtype=torch.long,
            )
            action_features = self.head.action_encoder(
                current.to(features["state_features"].dtype), timesteps, action["embodiment_id"]
            )
            if self.head.config.add_pos_embed:
                positions = torch.arange(self.horizon, device=self.device)
                action_features = action_features + self.head.position_embedding(positions)[None]
            hidden = torch.cat((features["state_features"], action_features), dim=1)
            kwargs = {}
            if self.head.config.use_alternate_vl_dit:
                kwargs = {
                    "image_mask": backbone["image_mask"],
                    "backbone_attention_mask": backbone["backbone_attention_mask"],
                }
            output = self.head.model(
                hidden_states=hidden, encoder_hidden_states=features["backbone_features"], timestep=timesteps, **kwargs
            )
            velocity = self.head.action_decoder(output, action["embodiment_id"])[:, -self.horizon :]
            mean = current.float() + velocity.float() / self.generation_steps
            means.append(mean)
            if taken is None:
                following = mean + self.sigma * torch.randn_like(mean)
                chain = torch.cat((chain, following[:, None]), dim=1)
        probability = chain_log_probability(chain, torch.stack(means, dim=1), self.sigma)
        if not torch.isfinite(probability).all():
            raise FloatingPointError("Non-finite generation-chain likelihood")
        return chain.flatten(1), {"log_prob": probability}

    def distribution(self, *, role: str = "") -> Normal:
        """Native std diagnostic describes conditional transition noise, not physical actions."""
        return Normal(torch.zeros(1, device=self.device), self.sigma)


class StateCritic(DeterministicMixin, Model):
    """Small state-only value network; inputs are XYZ [m], rotation vector [rad], fingers [m]."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, device: str):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32),
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(nn.Linear(8, 64), nn.Tanh(), nn.Linear(64, 1)).to(device)

    def compute(self, inputs: dict[str, torch.Tensor], role: str = "") -> tuple[torch.Tensor, dict]:
        """Predict value from the independent critic state."""
        return self.net(inputs["states"].float()), {}
