# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline NeRD fitting from native RecorderManager episodes."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import h5py
import torch
import torch.distributed as dist
from isaaclab_newton.physics.nerd import JointDynamics
from physicsnemo_robotics.dynamics import FieldNormalizer, RigidContactSetEncoder

from isaaclab.utils import configclass, string_to_callable


@configclass
class NeRDTrainCfg:
    """Offline model fitting settings; the dataset owns state, control and frame conventions."""

    max_updates: int = 30000
    batch_size: int = 1024
    learning_rate: float = 1.0e-3
    final_learning_rate: float = 1.0e-4
    learning_rate_decay_updates: int = 30000
    gradient_clip_norm: float = 1.0
    validation_fraction: float = 0.1
    seed: int = 7
    compile: bool = True
    network_type: str = "physicsnemo_robotics.models.parallel_nerd:ParallelNeRD"
    """Network class with a temporal_attention_spans configuration and a predict_last method."""
    precision: str = "float32"
    """Projection precision: float32, float16 or bfloat16; parameters and losses remain FP32."""
    max_gradient_retries: int = 8
    network: dict = {
        "num_heads": 4,
        "head_dim": 64,
        "intermediate_size": 640,
        "head_hidden": 256,
        "temporal_attention_spans": (4, 3, 3, 3),
        "rms_norm_eps": 1.0e-6,
        "linear_probe": True,
    }


class _PredictLast(torch.nn.Module):
    """Expose the newest prediction through forward so DDP owns gradient synchronization."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, inputs, contacts=None):
        return (
            self.network.predict_last(inputs, contacts) if contacts is not None else self.network.predict_last(inputs)
        )


class OfflineNeRD:
    """Own flattened transition tensors, boundary-aware history sampling and model optimization.

    Episodes, rather than overlapping windows, are split into train and validation sets. The
    terminal transition is retained. Every frame after an episode boundary starts a repeated
    first-frame prefix, matching deployed solver resets. Storage grows with transitions, not
    with the product of transitions and history length.
    """

    def __init__(self, datasets: list[Path], cfg: NeRDTrainCfg, device: str, *, checkpoint=None):
        if cfg.max_updates < 1 or cfg.batch_size < 1 or not 0 < cfg.validation_fraction < 1:
            raise ValueError("Positive update/batch counts and a validation fraction in (0, 1) are required.")
        if cfg.learning_rate_decay_updates < 1 or any(
            not math.isfinite(value) or value <= 0
            for value in (cfg.learning_rate, cfg.final_learning_rate, cfg.gradient_clip_norm)
        ):
            raise ValueError("Learning rates, decay duration and gradient clip norm must be positive and finite.")
        self.cfg, self.device = cfg, device
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        if cfg.batch_size % self.world_size:
            raise ValueError("The global batch size must be divisible by the distributed world size.")
        if cfg.precision not in ("float32", "float16", "bfloat16") or cfg.max_gradient_retries < 1:
            raise ValueError("Select float32, float16 or bfloat16 and a positive gradient retry limit.")
        self.dtype = getattr(torch, cfg.precision)
        self.device_type = torch.device(device).type
        if self.dtype == torch.float16 and self.device_type != "cuda":
            raise ValueError("FP16 fitting requires CUDA.")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.dtype == torch.float16)
        self.generator = torch.Generator(device=device).manual_seed(cfg.seed + self.rank)
        self.updates = 0
        self.identity = []
        self.dynamics = None
        episodes = []
        for path in datasets:
            path = Path(path)
            metadata = json.loads(path.with_suffix(".json").read_text())
            descriptor = metadata["dynamics"]
            if self.dynamics is None:
                self.dynamics = JointDynamics(descriptor)
            elif self.dynamics.descriptor != descriptor:
                raise ValueError("Datasets must use the same dynamics representation and physical model.")
            with path.open("rb") as stream:
                fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
            if fingerprint in self.identity:
                raise ValueError("Duplicate datasets would leak episodes between training and validation.")
            self.identity.append(fingerprint)
            with h5py.File(path, "r") as file:
                for name in file["data"]:
                    data = file["data"][name]["transition"]
                    state, successor, control, contacts = (
                        torch.from_numpy(data[key][:]).to(device)
                        for key in ("state", "next_state", "control", "contacts")
                    )
                    if not len(state):
                        continue
                    if not all(torch.isfinite(value).all() for value in (state, successor, control, contacts)):
                        raise ValueError(f"Nonfinite transition in {path}:{name}.")
                    if not torch.equal(successor[:-1], state[1:]):
                        raise ValueError(
                            f"Discontinuous episode in {path}:{name}; a reset cannot enter a training window."
                        )
                    if data["terminated"][:-1].any() or data["truncated"][:-1].any():
                        raise ValueError(f"Episode boundary inside {path}:{name}.")
                    inputs = self.dynamics.inputs(state, control, contacts)
                    targets = self.dynamics.codec.state_to_delta(state, successor)
                    episodes.append((inputs, targets, contacts))
        if len(episodes) < 2:
            raise ValueError("Record at least two nonempty episodes to split training and validation.")
        order = torch.randperm(len(episodes), generator=torch.Generator().manual_seed(cfg.seed)).tolist()
        validation_count = max(1, min(len(episodes) - 1, round(cfg.validation_fraction * len(episodes))))
        self.validation_episodes = order[:validation_count]
        self.train_episodes = order[validation_count:]
        self.inputs = torch.cat([item[0] for item in episodes])
        self.targets = torch.cat([item[1] for item in episodes])
        self.contacts = None
        lengths = torch.tensor([len(item[0]) for item in episodes], device=device)
        starts = lengths.cumsum(0) - lengths
        self.starts = torch.repeat_interleave(starts, lengths)
        ids = torch.repeat_interleave(torch.arange(len(episodes), device=device), lengths)
        held_out = torch.isin(ids, ids.new_tensor(self.validation_episodes))
        self.train_ids, self.validation_ids = (~held_out).nonzero().flatten(), held_out.nonzero().flatten()
        contacts = torch.cat([item[2] for item in episodes]) if self.dynamics.max_contacts else None
        if checkpoint is None:
            train_x, train_y = self.inputs[self.train_ids], self.targets[self.train_ids]
            means, stds = (
                {"input": train_x.mean(0), "target": train_y.mean(0)},
                {"input": train_x.std(0, correction=0), "target": train_y.std(0, correction=0)},
            )
            if contacts is not None:
                training = contacts[self.train_ids]
                geometry = training[..., 4:][training[..., 0] == 1]
                if not len(geometry):
                    raise ValueError("Contact-conditioned training requires contacts in its training split.")
                means["contact"] = torch.cat((geometry.new_zeros(4), geometry.mean(0)))
                stds["contact"] = torch.cat((geometry.new_ones(4), geometry.std(0, correction=0)))
            self.normalizer = FieldNormalizer(means, stds)
        else:
            if (
                checkpoint["format"] != "isaaclab-nerd-v3"
                or checkpoint["datasets"] != self.identity
                or checkpoint["dynamics"] != self.dynamics.descriptor
                or checkpoint["train_episodes"] != self.train_episodes
                or checkpoint["network_cfg"] != cfg.network
                or checkpoint["network_type"] != cfg.network_type
                or len(checkpoint["generators"]) != self.world_size
            ):
                raise ValueError(
                    "Resume requires v3 canonical inputs and the same datasets, split, network and rank count."
                )
            self.normalizer = FieldNormalizer.from_dict(checkpoint["normalizer"])
        self.inputs = self.normalizer.normalize("input", self.inputs)
        self.targets = self.normalizer.normalize("target", self.targets)
        if contacts is not None:
            # IDs/masks are categorical: normalize geometry, then restore them bit-exactly.
            normalized = self.normalizer.normalize_with_passthrough("contact", contacts, (0, 1, 2, 3))
            normalized = torch.where(contacts[..., :1] == 1, normalized, 0.0)
            self.contacts = RigidContactSetEncoder.to_variable_contact_set(normalized)
        history = 1 + sum(span - 1 for span in cfg.network["temporal_attention_spans"])
        self.offsets = torch.arange(1 - history, 1, device=device)
        torch.manual_seed(cfg.seed)
        self.model = string_to_callable(cfg.network_type)(
            self.dynamics.input_dim,
            self.dynamics.codec.prediction_shape[0],
            **cfg.network,
            **self.dynamics.contact_model_cfg,
        ).to(device)
        self.predict = _PredictLast(self.model)
        if self.world_size > 1:
            device_ids = [torch.device(device).index] if self.device_type == "cuda" else None
            self.predict = torch.nn.parallel.DistributedDataParallel(self.predict, device_ids=device_ids)
        if cfg.compile:
            self.predict = torch.compile(self.predict)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=0.0)
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.generator.set_state(checkpoint["generators"][self.rank].cpu())
            self.scaler.load_state_dict(checkpoint["scaler"])
            self.updates = checkpoint["updates"]

    def sample(self, ids, count):
        """Draw transitions uniformly, with a reset prefix that never crosses episode boundaries."""
        selected = ids[torch.randint(len(ids), (count,), device=self.device, generator=self.generator)]
        history = (selected[:, None] + self.offsets).clamp_min(self.starts[selected, None])
        return (
            self.inputs[history],
            self.targets[selected],
            self.contacts[history] if self.contacts is not None else None,
        )

    def update(self):
        self.model.train()
        self.optimizer.param_groups[0]["lr"] = self.cfg.learning_rate + (
            self.cfg.final_learning_rate - self.cfg.learning_rate
        ) * min(self.updates / self.cfg.learning_rate_decay_updates, 1.0)
        x, y, contacts = self.sample(self.train_ids, self.cfg.batch_size // self.world_size)
        for _ in range(self.cfg.max_gradient_retries):
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device_type, dtype=self.dtype, enabled=self.dtype != torch.float32):
                prediction = self.predict(x, contacts)
            loss = (prediction.float() - y).square().mean()
            finite = torch.isfinite(loss).to(torch.int32)
            if self.world_size > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite:
                raise FloatingPointError(f"Nonfinite NeRD training loss at update {self.updates}.")
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.gradient_clip_norm, error_if_nonfinite=not self.scaler.is_enabled()
            )
            scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() >= scale:
                break
        else:
            raise FloatingPointError(f"NeRD gradients overflowed repeatedly at update {self.updates}.")
        if self.world_size > 1:
            dist.all_reduce(loss.detach())
            loss.detach().div_(self.world_size)
        self.updates += 1
        return {"train/normalized_mse": float(loss.detach()), "train/gradient_norm": float(norm)}

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        totals = torch.zeros(3, dtype=torch.float64, device=self.device)
        for ids in self.validation_ids[self.rank :: self.world_size].split(self.cfg.batch_size):
            if not len(ids):
                continue
            history = (ids[:, None] + self.offsets).clamp_min(self.starts[ids, None])
            with torch.autocast(self.device_type, dtype=self.dtype, enabled=self.dtype != torch.float32):
                contacts = self.contacts[history] if self.contacts is not None else None
                prediction = (
                    self.model.predict_last(self.inputs[history], contacts)
                    if contacts is not None
                    else self.model.predict_last(self.inputs[history])
                )
            error = prediction.float() - self.targets[ids]
            totals[0] += error.square().sum()
            totals[1] += (error * (self.normalizer.stds["target"].to(error) + self.normalizer.eps)).square().sum()
            totals[2] += error.numel()
        if self.world_size > 1:
            dist.all_reduce(totals)
        return {
            "validation/normalized_mse": float(totals[0] / totals[2]),
            "validation/delta_mse": float(totals[1] / totals[2]),
        }

    def checkpoint(self):
        """Return a deployable and resumable snapshot; the composition root publishes it atomically."""
        generators = [self.generator.get_state().cpu()]
        if self.world_size > 1:
            generators = [None] * self.world_size
            dist.all_gather_object(generators, self.generator.get_state().cpu())
        return {
            "format": "isaaclab-nerd-v3",
            "dynamics": self.dynamics.descriptor,
            "datasets": self.identity,
            "train_episodes": self.train_episodes,
            "network_cfg": self.cfg.network,
            "network_type": str(self.cfg.network_type),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "normalizer": self.normalizer.to_dict(),
            "updates": self.updates,
            "generators": generators,
            "scaler": self.scaler.state_dict(),
        }
