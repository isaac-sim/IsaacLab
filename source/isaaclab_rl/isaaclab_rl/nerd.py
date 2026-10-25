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
from isaaclab_newton.physics.nerd import JointDynamics
from physicsnemo_robotics.models.parallel_nerd import ParallelNeRD

from isaaclab.utils import configclass


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
    network: dict = {
        "num_heads": 4,
        "head_dim": 64,
        "intermediate_size": 640,
        "head_hidden": 256,
        "temporal_attention_spans": (4, 3, 3, 3),
        "rms_norm_eps": 1.0e-6,
        "linear_probe": True,
    }


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
        self.generator = torch.Generator(device=device).manual_seed(cfg.seed)
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
                    episodes.append((inputs, targets))
        if len(episodes) < 2:
            raise ValueError("Record at least two nonempty episodes to split training and validation.")
        order = torch.randperm(len(episodes), generator=torch.Generator().manual_seed(cfg.seed)).tolist()
        validation_count = max(1, min(len(episodes) - 1, round(cfg.validation_fraction * len(episodes))))
        self.validation_episodes = order[:validation_count]
        self.train_episodes = order[validation_count:]
        self.inputs = torch.cat([item[0] for item in episodes])
        self.targets = torch.cat([item[1] for item in episodes])
        lengths = torch.tensor([len(item[0]) for item in episodes], device=device)
        starts = lengths.cumsum(0) - lengths
        self.starts = torch.repeat_interleave(starts, lengths)
        ids = torch.repeat_interleave(torch.arange(len(episodes), device=device), lengths)
        held_out = torch.isin(ids, ids.new_tensor(self.validation_episodes))
        self.train_ids, self.validation_ids = (~held_out).nonzero().flatten(), held_out.nonzero().flatten()
        if checkpoint is None:
            train_x, train_y = self.inputs[self.train_ids], self.targets[self.train_ids]
            self.input_mean, self.target_mean = train_x.mean(0), train_y.mean(0)
            self.input_scale = train_x.std(0, correction=0) + 1.0e-6
            self.target_scale = train_y.std(0, correction=0) + 1.0e-6
        else:
            if (
                checkpoint["datasets"] != self.identity
                or checkpoint["dynamics"] != self.dynamics.descriptor
                or checkpoint["train_episodes"] != self.train_episodes
                or checkpoint["network_cfg"] != cfg.network
            ):
                raise ValueError("Resume requires the same datasets, representation, split and network.")
            # The checkpoint owns fixed normalization, including across GPU/runtime changes.
            for name in ("input_mean", "input_scale", "target_mean", "target_scale"):
                setattr(self, name, checkpoint[name].to(device))
        self.inputs.sub_(self.input_mean).div_(self.input_scale)
        self.targets.sub_(self.target_mean).div_(self.target_scale)
        history = 1 + sum(span - 1 for span in cfg.network["temporal_attention_spans"])
        self.offsets = torch.arange(1 - history, 1, device=device)
        torch.manual_seed(cfg.seed)
        self.model = ParallelNeRD(self.dynamics.input_dim, self.dynamics.codec.prediction_shape[0], **cfg.network).to(
            device
        )
        self.predict = torch.compile(self.model.predict_last) if cfg.compile else self.model.predict_last
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=0.0)
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.generator.set_state(checkpoint["generator"].cpu())
            self.updates = checkpoint["updates"]

    def sample(self, ids, count):
        """Draw transitions uniformly, with a reset prefix that never crosses episode boundaries."""
        selected = ids[torch.randint(len(ids), (count,), device=self.device, generator=self.generator)]
        history = (selected[:, None] + self.offsets).clamp_min(self.starts[selected, None])
        return self.inputs[history], self.targets[selected]

    def update(self):
        self.model.train()
        self.optimizer.param_groups[0]["lr"] = self.cfg.learning_rate + (
            self.cfg.final_learning_rate - self.cfg.learning_rate
        ) * min(self.updates / self.cfg.learning_rate_decay_updates, 1.0)
        x, y = self.sample(self.train_ids, self.cfg.batch_size)
        self.optimizer.zero_grad(set_to_none=True)
        loss = (self.predict(x) - y).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite NeRD training loss at update {self.updates}.")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.gradient_clip_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.updates += 1
        return {"train/normalized_mse": float(loss.detach()), "train/gradient_norm": float(norm)}

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        squared, physical, count = 0.0, 0.0, 0
        for ids in self.validation_ids.split(self.cfg.batch_size):
            history = (ids[:, None] + self.offsets).clamp_min(self.starts[ids, None])
            error = self.predict(self.inputs[history]) - self.targets[ids]
            squared += error.square().sum()
            physical += (error * self.target_scale).square().sum()
            count += error.numel()
        return {"validation/normalized_mse": float(squared / count), "validation/delta_mse": float(physical / count)}

    def checkpoint(self):
        """Return a deployable and resumable snapshot; the composition root publishes it atomically."""
        return {
            "format": "isaaclab-nerd-v1",
            "dynamics": self.dynamics.descriptor,
            "datasets": self.identity,
            "train_episodes": self.train_episodes,
            "network_cfg": self.cfg.network,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "input_mean": self.input_mean,
            "input_scale": self.input_scale,
            "target_mean": self.target_mean,
            "target_scale": self.target_scale,
            "updates": self.updates,
            "generator": self.generator.get_state(),
        }
