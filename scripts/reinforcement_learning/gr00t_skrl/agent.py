# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lifecycle extensions only; PPO updates, GAE, losses and serialization remain native skrl."""

from __future__ import annotations

import hashlib
import json
import random
import resource
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from skrl.agents.torch.ppo import PPO

from .policy import module_digest

if TYPE_CHECKING:
    from .policy import FrozenEncoder


class RunState:
    """Native checkpoint module for cumulative counters, compatibility metadata and RNG."""

    def __init__(self, metadata: dict):
        self.metadata = metadata
        self.environment_steps = 0
        self.updates = 0
        self.optimizer_steps = 0
        self.pending_rng: dict | None = None

    def state_dict(self) -> dict:
        """Capture counters and RNG at a complete native update boundary."""
        return {
            "metadata": self.metadata,
            "environment_steps": self.environment_steps,
            "updates": self.updates,
            "optimizer_steps": self.optimizer_steps,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        }

    def load_state_dict(self, state: dict) -> None:
        """Reject incompatible policy contracts and defer RNG restoration until initialization ends."""
        if state.get("metadata") != self.metadata:
            raise ValueError("Incompatible GR00T skrl checkpoint metadata")
        self.environment_steps = state["environment_steps"]
        self.updates = state["updates"]
        self.optimizer_steps = state["optimizer_steps"]
        self.pending_rng = state["rng"]

    def restore_rng(self) -> None:
        """Restore after native load and all necessary model/trainer initialization."""
        if self.pending_rng is not None:
            random.setstate(self.pending_rng["python"])
            np.random.set_state(self.pending_rng["numpy"])
            torch.set_rng_state(self.pending_rng["torch"].cpu())
            if self.pending_rng["cuda"]:
                torch.cuda.set_rng_state_all([state.cpu() for state in self.pending_rng["cuda"]])
            self.pending_rng = None


def tensor_state_digest(state: dict) -> str:
    """Fingerprint native state tensors for independent verification after load."""
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(str(name).encode())
        if isinstance(value, dict):
            digest.update(tensor_state_digest(value).encode())
        elif isinstance(value, torch.Tensor):
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_checkpoint(path: str, metadata: dict) -> dict:
    """Read trusted local checkpoint on CPU and reject old or incompatible formats before native load."""
    modules = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(modules, dict) or set(modules) != {"policy", "value", "optimizer", "run_state"}:
        raise ValueError("Expected a GR00T native skrl checkpoint; old handwritten PPO is unsupported")
    state = modules["run_state"]
    if state.get("metadata") != metadata:
        raise ValueError("Incompatible GR00T skrl checkpoint metadata")
    steps = [int(item["step"].item()) for item in modules["optimizer"]["state"].values()]
    if not steps or min(steps) != state["optimizer_steps"] or max(steps) != state["optimizer_steps"]:
        raise ValueError("Checkpoint optimizer steps disagree with cumulative run state")
    result = {
        "optimizer_steps": state["optimizer_steps"],
        "environment_steps": state["environment_steps"],
        "updates": state["updates"],
        "policy_state_sha256": tensor_state_digest(modules["policy"]),
        "value_state_sha256": tensor_state_digest(modules["value"]),
        "optimizer_state_sha256": tensor_state_digest(modules["optimizer"]["state"]),
    }
    del modules
    return result


class Gr00tPPO(PPO):
    """Thin native agent hooks for terminal states, backbone residency and boundary checkpoints."""

    def __init__(self, *, encoder: FrozenEncoder | None, run_state: RunState, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.run_state = run_state
        self.checkpoint_modules["run_state"] = run_state
        self.diagnostics: list[dict] = []
        self._resume_offset = run_state.environment_steps
        self._gradient_norm = 0.0
        self._finite_gradients = True
        self.latest_losses: dict[str, float] = {}
        self.optimizer.param_groups[0]["foreach"] = False
        self.optimizer.register_step_pre_hook(self._before_optimizer_step)
        self.optimizer.register_step_post_hook(self._after_optimizer_step)

    def record_transition(
        self, *, next_states: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor, infos: dict, **kwargs
    ) -> None:
        """Let native PPO bootstrap only timeouts using reset-before-terminal critic state."""
        timeout = truncated & ~terminated
        if self.training and timeout.any():
            final_state = infos.get("final_state")
            if final_state is None or final_state.shape != next_states.shape:
                raise ValueError("Timeout transition requires reset-before-terminal final_state")
            next_states = torch.where(timeout, final_state, next_states)
        super().record_transition(
            next_states=next_states, terminated=terminated, truncated=timeout, infos=infos, **kwargs
        )
        self.run_state.environment_steps += 1

    def track_data(self, tag: str, value: float) -> None:
        """Retain native loss diagnostics and fail immediately on non-finite losses."""
        if tag.startswith("Loss /"):
            if not np.isfinite(value):
                raise FloatingPointError(f"Non-finite native {tag}: {value}")
            self.latest_losses[tag] = value
        super().track_data(tag, value)

    def write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
        """Use cumulative environment steps for native TensorBoard after a fresh trainer resumes."""
        super().write_tracking_data(timestep=timestep + self._resume_offset, timesteps=timesteps + self._resume_offset)

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Wrap native post_interaction without replacing its update or optimizer logic."""
        updating = (
            self.training and (self._rollout + 1) % self.cfg.rollouts == 0 and timestep >= self.cfg.learning_starts
        )
        if not updating:
            super().post_interaction(timestep=timestep, timesteps=timesteps)
            return
        before = module_digest(self.policy)
        if self.encoder is not None:
            self.encoder.offload()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            super().post_interaction(timestep=timestep, timesteps=timesteps)
            self.run_state.updates += 1
            after = module_digest(self.policy)
            optimizer_steps = [int(state["step"].item()) for state in self.optimizer.state.values() if "step" in state]
            if not optimizer_steps or max(optimizer_steps) != self.run_state.optimizer_steps:
                raise RuntimeError("Native optimizer and cumulative step counters disagree")
            if not self._finite_gradients or before == after:
                raise FloatingPointError("Non-finite gradients or unchanged original head")
            log_ratios = []
            with torch.no_grad():
                observations = self.memory.get_tensor_by_name("observations")
                actions = self.memory.get_tensor_by_name("actions")
                previous = self.memory.get_tensor_by_name("log_prob")
                for index in range(self.cfg.rollouts):
                    _, output = self.policy.act({"observations": observations[index], "taken_actions": actions[index]})
                    log_ratios.append(output["log_prob"] - previous[index])
            log_ratios = torch.cat(log_ratios)
            ratios = log_ratios.exp()
            if not torch.isfinite(ratios).all():
                raise FloatingPointError("Non-finite post-update PPO ratio")
            diagnostics = {
                "native_losses": self.latest_losses.copy(),
                "ratio_min": float(ratios.min()),
                "ratio_max": float(ratios.max()),
                "approximate_kl": float((ratios - 1 - log_ratios).mean()),
                "clip_fraction": float(((ratios - 1).abs() > self.cfg.ratio_clip).float().mean()),
                "recomputed_after_optimizer_step": len(getattr(self.policy, "recomputed_versions", set())) > 1,
                "environment_steps": self.run_state.environment_steps,
                "updates": self.run_state.updates,
                "optimizer_steps": self.run_state.optimizer_steps,
                "head_changed": before != after,
                "gradient_norm_after_clip": self._gradient_norm,
                "finite_gradients": self._finite_gradients,
                "head_sha256": after,
                "update_peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30
                if self.device.type == "cuda"
                else 0,
            }
            self.diagnostics.append(diagnostics)
            checkpoint_dir = Path(self.experiment_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.save(str(checkpoint_dir / f"agent_{self.run_state.environment_steps}.pt"))
            diagnostics["save_peak_allocated_gib"] = (
                torch.cuda.max_memory_allocated(self.device) / 2**30 if self.device.type == "cuda" else 0
            )
            print(json.dumps(diagnostics), flush=True)
        finally:
            if self.encoder is not None:
                self.encoder.restore()

    def resume(self, path: str) -> dict:
        """Validate on CPU, then use native load on CPU to avoid duplicating GPU checkpoint tensors."""
        summary = validate_checkpoint(path, self.run_state.metadata)
        if self.encoder is not None:
            self.encoder.offload()
        device = self.device
        self.policy.to("cpu")
        self.value.to("cpu")
        self.device = torch.device("cpu")
        try:
            super().load(path)
        finally:
            self.device = device
        self.policy.to(device)
        self.value.to(device)
        for state in self.optimizer.state.values():
            for name in ("exp_avg", "exp_avg_sq"):
                if name in state:
                    state[name] = state[name].to(device)
        for name, actual in (
            ("policy", self.policy.state_dict()),
            ("value", self.value.state_dict()),
            ("optimizer", self.optimizer.state_dict()["state"]),
        ):
            if tensor_state_digest(actual) != summary[f"{name}_state_sha256"]:
                raise RuntimeError(f"Native load failed to restore {name} tensors exactly")
        self._resume_offset = self.run_state.environment_steps
        if self.device.type == "cuda":
            summary["load_peak_allocated_gib"] = torch.cuda.max_memory_allocated(self.device) / 2**30
        if self.encoder is not None:
            self.encoder.restore()
        summary["load_cpu_peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        summary["loaded_head_sha256"] = module_digest(self.policy)
        return summary

    def _before_optimizer_step(self, optimizer: torch.optim.Optimizer, args: tuple, kwargs: dict) -> None:
        norms = []
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    self._finite_gradients &= bool(torch.isfinite(parameter.grad).all())
                    norms.append(torch.linalg.vector_norm(parameter.grad.float()))
        self._gradient_norm = float(torch.linalg.vector_norm(torch.stack(norms)))
        if not self._finite_gradients:
            raise FloatingPointError("Native PPO produced non-finite gradients")

    def _after_optimizer_step(self, optimizer: torch.optim.Optimizer, args: tuple, kwargs: dict) -> None:
        self.run_state.optimizer_steps += 1
