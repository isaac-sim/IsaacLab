# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import io
from typing import Any

from .catalog import LoadedCheckpoint


class RslRlPolicyAdapter:
    """One RSL-RL inference policy bound to a single comparison world."""

    def __init__(self, policy: Any, reset_target: Any):
        self.policy = policy
        self.reset_target = reset_target

    def __call__(self, observation: Any):
        return self.policy(observation)

    def reset(self) -> None:
        import torch

        reset = getattr(self.reset_target, "reset", None)
        if reset is not None:
            try:
                device = next(self.reset_target.parameters()).device
            except (AttributeError, StopIteration):
                device = None
            try:
                reset(torch.ones(1, dtype=torch.bool, device=device))
            except TypeError:
                reset()


class LegacyRslRlInferencePolicy:
    """Evaluate a legacy combined checkpoint with its original actor math."""

    def __init__(self, checkpoint: LoadedCheckpoint, cfg: dict[str, Any], device: str):
        state = checkpoint.payload["model_state_dict"]
        layer_indices = sorted(
            {int(name.split(".")[1]) for name in state if name.startswith("actor.") and name.endswith(".weight")}
        )
        if not layer_indices:
            raise ValueError("Legacy RSL-RL checkpoint contains no actor linear layers")
        self.weights = []
        self.biases = []
        previous_output = None
        for index in layer_indices:
            weight = state.get(f"actor.{index}.weight")
            bias = state.get(f"actor.{index}.bias")
            if weight is None or bias is None or weight.ndim != 2 or bias.shape != weight.shape[:1]:
                raise ValueError(f"Invalid legacy actor layer {index}")
            if previous_output is not None and weight.shape[1] != previous_output:
                raise ValueError(f"Legacy actor layer {index} has an incompatible input dimension")
            self.weights.append(weight.to(device))
            self.biases.append(bias.to(device))
            previous_output = weight.shape[0]

        policy_cfg = cfg.get("policy", {})
        self.activation = policy_cfg.get("activation", "elu")
        if self.activation != "elu":
            raise ValueError(f"Unsupported legacy actor activation: {self.activation}")
        obs_groups = cfg.get("obs_groups", {})
        self.obs_group_names = tuple(obs_groups.get("policy", obs_groups.get("actor", ("policy",))))
        self.mean = state.get("actor_obs_normalizer._mean")
        self.std = state.get("actor_obs_normalizer._std")
        if (self.mean is None) != (self.std is None):
            raise ValueError("Legacy actor observation normalizer state is incomplete")
        if self.mean is not None:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)

    def __call__(self, observation: Any):
        import torch
        import torch.nn.functional as functional

        if isinstance(observation, torch.Tensor):
            actor_obs = observation
        else:
            try:
                parts = [observation[name] for name in self.obs_group_names]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Actor observation is missing one of {self.obs_group_names}") from exc
            actor_obs = torch.cat(parts, dim=-1)
        if self.mean is not None:
            actor_obs = (actor_obs - self.mean) / (self.std + 1.0e-2)
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            actor_obs = functional.linear(actor_obs, weight, bias)
            if layer + 1 < len(self.weights):
                actor_obs = functional.elu(actor_obs)
        return actor_obs

    def reset(self) -> None:
        return None


class RslRlPolicyFactory:
    """Construct and strictly load either RSL-RL runner supported by play."""

    def __init__(self, env: Any, agent_cfg: Any, device: str):
        self.env = env
        self.agent_cfg = agent_cfg
        self.device = device

    def create(self, checkpoint: LoadedCheckpoint) -> RslRlPolicyAdapter:
        cfg = self.agent_cfg.to_dict() if hasattr(self.agent_cfg, "to_dict") else dict(self.agent_cfg)
        state = checkpoint.payload.get("model_state_dict")
        if isinstance(state, dict) and any(name.startswith("actor.") for name in state):
            policy = LegacyRslRlInferencePolicy(checkpoint, cfg, self.env.unwrapped.device)
            self._validate_legacy_policy_io(policy)
            return RslRlPolicyAdapter(policy, policy)

        from rsl_rl.runners import DistillationRunner, OnPolicyRunner

        class_name = getattr(self.agent_cfg, "class_name", cfg.get("class_name"))
        with contextlib.redirect_stdout(io.StringIO()):
            if class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(self.env, cfg, log_dir=None, device=self.device)
            elif class_name == "DistillationRunner":
                runner = DistillationRunner(self.env, cfg, log_dir=None, device=self.device)
            else:
                raise ValueError(f"Unsupported RSL-RL runner class: {class_name}")

        models = self._runner_models(runner, checkpoint)
        checkpoint_shapes = checkpoint.parameter_shapes
        grouped = len(models) > 1
        expected = {}
        for group_name, model in models.items():
            for name, value in model.state_dict().items():
                key = f"{group_name}.{name}" if grouped else name
                expected[key] = tuple(value.shape)
        mismatched = [
            f"{name}: checkpoint {checkpoint_shapes.get(name)}, task {shape}"
            for name, shape in expected.items()
            if checkpoint_shapes.get(name) != shape
        ]
        unexpected = sorted(set(checkpoint_shapes) - set(expected))
        if mismatched or unexpected:
            details = "; ".join([*mismatched[:5], *(f"unexpected: {name}" for name in unexpected[:5])])
            raise ValueError(f"Checkpoint model shape is incompatible with this task/runner: {details}")

        runner.load(str(checkpoint.path))
        policy = runner.get_inference_policy(device=self.env.unwrapped.device)
        reset_target = policy if hasattr(policy, "reset") else next(iter(models.values()))
        return RslRlPolicyAdapter(policy, reset_target)

    def _validate_legacy_policy_io(self, policy: LegacyRslRlInferencePolicy) -> None:
        import torch

        try:
            with torch.inference_mode():
                actions = policy(self.env.get_observations())
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Legacy checkpoint actor input is incompatible with this task's policy observation: {exc}"
            ) from exc
        expected_actions = int(self.env.num_actions)
        if not isinstance(actions, torch.Tensor) or actions.ndim != 2 or actions.shape[-1] != expected_actions:
            shape = tuple(actions.shape) if isinstance(actions, torch.Tensor) else type(actions).__name__
            raise ValueError(
                "Legacy checkpoint actor output is incompatible with this task's action space: "
                f"checkpoint {shape}, task (*, {expected_actions})"
            )

    @classmethod
    def _runner_models(cls, runner: Any, checkpoint: LoadedCheckpoint) -> dict[str, Any]:
        algorithm = runner.alg
        groups = {
            name
            for name in ("actor", "critic", "student", "teacher")
            if isinstance(checkpoint.payload.get(f"{name}_state_dict"), dict)
        }
        if groups == {"actor", "critic"}:
            if hasattr(algorithm, "_raw_actor") and hasattr(algorithm, "_raw_critic"):
                return {"actor": algorithm._raw_actor, "critic": algorithm._raw_critic}
            if hasattr(algorithm, "_raw_teacher"):
                return {"actor": algorithm._raw_teacher}
        if groups <= {"student", "teacher"} and groups:
            models = {}
            for name in groups:
                model = getattr(algorithm, f"_raw_{name}", None)
                if model is None:
                    raise ValueError(f"Runner has no {name} model required by this checkpoint")
                models[name] = model
            return models
        return {"model": cls._runner_model(runner)}

    @staticmethod
    def _runner_model(runner: Any) -> Any:
        algorithm = runner.alg
        for name in ("policy", "actor_critic", "student"):
            model = getattr(algorithm, name, None)
            if model is not None:
                return model
        raise ValueError("Unable to locate the RSL-RL policy model for shape validation")
