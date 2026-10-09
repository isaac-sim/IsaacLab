# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Term-driven reflection across the sagittal (XZ) plane for locomotion tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from isaaclab.managers import ActionTermCfg, ObservationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "MirrorAugmentation",
    "compute_mirrored_states",
    "mirror_identity",
    "mirror_vec3",
    "mirror_quat",
    "mirror_joints",
]


def mirror_identity(data: torch.Tensor) -> torch.Tensor:
    """Leave an explicitly reflection-invariant term unchanged."""
    return data.clone()


def mirror_vec3(data: torch.Tensor, axial: bool = False) -> torch.Tensor:
    """Reflect vectors with trailing shape ``(3,)`` across the XZ plane.

    Args:
        data: Vectors with arbitrary leading batch or history dimensions.
        axial: Whether vectors represent angular velocity, torque, or another axial
            quantity. Polar vectors negate Y; axial vectors negate X and Z.

    Returns:
        Reflected vectors, with the same units and shape as the input.
    """
    if data.shape[-1] != 3:
        raise ValueError(f"Expected three vector components, got {data.shape}.")
    signs = (-1, 1, -1) if axial else (1, -1, 1)
    return data * data.new_tensor(signs)


def mirror_quat(data: torch.Tensor) -> torch.Tensor:
    """Reflect XYZW orientations across the XZ plane, preserving quaternion sign.

    The corresponding rotation matrix transforms as ``S @ R @ S``, where
    ``S = diag(1, -1, 1)``. Arbitrary leading batch or history dimensions are supported.
    """
    if data.shape[-1] != 4:
        raise ValueError(f"Expected an XYZW quaternion, got {data.shape}.")
    return data * data.new_tensor((-1, 1, -1, 1))


def mirror_joints(data: torch.Tensor, permutation: Sequence[int], signs: Sequence[float] | None = None) -> torch.Tensor:
    """Swap and sign-correct joint channels in their configured order.

    Args:
        data: Joint values with joints on the last axis. Units are preserved.
        permutation: Source index for each output joint. Must be an involution:
            applying the permutation twice restores the original joint order.
        signs: Per-output signs, each +1 or -1. Paired joints must have equal signs.
            Defaults to keeping every sign.

    Returns:
        Mirrored joint values with the same shape as the input.
    """
    count = data.shape[-1]
    if len(permutation) != count or sorted(permutation) != list(range(count)):
        raise ValueError("Joint permutation must contain each input index exactly once.")
    if any(permutation[permutation[i]] != i for i in range(count)):
        raise ValueError("Joint permutation must restore the input when applied twice.")
    if signs is not None:
        if len(signs) != count or any(s not in (-1, 1) for s in signs):
            raise ValueError("Joint signs must contain one +1 or -1 per joint.")
        if any(signs[i] != signs[permutation[i]] for i in range(count)):
            raise ValueError("Paired joints must have equal mirror signs.")
    result = data[..., list(permutation)]
    return result if signs is None else result * data.new_tensor(signs)


def compute_mirrored_states(
    env: ManagerBasedRLEnv, obs: TensorDict | None = None, actions: torch.Tensor | None = None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """RSL-RL callback returning originals followed by their mirrored samples.

    Layout metadata is cached per environment. Construct a new environment after
    changing its observation or action configuration.
    """
    env = env.unwrapped
    augmentation = getattr(env, "_mirror_augmentation", None)
    if augmentation is None:
        augmentation = MirrorAugmentation(env)
        env._mirror_augmentation = augmentation
    return augmentation(obs, actions)


class MirrorAugmentation:
    """Reflect manager terms without fixed group names, offsets, or history lengths.

    Every supplied term must declare a callable ``mirror``. Missing rules raise an
    error instead of silently creating inconsistent augmented samples. Both
    concatenated observation groups and nested TensorDict groups are supported.
    """

    def __init__(self, env: ManagerBasedRLEnv):
        """Read layouts from initialized observation and action managers."""
        self._observation_manager = env.observation_manager
        self._action_terms = [
            (name, env.action_manager.get_term(name).cfg, size)
            for name, size in zip(env.action_manager.active_terms, env.action_manager.action_term_dim)
        ]

    @torch.no_grad()
    def __call__(
        self, obs: TensorDict | None = None, actions: torch.Tensor | None = None
    ) -> tuple[TensorDict | None, torch.Tensor | None]:
        """Append mirrored samples along the first batch axis, preserving inputs."""
        obs_aug = None if obs is None else torch.cat((obs, self.mirror_observations(obs)), dim=0)
        actions_aug = None if actions is None else torch.cat((actions, self.mirror_actions(actions)), dim=0)
        return obs_aug, actions_aug

    def mirror_observations(self, obs: TensorDict) -> TensorDict:
        """Reflect each supplied group using its active manager terms and shapes."""
        manager = self._observation_manager
        result = obs.clone()
        for group_name in obs.keys():
            if group_name not in manager.active_terms:
                raise ValueError(f"Unknown observation group {group_name!r}.")
            group_cfg = _get_cfg(manager.cfg, group_name)
            names = manager.active_terms[group_name]
            shapes = manager.group_obs_term_dim[group_name]
            concatenate = manager.group_obs_concatenate[group_name]
            if concatenate:
                rank = len(shapes[0])
                axis = group_cfg.concatenate_dim % rank
                dim = obs.batch_dims + axis
                value = obs[group_name]
                sizes = [shape[axis] for shape in shapes]
                if value.shape[dim] != sum(sizes):
                    raise ValueError(f"Observation group {group_name!r} does not match its manager layout.")
                terms = torch.split(value, sizes, dim=dim)
            else:
                if set(obs[group_name].keys()) != set(names):
                    raise ValueError(f"Observation group {group_name!r} must contain all its active terms.")
                terms = [obs[group_name][name] for name in names]
            mirrored = []
            for name, shape, value in zip(names, shapes, terms):
                if tuple(value.shape[obs.batch_dims :]) != tuple(shape):
                    raise ValueError(f"Observation {group_name}/{name} has shape inconsistent with its manager.")
                cfg = _get_cfg(group_cfg, name)
                original_shape = value.shape
                if cfg.history_length > 0 and cfg.flatten_history_dim:
                    value = value.reshape(*obs.batch_size, cfg.history_length, -1)
                value = _mirror(value, cfg, f"{group_name}/{name}").reshape(original_shape)
                if concatenate:
                    mirrored.append(value)
                else:
                    result[group_name][name] = value
            if concatenate:
                result[group_name] = torch.cat(mirrored, dim=dim)
        return result

    def mirror_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Reflect raw actions term by term, including multiple action terms."""
        sizes = [size for _, _, size in self._action_terms]
        if actions.shape[-1] != sum(sizes):
            raise ValueError("Action shape does not match the action manager layout.")
        values = torch.split(actions, sizes, dim=-1)
        return torch.cat(
            [_mirror(value, cfg, name) for (name, cfg, _), value in zip(self._action_terms, values)], dim=-1
        )


def _get_cfg(cfg: Any, name: str) -> Any:
    return cfg[name] if isinstance(cfg, dict) else getattr(cfg, name)


def _mirror(data: torch.Tensor, cfg: ObservationTermCfg | ActionTermCfg, name: str) -> torch.Tensor:
    function = getattr(cfg, "mirror", None)
    if not callable(function):
        raise ValueError(f"Term {name!r} requires a callable mirror; use mirror_identity for invariant terms.")
    result = function(data.clone(), **getattr(cfg, "mirror_params", {}))
    if not isinstance(result, torch.Tensor) or result.shape != data.shape:
        raise ValueError(f"Mirror for {name!r} must preserve the tensor shape {tuple(data.shape)}.")
    if result.dtype != data.dtype or result.device != data.device:
        raise ValueError(f"Mirror for {name!r} must preserve dtype and device.")
    return result
