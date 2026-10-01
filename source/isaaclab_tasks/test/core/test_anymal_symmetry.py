# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ANYmal symmetry augmentation."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from isaaclab_tasks.core.velocity.mdp.symmetry import anymal

# Native joint orders of ANYmal-D on the two physics backends.
PHYSX_ORDER = [
    "LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA",
    "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
    "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE",
]  # fmt: skip
NEWTON_ORDER = [
    "LF_HAA", "LF_HFE", "LF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
]  # fmt: skip


# Reference: the index-assignment formulation the permutation tables must reproduce.
_LEFT = [0, 4, 8, 1, 5, 9]
_RIGHT = [2, 6, 10, 3, 7, 11]
_FRONT = [0, 4, 8, 2, 6, 10]
_HIND = [1, 5, 9, 3, 7, 11]


def _reference_joints(joint_data: torch.Tensor, kind: str) -> torch.Tensor:
    first, second = (_LEFT, _RIGHT) if kind == "left_right" else (_FRONT, _HIND)
    switched = torch.zeros_like(joint_data)
    switched[..., first] = joint_data[..., second]
    switched[..., second] = joint_data[..., first]
    if kind == "left_right":
        switched[..., [0, 1, 2, 3]] *= -1.0
    else:
        switched[..., 4:] *= -1
    return switched


def _reference_obs(obs: torch.Tensor, kind: str, height_scan: bool) -> torch.Tensor:
    signs = {
        "left_right": ([1, -1, 1], [-1, 1, -1], [1, -1, 1], [1, -1, -1]),
        "front_back": ([-1, 1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1]),
    }[kind]
    obs = obs.clone()
    for block, sign in enumerate(signs):
        obs[:, 3 * block : 3 * block + 3] = obs[:, 3 * block : 3 * block + 3] * torch.tensor(sign)
    for start in (12, 24, 36):
        obs[:, start : start + 12] = _reference_joints(obs[:, start : start + 12], kind)
    if height_scan:
        flip_dim = 1 if kind == "left_right" else 2
        obs[:, 48:235] = obs[:, 48:235].view(-1, 11, 17).flip(dims=[flip_dim]).view(-1, 11 * 17)
    return obs


def _reference_augment(obs: torch.Tensor, actions: torch.Tensor, height_scan: bool):
    left_right = _reference_obs(obs, "left_right", height_scan)
    obs_aug = torch.cat(
        (
            obs,
            left_right,
            _reference_obs(obs, "front_back", height_scan),
            _reference_obs(left_right, "front_back", height_scan),
        )
    )
    actions_left_right = _reference_joints(actions, "left_right")
    actions_aug = torch.cat(
        (
            actions,
            actions_left_right,
            _reference_joints(actions, "front_back"),
            _reference_joints(actions_left_right, "front_back"),
        )
    )
    return obs_aug, actions_aug


@pytest.mark.parametrize("height_scan", [False, True])
def test_compute_symmetric_states_matches_reference(height_scan: bool):
    """Verify the cached permutation and sign tables reproduce the index-assignment formulation in the PhysX order."""
    num_envs = 5
    num_obs = 235 if height_scan else 48
    obs = torch.randn(num_envs, num_obs)
    actions = torch.randn(num_envs, 12)
    active_terms = ["base_lin_vel", "height_scan"] if height_scan else ["base_lin_vel"]
    unwrapped = SimpleNamespace(
        scene={"robot": SimpleNamespace(joint_names=PHYSX_ORDER)},
        observation_manager=SimpleNamespace(active_terms={"policy": active_terms}),
    )
    env = SimpleNamespace(unwrapped=unwrapped)

    obs_aug, actions_aug = anymal.compute_symmetric_states(
        env, obs=TensorDict({"policy": obs}, batch_size=[num_envs]), actions=actions
    )
    expected_obs, expected_actions = _reference_augment(obs, actions, height_scan)

    assert torch.equal(obs_aug["policy"], expected_obs)
    assert torch.equal(actions_aug, expected_actions)


def test_compute_symmetric_states_follows_joint_names():
    """Verify the Newton joint order is mirrored like the PhysX order, with the joint columns reordered."""
    joint_cols = [PHYSX_ORDER.index(name) for name in NEWTON_ORDER]
    obs_cols = list(range(12)) + [start + joint for start in (12, 24, 36) for joint in joint_cols]
    obs = torch.randn(5, 48)
    actions = torch.randn(5, 12)

    outputs = []
    for joint_names, obs_idx, action_idx in (
        (PHYSX_ORDER, list(range(48)), list(range(12))),
        (NEWTON_ORDER, obs_cols, joint_cols),
    ):
        env = SimpleNamespace(
            scene={"robot": SimpleNamespace(joint_names=joint_names)},
            observation_manager=SimpleNamespace(active_terms={"policy": []}),
        )
        env.unwrapped = env
        policy = TensorDict({"policy": obs[:, obs_idx]}, batch_size=[5])
        outputs.append(anymal.compute_symmetric_states(env, policy, actions[:, action_idx]))
    (physx_obs, physx_actions), (newton_obs, newton_actions) = outputs

    assert torch.equal(newton_obs["policy"], physx_obs["policy"][:, obs_cols])
    assert torch.equal(newton_actions, physx_actions[:, joint_cols])


def test_compute_symmetric_states_without_inputs():
    """Return ``(None, None)`` without reading the scene when there is nothing to augment."""
    assert anymal.compute_symmetric_states(SimpleNamespace()) == (None, None)
