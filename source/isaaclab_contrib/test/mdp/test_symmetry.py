# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reflection contracts using layouts produced by the observation and action managers."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from isaaclab.managers import ActionManager, ActionTerm, ObservationGroupCfg, ObservationManager

from isaaclab_contrib.mdp.symmetry import (
    MirrorActionTermCfg,
    MirrorAugmentation,
    MirrorObservationTermCfg,
    compute_mirrored_states,
    mirror_joints,
    mirror_quat,
    mirror_vec3,
)


@pytest.fixture
def env():
    environment = SimpleNamespace(
        num_envs=1,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        scene={"robot": object()},
        vectors=torch.tensor([[1.0, 2.0, 3.0]]),
        orientation=torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
    )
    environment.unwrapped = environment
    actor = ObservationGroupCfg(enable_corruption=False)
    actor.velocity = MirrorObservationTermCfg(
        func=lambda env: env.vectors.clone(), mirror=mirror_vec3, history_length=2, flatten_history_dim=True
    )
    actor.orientation = MirrorObservationTermCfg(func=lambda env: env.orientation.clone(), mirror=mirror_quat)
    critic = ObservationGroupCfg(enable_corruption=False, concatenate_terms=False)
    critic.angular_velocity = MirrorObservationTermCfg(
        func=lambda env: env.vectors.clone(), mirror=mirror_vec3, mirror_params={"axial": True}
    )
    environment.observation_manager = ObservationManager({"actor": actor, "critic": critic}, environment)
    action = MirrorActionTermCfg(
        class_type=_TwoChannelAction,
        asset_name="robot",
        mirror=mirror_joints,
        mirror_params={"permutation": [1, 0], "signs": [-1, -1]},
    )
    environment.action_manager = ActionManager(SimpleNamespace(joints=action), environment)
    return environment


def test_manager_history_and_action_reflection(env):
    """Preserve history order while reflecting polar/axial vectors, orientations, and raw actions."""
    env.observation_manager.compute(update_history=True)
    env.vectors += 10
    obs = TensorDict(env.observation_manager.compute(update_history=True), batch_size=[1])
    original = obs.clone()
    actions = torch.tensor([[2.0, 3.0]])
    augmented, augmented_actions = compute_mirrored_states(env, obs, actions)
    torch.testing.assert_close(
        augmented["actor"][1], torch.tensor([1.0, -2.0, 3.0, 11.0, -12.0, 13.0, -0.5, 0.5, -0.5, 0.5])
    )
    torch.testing.assert_close(augmented["critic", "angular_velocity"][1], torch.tensor([-11.0, 12.0, -13.0]))
    torch.testing.assert_close(augmented_actions, torch.tensor([[2.0, 3.0], [-3.0, -2.0]]))
    reflection = MirrorAugmentation(env)
    restored = reflection.mirror_observations(reflection.mirror_observations(obs))
    for key in obs.keys(include_nested=True, leaves_only=True):
        torch.testing.assert_close(obs[key], original[key])
        torch.testing.assert_close(augmented[key][:1], original[key])
        torch.testing.assert_close(restored[key], original[key])
    torch.testing.assert_close(actions, torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(reflection.mirror_actions(reflection.mirror_actions(actions)), actions)
    assert compute_mirrored_states(env, actions=actions)[0] is None
    assert compute_mirrored_states(env, obs=obs)[1] is None


def test_missing_reflection_rule_and_action_layout_are_rejected(env):
    """An unconfigured reflection must fail rather than silently change augmentation semantics."""
    obs = TensorDict(env.observation_manager.compute(), batch_size=[1])
    reflection = MirrorAugmentation(env)
    env.observation_manager.cfg["actor"].velocity.mirror = None
    with pytest.raises(ValueError, match="requires.*mirror"):
        reflection.mirror_observations(obs)
    with pytest.raises(ValueError, match="Action shape"):
        reflection.mirror_actions(torch.zeros(1, 3))


class _TwoChannelAction(ActionTerm):
    """Minimal concrete action for the manager's public term-layout interface."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._actions = torch.zeros(env.num_envs, 2, device=env.device)

    @property
    def action_dim(self):
        return 2

    @property
    def raw_actions(self):
        return self._actions

    @property
    def processed_actions(self):
        return self._actions

    def process_actions(self, actions):
        self._actions.copy_(actions)

    def apply_actions(self):
        pass
