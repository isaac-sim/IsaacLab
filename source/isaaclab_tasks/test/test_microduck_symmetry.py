# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical reflection of MicroDuck's deployed observations and servo actions."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from isaaclab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg

from isaaclab_tasks.contrib.microduck.mdp.symmetry import compute_symmetric_states

_POLICY_TERMS = (
    ("imu", 6),
    ("joint_pos", 14),
    ("joint_vel", 14),
    ("actions", 14),
    ("velocity_commands", 3),
    ("head_pose_commands", 4),
    ("body_pose_commands", 6),
)
_CRITIC_TERMS = (
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 14),
    ("joint_vel", 14),
    ("actions", 14),
    ("velocity_commands", 3),
    ("foot_height", 2),
    ("foot_air_time", 2),
    ("foot_contact", 2),
    ("foot_contact_forces", 6),
    ("head_pose_commands", 4),
    ("body_pose_commands", 6),
)


def _zeros(env, width):
    return torch.zeros(env.num_envs, width, device=env.device)


def _make_env(policy_terms=_POLICY_TERMS, critic_terms=_CRITIC_TERMS):
    """Let the real manager produce term names and dimensions without a physics scene."""
    env = SimpleNamespace(num_envs=1, device="cpu", sim=SimpleNamespace(is_playing=lambda: True))
    groups = {}
    for name, terms in (("policy", policy_terms), ("critic", critic_terms)):
        group = ObservationGroupCfg(enable_corruption=False)
        for term_name, width in terms:
            setattr(group, term_name, ObservationTermCfg(func=_zeros, params={"width": width}))
        groups[name] = group
    env.observation_manager = ObservationManager(groups, env)
    env.unwrapped = env
    return env


def test_servo_reflection_and_optional_inputs():
    """Swap the legs, reverse their servo axes and reflect only head yaw/roll."""
    actions = torch.arange(1, 15, dtype=torch.float64).unsqueeze(0)
    expected = torch.tensor([[-10, -11, -12, -13, -14, 6, 7, -8, -9, -1, -2, -3, -4, -5]], dtype=actions.dtype)
    obs, augmented = compute_symmetric_states(None, actions=actions)
    assert obs is None
    torch.testing.assert_close(augmented, torch.cat((actions, expected)))
    _, twice = compute_symmetric_states(None, actions=augmented[1:])
    torch.testing.assert_close(twice[1:], actions)
    torch.testing.assert_close(actions, torch.arange(1, 15, dtype=actions.dtype).unsqueeze(0))
    assert compute_symmetric_states(None) == (None, None)


def test_reflection_obeys_vector_frames_and_foot_order():
    """Axial vectors, polar vectors, commands and left/right foot features reflect consistently."""
    policy = torch.arange(1, 62, dtype=torch.float32).unsqueeze(0)
    critic = torch.arange(1, 77, dtype=torch.float32).unsqueeze(0)
    obs = TensorDict({"policy": policy, "critic": critic}, batch_size=[1])
    before = obs.clone()
    env = _make_env()
    augmented, actions = compute_symmetric_states(env, obs=obs)
    assert actions is None
    assert augmented.batch_size == torch.Size([2])
    torch.testing.assert_close(augmented[0], obs[0])
    mirrored = augmented[1]
    torch.testing.assert_close(mirrored["policy"][:6], torch.tensor([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]))
    torch.testing.assert_close(mirrored["critic"][:9], torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0]))
    torch.testing.assert_close(
        mirrored["policy"][48:],
        torch.tensor([49.0, -50.0, -51.0, 52.0, 53.0, -54.0, -55.0, 56.0, -57.0, 58.0, -59.0, 60.0, -61.0]),
    )
    torch.testing.assert_close(
        mirrored["critic"][51:66],
        torch.tensor([52.0, -53.0, -54.0, 56.0, 55.0, 58.0, 57.0, 60.0, 59.0, 64.0, -65.0, 66.0, 61.0, -62.0, 63.0]),
    )
    torch.testing.assert_close(
        mirrored["policy"][6:20],
        torch.tensor([-16.0, -17.0, -18.0, -19.0, -20.0, 12.0, 13.0, -14.0, -15.0, -7.0, -8.0, -9.0, -10.0, -11.0]),
    )
    twice, _ = compute_symmetric_states(env, obs=augmented[1:])
    torch.testing.assert_close(twice[1:], obs)
    torch.testing.assert_close(obs, before)


def test_reflection_follows_same_width_term_reordering():
    """Cached reflections must follow manager metadata even when the total width stays fixed."""
    obs = TensorDict(
        {"policy": torch.arange(1, 62).float().unsqueeze(0), "critic": torch.arange(1, 77).float().unsqueeze(0)},
        batch_size=[1],
    )
    compute_symmetric_states(_make_env(), obs=obs)
    policy_terms = (_POLICY_TERMS[4], *_POLICY_TERMS[:4], *_POLICY_TERMS[5:])
    critic_terms = (_CRITIC_TERMS[10], *_CRITIC_TERMS[:10], *_CRITIC_TERMS[11:])
    mirrored, _ = compute_symmetric_states(_make_env(policy_terms, critic_terms), obs=obs)
    torch.testing.assert_close(
        mirrored["policy"][1, :9], torch.tensor([1.0, -2.0, -3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0])
    )
    torch.testing.assert_close(mirrored["critic"][1, :6], torch.tensor([4.0, -5.0, 6.0, 1.0, -2.0, 3.0]))


@pytest.mark.parametrize(
    "invalid_term, width, match",
    [(None, 62, "policy"), (("unknown_imu", 6), 61, "unknown_imu"), (("imu", 7), 61, "imu")],
)
def test_reflection_rejects_unknown_observation_contract(invalid_term, width, match):
    """Unknown semantics and term widths must fail before augmenting unrelated features."""
    terms = _POLICY_TERMS if invalid_term is None else (invalid_term, *_POLICY_TERMS[1:])
    with pytest.raises(ValueError, match=match):
        compute_symmetric_states(_make_env(terms), obs=TensorDict({"policy": torch.zeros(1, width)}, batch_size=[1]))
