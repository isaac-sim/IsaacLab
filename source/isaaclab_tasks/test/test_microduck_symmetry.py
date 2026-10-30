# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical reflection of MicroDuck's deployed observations and servo actions."""

import pytest
import torch
from tensordict import TensorDict

from isaaclab_tasks.contrib.microduck.mdp.symmetry import compute_symmetric_states


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
    augmented, actions = compute_symmetric_states(None, obs=obs)
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
    twice, _ = compute_symmetric_states(None, obs=augmented[1:])
    torch.testing.assert_close(twice[1:], obs)
    torch.testing.assert_close(obs, before)


def test_reflection_rejects_unknown_observation_contract():
    """A changed deploy layout must fail before silently augmenting unrelated features."""
    with pytest.raises(ValueError, match="policy"):
        compute_symmetric_states(None, obs=TensorDict({"policy": torch.zeros(2, 62)}, batch_size=[2]))
