# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heading-frame model inputs and base-frame gravity of a free-root robot."""

import math

import pytest

pytest.importorskip("physicsnemo_robotics")

import newton
import torch
from isaaclab_physicsnemo import HeadingJointStateCodec, RootGravityEncoder
from tensordict import TensorDict

from isaaclab.utils.math import normalize, quat_apply, quat_from_euler_xyz, quat_mul


@pytest.fixture(scope="module")
def codec() -> HeadingJointStateCodec:
    """A free-floating base with one hinged link."""
    robot = newton.ModelBuilder()
    base = robot.add_link(mass=1.0)
    root = robot.add_joint_free(base)
    link = robot.add_link(mass=1.0)
    hinge = robot.add_joint_revolute(base, link)
    robot.add_articulation([root, hinge])
    return HeadingJointStateCodec(robot.finalize(device="cpu"))


def _random_states(count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    position = torch.randn(count, 3, generator=generator)
    orientation = normalize(torch.randn(count, 4, generator=generator))
    hinge = torch.rand(count, 1, generator=generator) * 2.0 - 1.0
    velocity = torch.randn(count, 7, generator=generator)
    return torch.cat((position, orientation, hinge, velocity), dim=-1)


def _yawed(states: torch.Tensor, yaw: float, shift: tuple[float, float]) -> torch.Tensor:
    """Rotate world states about z and shift them horizontally."""
    count = states.shape[0]
    zero = torch.zeros(count)
    turn = quat_from_euler_xyz(zero, zero, torch.full((count,), yaw))
    return torch.cat(
        (
            quat_apply(turn, states[:, :3]) + torch.tensor([*shift, 0.0]),
            quat_mul(turn, states[:, 3:7]),
            states[:, 7:8],
            quat_apply(turn, states[:, 8:11]),
            quat_apply(turn, states[:, 11:14]),
            states[:, 14:],
        ),
        dim=-1,
    )


def test_model_inputs_and_targets_ignore_world_position_and_yaw(codec):
    """A robot that walks the same way facing elsewhere produces the same inputs and targets."""
    current, following = _random_states(64, seed=1), _random_states(64, seed=2)
    moved_current, moved_following = _yawed(current, 1.3, (4.0, -2.0)), _yawed(following, 1.3, (4.0, -2.0))

    torch.testing.assert_close(codec.encode_state(moved_current), codec.encode_state(current), atol=1e-5, rtol=0)
    torch.testing.assert_close(
        codec.state_to_delta(moved_current, moved_following),
        codec.state_to_delta(current, following),
        atol=1e-5,
        rtol=0,
    )


def test_predictions_return_to_world_state(codec):
    """Integrating a heading-frame delta reproduces the world state it was encoded from."""
    current, following = _random_states(64, seed=3), _random_states(64, seed=4)
    following[:, 3:7] = torch.where(following[:, 6:7] < 0.0, -following[:, 3:7], following[:, 3:7])

    recovered = codec.delta_to_state(current, codec.state_to_delta(current, following))

    torch.testing.assert_close(recovered[:, :3], following[:, :3], atol=1e-5, rtol=0)
    orientation_alignment = (recovered[:, 3:7] * following[:, 3:7]).sum(dim=-1).abs()
    torch.testing.assert_close(orientation_alignment, torch.ones(64), atol=1e-5, rtol=0)
    torch.testing.assert_close(recovered[:, 7:], following[:, 7:], atol=1e-4, rtol=0)


def test_gravity_is_expressed_in_the_base_frame(codec):
    """An upright base sees gravity straight down; a base pitched by 90 degrees sees it along its x axis."""
    gravity = RootGravityEncoder(codec, world_gravity=(0.0, 0.0, -9.81))
    states = _random_states(2, seed=5)
    states[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    states[1, 3:7] = torch.tensor([0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)])

    # Training derives gravity from the stored states of each transition window.
    encoded = gravity.encode_window(TensorDict({"history/state": states[:, None]}, batch_size=[2, 1]))

    expected = torch.tensor([[[0.0, 0.0, -1.0]], [[1.0, 0.0, 0.0]]])
    torch.testing.assert_close(encoded, expected, atol=1e-6, rtol=0)
