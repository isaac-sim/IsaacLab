# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gaze geometry and gaze-supervised PPO, without a simulator."""

import pytest
import torch
import warp as wp
from fovi.sensing.projection import CameraModel
from isaaclab_fovi.mdp import angles_to_direction, clamp_gaze, direction_to_angles
from isaaclab_fovi.mdp.gaze import gaze_step_kernel, teacher_kernel, teacher_labels_kernel
from isaaclab_fovi.rsl_rl.gaze_supervised_ppo import GazeSupervisedPPO, gaze_loss
from isaaclab_fovi.sensors.gaze_kernels import camera_params, torch_stream
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from isaaclab.utils.math import quat_apply, quat_apply_inverse

NUM_ENVS = 64
STEPS = 8


def test_gaze_angles_round_trip_and_follow_the_camera_convention():
    angles = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [-20.0, 35.0]])
    directions = angles_to_direction(angles)
    torch.testing.assert_close(direction_to_angles(directions), angles, atol=1e-4, rtol=0)
    # OpenCV camera frame: z forward, y down, x right; positive tilt looks up, positive pan looks left.
    torch.testing.assert_close(directions[0], torch.tensor([0.0, 0.0, 1.0]))
    assert directions[1, 1] < 0 and directions[2, 0] < 0


def test_clamp_gaze_is_radial():
    gaze = torch.tensor([[0.3, 0.4], [3.0, 4.0]])
    torch.testing.assert_close(clamp_gaze(gaze, 1.0), torch.tensor([[0.3, 0.4], [0.6, 0.8]]))


def test_teacher_kernel_matches_the_torch_teacher():
    camera = CameraModel("pinhole", (480, 640), (400.0, 400.0, 319.5, 239.5), distortion=(-0.1, 0.02, 0.0, 0.0))
    field_radius, step, limit = 30.0, 0.1, 1.0
    generator = torch.Generator().manual_seed(3)
    count = 4096
    mount = torch.nn.functional.normalize(torch.randn(4, generator=generator), dim=0)
    positions = torch.randn(count, 3, generator=generator)
    # Points mostly in front of the camera, spread across and beyond its field of view.
    lateral = torch.randn(count, 2, generator=generator) * 0.6
    local = torch.cat((lateral, torch.rand(count, 1, generator=generator) + 0.2), -1)
    points = positions + quat_apply(mount.expand(count, -1), local)
    gaze = (torch.rand(count, 2, generator=generator) * 2 - 1) * 0.7

    # The torch teacher the kernel replaces.
    direction = torch.nn.functional.normalize(quat_apply_inverse(mount.expand(count, -1), points - positions), dim=-1)
    desired = direction_to_angles(direction) / field_radius
    expected_valid = camera.project(direction)[1] & (desired.norm(dim=-1) <= limit)
    expected_action = torch.where(expected_valid[:, None], ((desired - gaze) / step).clamp(-1, 1), 0.0)
    looking = angles_to_direction(gaze * field_radius)
    expected_error = torch.atan2(torch.cross(looking, direction, dim=-1).norm(dim=-1), (looking * direction).sum(-1))

    action = torch.empty(count, 2)
    valid = torch.empty(count, dtype=torch.bool)
    error = torch.empty(count)
    with torch_stream("cpu"):
        wp.launch(
            teacher_kernel,
            dim=count,
            inputs=[
                wp.from_torch(points.contiguous(), dtype=wp.vec3f),
                wp.from_torch(positions.contiguous(), dtype=wp.vec3f),
                wp.quatf(*mount.tolist()),
                wp.from_torch(gaze.contiguous(), dtype=wp.vec2f),
                field_radius,
                step,
                limit,
                camera_params(camera),
            ],
            outputs=[wp.from_torch(action, dtype=wp.vec2f), wp.from_torch(valid), wp.from_torch(error)],
        )
    agree = valid == expected_valid
    assert agree.float().mean() > 0.999 and expected_valid.any() and (~expected_valid).any()
    torch.testing.assert_close(action[agree], expected_action[agree], atol=1e-4, rtol=0)
    torch.testing.assert_close(error, expected_error.rad2deg(), atol=1e-3, rtol=0)


def test_gaze_step_kernel_matches_the_torch_update():
    generator = torch.Generator().manual_seed(4)
    count, step, limit, dt, radius = 2048, 0.1, 1.0, 1 / 60, 23.6
    actions = torch.randn(count, 2, generator=generator) * 1.5
    gaze = (torch.rand(count, 2, generator=generator) * 2 - 1) * 0.75
    expected = clamp_gaze(gaze + actions.clamp(-1, 1) * step, limit)
    expected_velocity = (expected - gaze) / dt

    velocity, angles = torch.empty_like(gaze), torch.empty_like(gaze)
    state = torch.empty(count, 4)
    with torch_stream("cpu"):
        wp.launch(
            gaze_step_kernel,
            dim=count,
            inputs=[
                *(wp.from_torch(buffer, dtype=wp.vec2f) for buffer in (actions, gaze, velocity, angles)),
                wp.from_torch(state, dtype=wp.vec4f),
                step,
                limit,
                dt,
                radius,
            ],
        )
    torch.testing.assert_close(gaze, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(velocity, expected_velocity, atol=1e-3, rtol=1e-5)
    torch.testing.assert_close(angles, expected * radius, atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(state, torch.cat((gaze, velocity), dim=-1))


def test_teacher_labels_kernel_matches_the_torch_labels():
    camera = CameraModel("pinhole", (480, 640), (400.0, 400.0, 319.5, 239.5))
    field_radius, step, limit = 30.0, 0.1, 1.0
    generator = torch.Generator().manual_seed(5)
    count = 4096
    mount = torch.nn.functional.normalize(torch.randn(4, generator=generator), dim=0)
    positions = torch.randn(count, 3, generator=generator)
    lateral = torch.randn(count, 2, generator=generator) * 0.6
    local = torch.cat((lateral, torch.rand(count, 1, generator=generator) + 0.2), -1)
    targets = positions + quat_apply(mount.expand(count, -1), local)
    orientations = torch.nn.functional.normalize(torch.randn(count, 4, generator=generator), dim=-1)
    offset = torch.tensor([0.0, 0.0, 0.3])
    gaze = (torch.rand(count, 2, generator=generator) * 2 - 1) * 0.7

    # The torch path the kernel replaces: entity_point, then the teacher, then the labels and logs.
    points = targets + quat_apply(orientations, offset.expand(count, -1))
    direction = torch.nn.functional.normalize(quat_apply_inverse(mount.expand(count, -1), points - positions), dim=-1)
    desired = direction_to_angles(direction) / field_radius
    valid = camera.project(direction)[1] & (desired.norm(dim=-1) <= limit)
    action = torch.where(valid[:, None], ((desired - gaze) / step).clamp(-1, 1), 0.0)
    looking = angles_to_direction(gaze * field_radius)
    error = torch.atan2(torch.cross(looking, direction, dim=-1).norm(dim=-1), (looking * direction).sum(-1)).rad2deg()
    expected = torch.cat((action, valid[:, None].float()), dim=-1)

    labels = torch.empty(count, 3)
    sums = torch.zeros(2)
    with torch_stream("cpu"):
        wp.launch(
            teacher_labels_kernel,
            dim=count,
            inputs=[
                wp.from_torch(targets, dtype=wp.vec3f),
                wp.from_torch(orientations, dtype=wp.quatf),
                wp.vec3f(*offset.tolist()),
                1,
                wp.from_torch(positions, dtype=wp.vec3f),
                wp.quatf(*mount.tolist()),
                wp.from_torch(gaze, dtype=wp.vec2f),
                field_radius,
                step,
                limit,
                camera_params(camera),
            ],
            outputs=[wp.from_torch(labels), wp.from_torch(sums)],
        )
    agree = (labels[:, 2] > 0.5) == valid
    assert agree.float().mean() > 0.999 and valid.any() and (~valid).any()
    torch.testing.assert_close(labels[agree], expected[agree], atol=1e-4, rtol=0)
    both = agree & valid
    assert float(sums[1]) == pytest.approx(float((labels[:, 2] > 0.5).sum()))
    assert float(sums[0]) == pytest.approx(float(torch.where(labels[:, 2] > 0.5, error, 0.0).sum()), rel=1e-4)
    assert both.sum() > 0


def test_gaze_loss_ignores_invalid_labels():
    mean = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True)
    labels = torch.tensor([[0.3, 0.4, 1.0], [float("nan"), float("nan"), 0.0]])
    loss = gaze_loss(mean, labels)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(0.5)
    assert torch.isfinite(mean.grad).all() and not mean.grad[1].any()


def _algorithm() -> tuple[GazeSupervisedPPO, TensorDict]:
    torch.manual_seed(0)
    labels = torch.rand(NUM_ENVS, 3) * 2 - 1
    labels[:, 2] = (torch.arange(NUM_ENVS) % 4 != 0).float()
    labels[labels[:, 2] == 0, :2] = float("nan")
    obs = TensorDict({"policy": torch.randn(NUM_ENVS, 6), "oracle": labels}, batch_size=[NUM_ENVS])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    distribution = {"class_name": "GaussianDistribution", "init_std": 1.0}
    actor = MLPModel(obs, groups, "actor", 3, hidden_dims=[32], distribution_cfg=distribution)
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[32])
    storage = RolloutStorage("rl", NUM_ENVS, STEPS, obs, [3], "cpu")
    settings = dict(num_learning_epochs=4, num_mini_batches=4, learning_rate=3e-3, schedule="fixed", device="cpu")
    return GazeSupervisedPPO(actor, critic, storage, **settings), obs


def test_update_pulls_the_gaze_toward_the_teacher():
    algorithm, obs = _algorithm()
    valid = obs["oracle"][:, 2] > 0.5

    def distance() -> float:
        with torch.no_grad():
            mean = algorithm.actor(obs)[:, -2:]
        return float((mean - obs["oracle"][:, :2])[valid].norm(dim=-1).mean())

    before = distance()
    for _ in range(40):
        for _ in range(STEPS):
            with torch.inference_mode():
                algorithm.act(obs)
                algorithm.process_env_step(obs, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), {})
        algorithm.compute_returns(obs)
        losses = algorithm.update()
    assert losses["oracle_valid"] == pytest.approx(0.75)
    assert all(torch.isfinite(p).all() for p in algorithm.actor.parameters())
    assert distance() < 0.7 * before
