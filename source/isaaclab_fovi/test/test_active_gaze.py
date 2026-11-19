# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The gaze action, its observation, without a simulator."""

import math
from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_fovi.mdp.gaze import GazeAction, gaze_state, gaze_step_kernel
from isaaclab_fovi.mdp.gaze_cfg import GazeActionCfg
from isaaclab_fovi.sensors.gaze_kernels import torch_stream

from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.kitless


def clamp_gaze(gaze: torch.Tensor, limit: float) -> torch.Tensor:
    """Gazes scaled back onto the circle of radius ``limit`` when they leave it: the reference for the step kernel."""
    return gaze * (limit / gaze.norm(dim=-1, keepdim=True).clamp_min(limit))


def test_gaze_step_kernel_matches_the_torch_update():
    generator = torch.Generator().manual_seed(4)
    count, step, limit, dt, radius = 2048, 0.1, 1.0, 1 / 60, 23.6
    actions = torch.randn(count, 2, generator=generator) * 1.5
    gaze = (torch.rand(count, 2, generator=generator) * 2 - 1) * 0.75
    expected = clamp_gaze(gaze + actions.clamp(-1, 1) * step, limit)
    expected_velocity = (expected - gaze) / dt

    # The step starts from the camera's stored angles, not from the state it records.
    state = torch.randn(count, 4, generator=generator)
    angles = gaze * radius
    with torch_stream("cpu"):
        wp.launch(
            gaze_step_kernel,
            dim=count,
            inputs=[
                wp.from_torch(actions, dtype=wp.vec2f),
                wp.from_torch(state, dtype=wp.vec4f),
                wp.from_torch(angles, dtype=wp.vec2f),
                step,
                limit,
                dt,
                radius,
            ],
        )
    torch.testing.assert_close(state[:, :2], expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(state[:, 2:], expected_velocity, atol=2e-3, rtol=1e-5)
    torch.testing.assert_close(angles, expected * radius, atol=1e-5, rtol=1e-6)


def test_gaze_config_rejects_invalid_bounds():
    GazeActionCfg(asset_name="camera").validate_config()
    for field, value in (("step_size", 0.0), ("step_size", float("nan")), ("gaze_limit", math.inf)):
        cfg = GazeActionCfg(asset_name="camera")
        setattr(cfg, field, value)
        with pytest.raises(ValueError, match=field):
            cfg.validate_config()
    # The config names its term lazily, so Isaac Lab's action manager resolves it to the gaze action.
    assert GazeActionCfg(asset_name="camera").class_type._resolve() is GazeAction


def test_process_actions_moves_the_gaze_by_the_supplied_actions():
    """The term reads whatever actions it is handed: the manager's slice in place, or another tensor."""
    num_envs = 4
    angles = ProxyArray(wp.zeros(num_envs, dtype=wp.vec2f, device="cpu"))
    set_angles = []
    camera = SimpleNamespace(gaze_angles=angles, set_gaze_angles=lambda value: set_angles.append(value))
    term = object.__new__(GazeAction)
    term._env = SimpleNamespace(num_envs=num_envs, device="cpu", step_dt=0.5)
    term.cfg = GazeActionCfg(asset_name="camera", step_size=0.1, gaze_limit=1.0)
    term._asset = camera
    term.field_radius_deg = 20.0
    term._state = ProxyArray(wp.zeros(num_envs, dtype=wp.vec4f, device="cpu"))
    term._all_ids = wp.array(list(range(num_envs)), dtype=wp.int32, device="cpu")
    term._step_launch = term._raw = None

    manager_actions = torch.zeros(num_envs, 5)
    manager_actions[:, 2:4] = torch.tensor([[1.0, 0.0], [0.0, -1.0], [0.5, 0.5], [3.0, 0.0]])
    term.process_actions(manager_actions[:, 2:4])
    expected = clamp_gaze(manager_actions[:, 2:4].clamp(-1, 1) * 0.1, 1.0)
    torch.testing.assert_close(term.processed_actions, expected)
    torch.testing.assert_close(term.state[:, 2:], expected / 0.5)
    torch.testing.assert_close(angles.torch, expected * 20.0)
    assert set_angles[-1] is angles.torch
    assert term.raw_actions.data_ptr() == manager_actions[:, 2:4].data_ptr()

    supplied = torch.tensor([[-1.0, 0.0]]).repeat(num_envs, 1)
    term.process_actions(supplied)
    torch.testing.assert_close(term.processed_actions, clamp_gaze(expected - 0.1 * supplied.abs(), 1.0))
    assert term.raw_actions is supplied
    torch.testing.assert_close(
        gaze_state(SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: term))), term.state
    )

    # A gaze set on the camera directly is where the next step starts.
    angles.torch.copy_(torch.tensor([[10.0, 0.0]]).repeat(num_envs, 1))
    term.process_actions(torch.zeros(num_envs, 2))
    torch.testing.assert_close(term.processed_actions, torch.tensor([[0.5, 0.0]]).repeat(num_envs, 1))

    previous = term.state.clone()
    term.reset([3, 1])
    torch.testing.assert_close(term.state[[0, 2]], previous[[0, 2]])
    assert not term.state[[1, 3]].any() and not angles.torch[[1, 3]].any()
    term.reset()
    assert not term.state.any() and not angles.torch.any()
