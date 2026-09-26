# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for gravity randomization and observations."""

import math
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.envs.mdp.events import randomize_physics_scene_gravity
from isaaclab.envs.mdp.observations import body_projected_gravity_b
from isaaclab.managers import EventTermCfg, SceneEntityCfg


@pytest.mark.parametrize("backend", ["physx", "ovphysx"])
def test_scene_wide_backends_use_configured_distribution(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    """PhysX and OvPhysX should use the distribution configured at initialization."""
    gravity_sink = SimpleNamespace()
    physics_manager = type(
        f"{backend}Manager",
        (),
        {"set_gravity": staticmethod(lambda gravity: setattr(gravity_sink, "value", gravity))},
    )
    monkeypatch.setattr(randomize_physics_scene_gravity, "_init_physx", lambda *_args: None)
    env = SimpleNamespace(
        device="cpu",
        sim=SimpleNamespace(
            cfg=SimpleNamespace(gravity=(0.0, 0.0, -9.81)),
            physics_manager=physics_manager,
        ),
    )
    cfg = EventTermCfg(
        func=randomize_physics_scene_gravity,
        params={
            "gravity_distribution_params": ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0)),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )
    gravity_event = randomize_physics_scene_gravity(cfg, env)
    gravity_event._carb = SimpleNamespace(Float3=lambda *values: values)
    gravity_event._physics_sim_view = physics_manager
    torch.manual_seed(0)
    gravity_event(env, env_ids=None, **cfg.params)
    assert gravity_sink.value == pytest.approx((1.0, 2.0, 3.0))


def test_newton_gravity_selectors_preserve_global_world(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment selectors must exclude Newton's trailing global-world gravity row."""
    wp.init()
    gravity = torch.full((5, 3), -9.81)
    model = SimpleNamespace(gravity=wp.from_torch(gravity, dtype=wp.vec3))
    notifications = []
    manager = SimpleNamespace(get_model=lambda: model, add_model_change=notifications.append)
    monkeypatch.setattr(randomize_physics_scene_gravity, "_init_newton", lambda *_args: None)
    env = SimpleNamespace(device="cpu", num_envs=4, sim=SimpleNamespace(physics_manager=type("NewtonManager", (), {})))
    cfg = EventTermCfg(
        func=randomize_physics_scene_gravity,
        params={"gravity_distribution_params": ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)), "operation": "abs"},
    )
    event = randomize_physics_scene_gravity(cfg, env)
    event._newton_manager, event._notify_model_properties = manager, "gravity"
    for selector, rows in (
        (None, [0, 1, 2, 3]),
        (slice(None), [0, 1, 2, 3]),
        (slice(1, None, 2), [1, 3]),
        (slice(-2, None), [2, 3]),
        (torch.tensor([1, 3], dtype=torch.int32), [1, 3]),
        (slice(0, 0), []),
    ):
        gravity.fill_(-9.81)
        notifications.clear()
        event(env, selector, **cfg.params)
        expected = torch.full_like(gravity, -9.81)
        expected[rows] = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(gravity, expected)
        assert notifications == (["gravity"] if rows else [])


@pytest.mark.unit
def test_body_projected_gravity_b_stacks_every_selected_body():
    """Each selected body receives its own environment's gravity, including integer selections."""
    num_envs = 2
    angles = (0.0, 0.5 * math.pi, math.pi)
    body_quat = torch.tensor([[math.sin(0.5 * a), 0.0, 0.0, math.cos(0.5 * a)] for a in angles]).repeat(num_envs, 1, 1)
    asset = SimpleNamespace(
        data=SimpleNamespace(
            body_quat_w=SimpleNamespace(torch=body_quat),
            GRAVITY_VEC_W=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, -9.81], [0.0, 9.81, 0.0]])),
        )
    )
    env = SimpleNamespace(scene={"robot": asset}, num_envs=num_envs)
    # R_x(a)^T applied to -Z in the first environment and +Y in the second.
    expected_z = [[0.0, -math.sin(a), -math.cos(a)] for a in angles]
    expected_y = [[0.0, math.cos(a), -math.sin(a)] for a in angles]
    expected = torch.tensor([expected_z, expected_y]).reshape(num_envs, -1)
    torch.testing.assert_close(body_projected_gravity_b(env, SceneEntityCfg("robot")), expected)
    for body_ids in ([1], 1):
        asset_cfg = SceneEntityCfg("robot", body_ids=body_ids)
        torch.testing.assert_close(body_projected_gravity_b(env, asset_cfg), expected[:, 3:6])
