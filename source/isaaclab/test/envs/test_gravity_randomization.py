# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for gravity randomization and observations."""

import math
import sys
from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_newton.envs.mdp import randomize_world_gravity
from isaaclab_newton.physics import NewtonManager
from isaaclab_ov.envs.mdp import randomize_physics_scene_gravity as ov_gravity
from isaaclab_physx.envs.mdp import randomize_physics_scene_gravity as physx_gravity

from isaaclab.envs.mdp.events import randomize_physics_scene_gravity
from isaaclab.envs.mdp.observations import body_projected_gravity_b
from isaaclab.managers import EventManager, EventTermCfg, SceneEntityCfg


@pytest.mark.parametrize("backend", ["physx", "ovphysx"])
def test_scene_wide_backends_use_configured_distribution(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    """PhysX and OvPhysX should use the distribution configured at initialization."""
    gravity_sink = SimpleNamespace()
    physics_manager = type(
        f"{backend}Manager",
        (),
        {"set_gravity": staticmethod(lambda gravity: setattr(gravity_sink, "value", gravity))},
    )
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(Float3=lambda *values: values))
    term_type = physx_gravity if backend == "physx" else ov_gravity
    env = SimpleNamespace(
        device="cpu",
        num_envs=2,
        sim=SimpleNamespace(
            cfg=SimpleNamespace(gravity=(0.0, 0.0, -9.81)),
            physics_manager=physics_manager,
            physics_sim_view=physics_manager,
            is_playing=lambda: True,
        ),
    )
    cfg = EventTermCfg(
        func=term_type,
        mode="startup",
        params={
            "gravity_distribution_params": ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0)),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )
    manager = EventManager({"gravity": cfg}, env)
    torch.manual_seed(0)
    manager.apply("startup")
    assert gravity_sink.value == pytest.approx((1.0, 2.0, 3.0))


def test_newton_gravity_selectors_preserve_global_world(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment selectors must exclude Newton's trailing global-world gravity row."""
    wp.init()
    gravity = torch.full((5, 3), -9.81)
    model = SimpleNamespace(gravity=wp.from_torch(gravity, dtype=wp.vec3))
    notifications = []

    class CustomDynamics(NewtonManager):
        pass

    monkeypatch.setattr(CustomDynamics, "get_model", classmethod(lambda cls: model))
    monkeypatch.setattr(CustomDynamics, "add_model_change", classmethod(lambda cls, flag: notifications.append(flag)))
    env = SimpleNamespace(
        device="cpu",
        num_envs=4,
        sim=SimpleNamespace(physics_manager=CustomDynamics, is_playing=lambda: True),
    )
    cfg = EventTermCfg(
        func=randomize_world_gravity,
        mode="startup",
        params={"gravity_distribution_params": ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)), "operation": "abs"},
    )
    from newton import ModelFlags

    manager = EventManager({"gravity": cfg}, env)
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
        manager.apply("startup", env_ids=selector)
        expected = torch.full_like(gravity, -9.81)
        expected[rows] = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(gravity, expected)
        assert notifications == ([ModelFlags.MODEL_PROPERTIES] if rows else [])

    with pytest.warns(DeprecationWarning, match="Select the physics backend"):
        legacy = randomize_physics_scene_gravity(cfg, env)
    gravity.fill_(-9.81)
    legacy(env, torch.tensor([2]), **cfg.params)
    expected = torch.full_like(gravity, -9.81)
    expected[2] = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(gravity, expected)


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
