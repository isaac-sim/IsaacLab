# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for scene-wide gravity randomization."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp.events import randomize_physics_scene_gravity
from isaaclab.managers import EventTermCfg
from isaaclab.utils import math as math_utils


@pytest.mark.parametrize("backend", ["physx", "ovphysx"])
@pytest.mark.parametrize(("distribution", "expected"), [("uniform", 1.0), ("log_uniform", 2.0), ("gaussian", 3.0)])
def test_scene_wide_backends_use_configured_distribution(
    monkeypatch: pytest.MonkeyPatch, backend: str, distribution: str, expected: float
) -> None:
    """PhysX and OvPhysX should use the distribution configured at initialization."""

    class OvPhysxManager:
        gravity = None

        @classmethod
        def set_gravity(cls, gravity) -> None:
            cls.gravity = gravity

    class PhysxManager:
        pass

    monkeypatch.setattr(randomize_physics_scene_gravity, "_init_physx", lambda *_args: None)
    env = SimpleNamespace(
        device="cpu",
        sim=SimpleNamespace(
            cfg=SimpleNamespace(gravity=(0.0, 0.0, -9.81)),
            physics_manager=PhysxManager if backend == "physx" else OvPhysxManager,
        ),
    )
    for sampler_name, sampled_value in (
        ("sample_uniform", 1.0),
        ("sample_log_uniform", 2.0),
        ("sample_gaussian", 3.0),
    ):
        monkeypatch.setattr(
            math_utils,
            sampler_name,
            lambda _param_0, _param_1, size, device, value=sampled_value: torch.full(size, value, device=device),
        )
    cfg = EventTermCfg(
        func=randomize_physics_scene_gravity,
        params={
            "gravity_distribution_params": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            "operation": "abs",
            "distribution": distribution,
        },
    )
    gravity_event = randomize_physics_scene_gravity(cfg, env)

    if backend == "physx":
        physics_sim_view = SimpleNamespace(set_gravity=lambda gravity: setattr(physics_sim_view, "gravity", gravity))
        gravity_event._carb = SimpleNamespace(Float3=lambda *values: values)
        gravity_event._physics_sim_view = physics_sim_view

    gravity_event(env, env_ids=None, **cfg.params)
    actual = physics_sim_view.gravity if backend == "physx" else OvPhysxManager.gravity

    assert actual == pytest.approx((expected, expected, expected))
