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
