# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only tests for gravity-randomization backend dispatch."""

from types import SimpleNamespace

import torch

from isaaclab.envs.mdp.events import randomize_physics_scene_gravity
from isaaclab.managers import EventTermCfg


def test_ovphysx_gravity_randomization_dispatches_to_the_manager_setter():
    """OvPhysX must select and call its manager path instead of the Kit simulation view."""
    applied_gravity = []

    class FakeOvPhysxManager:
        @staticmethod
        def set_gravity(gravity):
            applied_gravity.append(gravity)

    env = SimpleNamespace(
        device="cpu",
        sim=SimpleNamespace(
            cfg=SimpleNamespace(gravity=(0.0, 0.0, -9.81)),
            physics_manager=FakeOvPhysxManager,
        ),
    )
    cfg = EventTermCfg(
        func=randomize_physics_scene_gravity,
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            "operation": "abs",
        },
    )
    event = randomize_physics_scene_gravity(cfg, env)

    event(
        env,
        env_ids=torch.tensor([0]),
        gravity_distribution_params=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        operation="abs",
    )

    assert event._backend == "ovphysx"
    assert applied_gravity == [(0.0, 0.0, 0.0)]
