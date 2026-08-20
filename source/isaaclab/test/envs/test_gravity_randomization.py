# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only tests for gravity-randomization backend dispatch."""

from types import SimpleNamespace

import torch

from isaaclab.envs.mdp.events import randomize_physics_scene_gravity


def test_ovphysx_gravity_randomization_uses_the_manager_setter():
    """OvPhysX must update gravity through its manager instead of a Kit simulation view."""
    applied_gravity = []

    class FakeOvPhysxManager:
        @staticmethod
        def set_gravity(gravity):
            applied_gravity.append(gravity)

    event = object.__new__(randomize_physics_scene_gravity)
    event._ovphysx_manager = FakeOvPhysxManager
    event._dist_param_0 = torch.zeros(3)
    event._dist_param_1 = torch.zeros(3)

    event._call_ovphysx(
        SimpleNamespace(sim=SimpleNamespace(cfg=SimpleNamespace(gravity=(0.0, 0.0, -9.81)))),
        operation="abs",
    )

    assert applied_gravity == [(0.0, 0.0, 0.0)]
