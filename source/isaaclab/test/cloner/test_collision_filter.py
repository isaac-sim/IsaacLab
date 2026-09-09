# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility tests for standalone clone collision filtering."""

from unittest import mock

import pytest

import isaaclab.physics._physx_collision_filter as physx_collision_filter
from isaaclab.cloner import filter_collisions
from isaaclab.physics import PhysicsManager


def test_filter_collisions_compatibility_facade(monkeypatch) -> None:
    stage = object()
    with mock.patch.object(physx_collision_filter, "_author_compact_environment_isolation") as author:
        monkeypatch.setattr(PhysicsManager, "_collision_filter_applied", False)
        with pytest.warns(DeprecationWarning, match=r"cloner\.filter_collisions\(\) is deprecated"):
            filter_collisions(stage, "/physicsScene", "/Filters", ("/env_0", "/env_1"), ("/Ground",))
        author.assert_called_once_with(stage, "/physicsScene", "/Filters", ["/env_0", "/env_1"], ["/Ground"])

        author.reset_mock()
        monkeypatch.setattr(PhysicsManager, "_collision_filter_applied", True)
        with pytest.raises(RuntimeError, match=r"cannot run after PhysicsManager\.apply_collision_filter"):
            filter_collisions(object(), "/physicsScene", "/Filters", ("/env_0",))
        author.assert_not_called()
