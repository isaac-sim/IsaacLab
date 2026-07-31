# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ``require_kit`` override of :func:`isaaclab.app.launch_simulation`.

``require_kit`` lets a tool state that it needs Kit for a reason the config cannot express --
the URDF/MJCF converters use it because they reach a Kit-only importer extension whenever the
standalone importer wheel is absent. The override is additive: it can turn a kitless launch into
a Kit one, never the reverse.

Kit is never actually started here: ``has_kit`` is faked so the launcher skips ``AppLauncher``
construction, and reaching ``_ensure_isaac_sim_available`` is the signal that the Kit branch was
taken. No Kit/GPU required.
"""

import pytest

import isaaclab.app.sim_launcher as sim_launcher
import isaaclab.utils as isaaclab_utils
from isaaclab.app import launch_simulation
from isaaclab.physics import PhysicsCfg


@pytest.fixture
def kit_branch_taken(monkeypatch: pytest.MonkeyPatch):
    """Report whether ``launch_simulation`` entered its Kit branch, without starting Kit."""
    taken: list[bool] = []
    # pretend Kit is already running so the launcher skips constructing AppLauncher
    monkeypatch.setattr(isaaclab_utils, "has_kit", lambda: True)
    monkeypatch.setattr(sim_launcher, "_ensure_isaac_sim_available", lambda: taken.append(True))
    return taken


def test_default_stays_kitless_for_a_kitless_config(kit_branch_taken):
    with launch_simulation(cfg=PhysicsCfg(), launcher_args={}):
        pass

    assert kit_branch_taken == []


def test_require_kit_launches_kit_for_a_kitless_config(kit_branch_taken):
    with launch_simulation(cfg=PhysicsCfg(), launcher_args={}, require_kit=True):
        pass

    assert kit_branch_taken == [True]


def test_require_kit_false_does_not_suppress_a_kit_config(kit_branch_taken):
    # '--viz kit' requires Kit on its own; require_kit=False must not override that
    launcher_args = {"visualizer": ["kit"], "visualizer_explicit": True}

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=launcher_args, require_kit=False):
        pass

    assert kit_branch_taken == [True]
