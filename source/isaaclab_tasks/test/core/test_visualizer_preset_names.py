# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The ``newton`` visualizer name and the ``newton`` physics alias do not select each other."""

import pytest
from isaaclab_visualizers.newton import NewtonGLBackendCfg

from isaaclab_tasks.utils.hydra import resolve_task_config

pytestmark = [pytest.mark.unit]

_TASK = "Isaac-Cartpole-Camera-Direct"


def test_global_newton_preset_selects_physics_without_opening_a_viewer():
    """``presets=newton`` picks Newton physics and leaves the visualizer unset."""
    env_cfg, _ = resolve_task_config(_TASK, None, overrides=["presets=newton"])

    assert type(env_cfg.sim.physics).__name__ == "NewtonCfg"
    assert env_cfg.sim.visualizer_cfgs.backend == []


def test_visualizer_newton_selects_the_viewer_only():
    """``visualizer=newton`` opens the Newton viewer and does not change the physics backend."""
    env_cfg, _ = resolve_task_config(_TASK, None, overrides=["visualizer=newton"])

    assert [type(view.backend) for view in env_cfg.sim.visualizer_cfgs] == [NewtonGLBackendCfg]
