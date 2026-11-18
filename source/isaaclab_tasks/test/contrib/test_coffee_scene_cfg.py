# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coffee fill configuration follows the vessels before runtime initialization."""

import pytest

from isaaclab_tasks.contrib.rizon_sharpa_coffee.demo_env_cfg import CoffeeDemoEnvCfg, configure_scene

pytestmark = [pytest.mark.unit, pytest.mark.kitless, pytest.mark.windows_ci]


def test_coffee_material_fills_follow_configured_vessels():
    """Resolve default props for legacy configs and preserve explicit vessel positions."""
    cfg = CoffeeDemoEnvCfg(water_enabled=True)
    cfg.sim.visualizer_cfgs = []
    cfg.scene.bowl.init_state.pos = (0.51, -0.11, 0.02)
    cfg.scene.cup.init_state.pos = (0.63, 0.01, 0.03)
    configure_scene(cfg)
    assert cfg.scene.coffee.init_state.pos == (0.51, -0.11, 0.02)
    assert cfg.scene.water.init_state.pos == (0.63, 0.01, 0.03)

    cfg = CoffeeDemoEnvCfg(water_enabled=True)
    cfg.sim.visualizer_cfgs = []
    cfg.scene.bowl = None
    cfg.scene.cup = None
    configure_scene(cfg)
    assert cfg.scene.coffee.init_state.pos == cfg.scene.bowl.init_state.pos
    assert cfg.scene.water.init_state.pos == cfg.scene.cup.init_state.pos
