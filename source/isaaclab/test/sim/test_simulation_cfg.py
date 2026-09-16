# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, is_dataclass

import pytest

from isaaclab.sim import SimulationCfg
from isaaclab.utils import config_to_dict, configclass, copy_config, replace_config, update_config, validate_config


def test_simulation_cfg_uses_standard_dataclass():
    """SimulationCfg should support standard dataclass inheritance."""

    @dataclass
    class DerivedSimulationCfg(SimulationCfg):
        substeps: int = 1

    cfg = DerivedSimulationCfg(dt=0.01, substeps=2)

    assert is_dataclass(SimulationCfg)
    assert cfg.dt == 0.01
    assert cfg.substeps == 2


def test_simulation_cfg_preserves_config_helpers():
    """The dataclass migration should preserve the public configuration helper methods."""
    cfg = SimulationCfg(dt=0.01)

    copied_cfg = copy_config(cfg)
    replaced_cfg = replace_config(cfg, dt=0.02)
    update_config(cfg, {"dt": 0.03})

    assert config_to_dict(copied_cfg)["dt"] == 0.01
    assert copied_cfg is not cfg
    assert copied_cfg.physics_material is not cfg.physics_material
    assert replaced_cfg.dt == 0.02
    assert cfg.dt == 0.03
    assert validate_config(cfg) == []


def test_simulation_cfg_mutable_defaults_are_independent():
    """Standard dataclass factories should not share mutable defaults between instances."""
    first = SimulationCfg()
    second = SimulationCfg()

    first.visualizer_cfgs.append(object())

    assert second.visualizer_cfgs == []
    assert first.physics_material is not second.physics_material


def test_simulation_cfg_supports_legacy_configclass_subclasses():
    """Existing downstream configclass subclasses should remain compatible during migration."""

    with pytest.deprecated_call(match="functional configuration utilities"):

        @configclass
        class LegacyDerivedSimulationCfg(SimulationCfg):
            substeps: int = 1

    cfg = LegacyDerivedSimulationCfg(dt=0.01, substeps=2)

    assert cfg.dt == 0.01
    assert cfg.substeps == 2
