# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, is_dataclass

import pytest

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


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

    copied_cfg = cfg.copy()
    replaced_cfg = cfg.replace(dt=0.02)
    cfg.from_dict({"dt": 0.03})

    assert copied_cfg.to_dict()["dt"] == 0.01
    assert copied_cfg is not cfg
    assert copied_cfg.physics_material is not cfg.physics_material
    assert replaced_cfg.dt == 0.02
    assert cfg.dt == 0.03
    assert cfg.validate() == []


def test_simulation_cfg_mutable_defaults_are_independent():
    """Standard dataclass factories should not share mutable defaults between instances."""
    first = SimulationCfg()
    second = SimulationCfg()

    first.visualizer_cfgs.append(object())

    assert second.visualizer_cfgs == []
    assert first.physics_material is not second.physics_material


def test_simulation_cfg_supports_legacy_configclass_subclasses():
    """Existing downstream configclass subclasses should remain compatible during migration."""

    with pytest.deprecated_call(match="Use dataclasses.dataclass with ConfigMixin"):

        @configclass
        class LegacyDerivedSimulationCfg(SimulationCfg):
            substeps: int = 1

    cfg = LegacyDerivedSimulationCfg(dt=0.01, substeps=2)

    assert cfg.dt == 0.01
    assert cfg.substeps == 2
