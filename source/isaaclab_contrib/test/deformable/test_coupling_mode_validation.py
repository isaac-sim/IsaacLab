# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Pure-Python tests for rigid-deformable coupling-mode validation."""

from __future__ import annotations

import re
from types import ModuleType
from typing import Any, cast

import pytest
from isaaclab_newton.physics.newton_manager import NewtonManager

import isaaclab_contrib.deformable.coupled_featherstone_vbd_manager as coupled_featherstone_vbd_manager
import isaaclab_contrib.deformable.coupled_mjwarp_vbd_manager as coupled_mjwarp_vbd_manager
from isaaclab_contrib.deformable.coupled_featherstone_vbd_manager import NewtonCoupledFeatherstoneVBDManager
from isaaclab_contrib.deformable.coupled_mjwarp_vbd_manager import NewtonCoupledMJWarpVBDManager
from isaaclab_contrib.deformable.newton_manager_cfg import (
    CoupledFeatherstoneVBDSolverCfg,
    CoupledMJWarpVBDSolverCfg,
)

_INVALID_MODE = cast(Any, "invalid")


@pytest.mark.parametrize(
    "manager_module, manager_type, solver_cfg, constructor_names, expected_modes",
    [
        pytest.param(
            coupled_mjwarp_vbd_manager,
            NewtonCoupledMJWarpVBDManager,
            CoupledMJWarpVBDSolverCfg(coupling_mode=_INVALID_MODE),
            ("SolverMuJoCo", "SolverVBD", "SolverBase"),
            "{'one_way', 'two_way'}",
            id="mjwarp",
        ),
        pytest.param(
            coupled_featherstone_vbd_manager,
            NewtonCoupledFeatherstoneVBDManager,
            CoupledFeatherstoneVBDSolverCfg(coupling_mode=_INVALID_MODE),
            ("SolverFeatherstone", "SolverVBD", "SolverBase"),
            "{'kinematic', 'one_way', 'two_way'}",
            id="featherstone",
        ),
    ],
)
def test_invalid_coupling_mode_fails_before_solver_construction_or_state_mutation(
    monkeypatch: pytest.MonkeyPatch,
    manager_module: ModuleType,
    manager_type: type,
    solver_cfg: CoupledMJWarpVBDSolverCfg | CoupledFeatherstoneVBDSolverCfg,
    constructor_names: tuple[str, ...],
    expected_modes: str,
) -> None:
    """Invalid modes fail before constructors run or manager state changes."""
    sentinel = object()

    def unexpected_constructor(*args, **kwargs):
        pytest.fail("solver construction must not run for an invalid coupling mode")

    for constructor_name in constructor_names:
        monkeypatch.setattr(manager_module, constructor_name, unexpected_constructor)

    monkeypatch.setattr(manager_type, "_coupling_mode", sentinel)
    monkeypatch.setattr(manager_type, "_rigid_solver", sentinel, raising=False)
    monkeypatch.setattr(manager_type, "_soft_solver", sentinel, raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", sentinel)
    monkeypatch.setattr(NewtonManager, "_use_single_state", sentinel)
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", sentinel)

    with pytest.raises(ValueError, match=rf"coupling_mode='invalid'.*{re.escape(expected_modes)}"):
        manager_type._build_solver(object(), solver_cfg)

    assert manager_type._coupling_mode is sentinel
    assert manager_type._rigid_solver is sentinel
    assert manager_type._soft_solver is sentinel
    assert NewtonManager._solver is sentinel
    assert NewtonManager._use_single_state is sentinel
    assert NewtonManager._needs_collision_pipeline is sentinel
