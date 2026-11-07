# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check the migration from contrib coupling imports to the Newton package."""

import pytest
from isaaclab_newton.physics import coupling

pytestmark = pytest.mark.kitless


def test_legacy_coupling_imports_preserve_classes_and_warn() -> None:
    """Old package and submodule imports retain class identity while directing callers to Newton."""
    with pytest.warns(DeprecationWarning, match="isaaclab_newton.physics.coupling") as records:
        import isaaclab_contrib.coupling as legacy
        from isaaclab_contrib.coupling.coupler import CouplerSolver
        from isaaclab_contrib.coupling.coupler_cfg import (
            CouplerAdmmCfg,
            CouplerCfg,
            CouplerEntryCfg,
            CouplerProxyCfg,
            CouplerProxyMappingCfg,
        )

    assert len(records) == 1
    assert legacy.CouplerSolver is CouplerSolver is coupling.CouplerSolver
    assert legacy.CouplerAdmmCfg is CouplerAdmmCfg is coupling.CouplerAdmmCfg
    assert legacy.CouplerCfg is CouplerCfg is coupling.CouplerCfg
    assert legacy.CouplerEntryCfg is CouplerEntryCfg is coupling.CouplerEntryCfg
    assert legacy.CouplerProxyCfg is CouplerProxyCfg is coupling.CouplerProxyCfg
    assert legacy.CouplerProxyMappingCfg is CouplerProxyMappingCfg is coupling.CouplerProxyMappingCfg
