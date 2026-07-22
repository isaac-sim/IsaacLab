# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual-color randomization render contract on the Newton-Warp (kit-less) renderer.

One renderer per file: renderers cannot share a process, so each backend lives in its own file
(mirroring ``test_camera_ppisp_gaussian_{newton,ovrtx}``). The scene + assertions are shared in
``color_dr_contract_common``. Kit-less: Newton physics + the Newton-Warp renderer need no Kit, so no
app is launched.
"""

import importlib.util

import pytest

_REQUIRED_MODULES = ("isaaclab_newton", "newton")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(bool(_MISSING_MODULES), reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}"),
]

if not _MISSING_MODULES:
    from color_dr_contract_common import assert_color_contract
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg


def test_visual_color_contract_newton_warp():
    """Color lands, per-prim/per-env distinct, and env_ids subset honored on the Newton-Warp renderer."""
    assert_color_contract(NewtonWarpRendererCfg(), physics_cfg=NewtonCfg(solver_cfg=MJWarpSolverCfg()))
