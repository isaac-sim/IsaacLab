# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual-color randomization render contract on the OVRTX (kit-less) renderer.

One renderer per file: renderers cannot share a process, so each backend lives in its own file
(mirroring ``test_camera_ppisp_gaussian_{newton,ovrtx}``). The scene + assertions are shared in
``color_dr_contract_common``. OVRTX is kit-less (Newton physics + OVRTX renderer): no Kit is launched,
since ovrtx and Kit cannot share a process.

The OVRTX renderer loads the stage via ``Renderer.open_usd_from_string`` (added in ovrtx 0.3.0), which
``isaaclab_ov`` pins (``ovrtx>=0.3.0,<0.4.0``). The test is version-gated so it skips cleanly on an
environment with an older wheel instead of erroring at stage load.
"""

import importlib.metadata
import importlib.util

import pytest

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx", "isaaclab_newton")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]


def _ovrtx_below_0_3() -> bool:
    """True if the installed ovrtx is older than 0.3.0 (lacks ``Renderer.open_usd_from_string``)."""
    try:
        major, minor = (int(part) for part in importlib.metadata.version("ovrtx").split(".")[:2])
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return False
    return (major, minor) < (0, 3)


pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(bool(_MISSING_MODULES), reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}"),
    pytest.mark.skipif(
        _ovrtx_below_0_3(),
        reason="OVRTX renderer requires ovrtx>=0.3.0 (Renderer.open_usd_from_string); installed wheel is older",
    ),
]

if not _MISSING_MODULES and not _ovrtx_below_0_3():
    from color_dr_contract_common import assert_color_contract
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_ov.renderers import OVRTXRendererCfg


def test_visual_color_contract_ovrtx():
    """Color lands, per-prim/per-env distinct, and env_ids subset honored on the OVRTX renderer."""
    assert_color_contract(OVRTXRendererCfg(), physics_cfg=NewtonCfg(solver_cfg=MJWarpSolverCfg()))
