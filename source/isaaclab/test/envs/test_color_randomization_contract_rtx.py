# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual-color randomization render contract on the Isaac RTX (Kit/PhysX) renderer.

One renderer per file: renderers cannot share a process, so each backend lives in its own file
(mirroring ``test_camera_ppisp_gaussian_{newton,ovrtx}``). The scene + assertions are shared in
``color_dr_contract_common``. This file launches Kit via ``AppLauncher`` before importing the helper,
since the Isaac RTX renderer requires Kit.
"""

import importlib.util

import pytest

_REQUIRED_MODULES = ("isaaclab_physx",)
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(bool(_MISSING_MODULES), reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}"),
]

if not _MISSING_MODULES:
    # Kit must be launched before importing omni-dependent modules (ManagerBasedEnv et al.).
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(headless=True, enable_cameras=True).app

    from color_dr_contract_common import assert_color_contract
    from isaaclab_physx.renderers import IsaacRtxRendererCfg


def test_visual_color_contract_isaac_rtx():
    """Color lands, per-prim/per-env distinct, and env_ids subset honored on the Isaac RTX renderer."""
    assert_color_contract(IsaacRtxRendererCfg(), physics_cfg=None)  # PhysX is the SimulationCfg default
