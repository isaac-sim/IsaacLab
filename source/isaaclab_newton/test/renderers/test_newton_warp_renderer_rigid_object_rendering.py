# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton Warp adapter for the shared rigid-object rendering contract."""

import sys
from pathlib import Path

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import pytest
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.renderers import NewtonWarpRendererCfg

from isaaclab.sim import build_simulation_context

_CONTRACT_DIR = Path(__file__).resolve().parents[3] / "isaaclab" / "test" / "renderers"
if str(_CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTRACT_DIR))

from rigid_object_rendering_contract import (  # noqa: E402
    RigidObjectRenderingBackend,
    run_rigid_object_scale_and_pose_rendering_contract,
)

pytestmark = [pytest.mark.integration, pytest.mark.rendering, pytest.mark.isaacsim_ci]


def test_kinematic_rigid_object_scale_and_pose_are_rendered() -> None:
    """Kinematic PhysX transforms and root scale must reach Newton Warp."""
    run_rigid_object_scale_and_pose_rendering_contract(
        RigidObjectRenderingBackend(
            name="newton_warp (PhysX)",
            simulation_context_factory=lambda: build_simulation_context(device="cuda:0", gravity_enabled=False),
            renderer_cfg=NewtonWarpRendererCfg(),
            cleanup=NewtonManager.clear,
        )
    )
