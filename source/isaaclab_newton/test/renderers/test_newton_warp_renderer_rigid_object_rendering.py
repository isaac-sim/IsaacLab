# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton Warp adapter for the shared rigid-object rendering contract."""

import sys
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_newton.renderers.newton_warp_renderer import NewtonWarpRenderer

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


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_intrinsic_updates_preserve_shared_ray_storage(device):
    """Uniform updates reuse the ray field; nonuniform calibration fails without changing rays."""
    data = SimpleNamespace(
        _distortion=None,
        width=16,
        height=8,
        _intrinsic_status=wp.zeros(1, dtype=wp.int32, device=device),
        camera_rays=wp.empty((1, 8, 16, 2), dtype=wp.vec3f, device=device),
    )
    renderer = NewtonWarpRenderer.__new__(NewtonWarpRenderer)
    matrix = np.array([[20.0, 0, 8], [0, 20.0, 4], [0, 0, 1]], dtype=np.float32)
    matrices = wp.array([matrix, matrix], dtype=wp.mat33f, device=device)
    parameters = wp.empty((5, 2), dtype=wp.float32, device=device)
    pointer = data.camera_rays.ptr
    renderer.update_camera_intrinsics(data, matrices, parameters)
    before = data.camera_rays.numpy().copy()
    matrix[0, 0] *= 0.5
    matrices.assign(np.stack([matrix, matrix]))
    renderer.update_camera_intrinsics(data, matrices, parameters)
    after = data.camera_rays.numpy().copy()
    assert data.camera_rays.ptr == pointer
    assert not np.array_equal(before, after)
    different = np.stack([matrix, matrix])
    different[1, 0, 0] *= 0.5
    matrices.assign(different)
    with pytest.raises(ValueError, match="identical camera intrinsics"):
        renderer.update_camera_intrinsics(data, matrices, parameters)
    np.testing.assert_array_equal(data.camera_rays.numpy(), after)
