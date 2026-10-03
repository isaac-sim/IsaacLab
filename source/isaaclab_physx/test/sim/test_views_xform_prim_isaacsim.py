# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare the USD FrameView with Isaac Sim's XformPrim."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(physics="isaacsim_physx")

import pytest
import torch
import warp as wp

import omni.kit.app
import omni.usd

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import enable_extension
from isaaclab.sim.views import UsdFrameView as FrameView

enable_extension("isaacsim.core.experimental.prims")
from isaacsim.core.experimental.prims import XformPrim as _IsaacSimXformPrimView  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci]


@pytest.fixture(autouse=True)
def test_setup_teardown():
    sim_utils.create_new_stage()
    yield
    sim_utils.clear_stage()
    sim_utils.SimulationContext.clear_instance()


def test_compare_get_world_poses_with_isaacsim():
    """Compare get_world_poses with Isaac Sim's implementation."""
    stage = sim_utils.get_current_stage()
    num_prims = 10
    for i in range(num_prims):
        pos = (i * 2.0, i * 0.5, i * 1.5)
        quat = (0.0, 0.0, 0.0, 1.0) if i % 2 == 0 else (0.0, 0.0, 0.7071068, 0.7071068)
        sim_utils.create_prim(f"/World/Env_{i}/Object", "Xform", translation=pos, orientation=quat, stage=stage)

    pattern = "/World/Env_[^/]*/Object"
    isaacsim_paths = [f"/World/Env_{i}/Object" for i in range(num_prims)]
    isaaclab_view = FrameView(pattern, device="cpu")

    context = omni.usd.get_context()
    context.attach_stage_with_callback(sim_utils.get_current_stage_id())
    omni.kit.app.get_app().update()

    for kwargs in ({"reset_xform_properties": False}, {"reset_xform_op_properties": False}, {}):
        try:
            isaacsim_view = _IsaacSimXformPrimView(isaacsim_paths, **kwargs)
            break
        except TypeError as exc:
            if kwargs and next(iter(kwargs)) in str(exc):
                continue
            raise

    isaaclab_pos, isaaclab_quat = (a.torch for a in isaaclab_view.get_world_poses())
    isaacsim_pos, isaacsim_quat = (wp.to_torch(a).cpu() for a in isaacsim_view.get_world_poses())

    torch.testing.assert_close(isaaclab_pos, isaacsim_pos, atol=1e-5, rtol=0)
    # Isaac Sim returns (w, x, y, z); Isaac Lab returns (x, y, z, w)
    torch.testing.assert_close(isaaclab_quat, isaacsim_quat[:, [1, 2, 3, 0]], atol=1e-5, rtol=0)
