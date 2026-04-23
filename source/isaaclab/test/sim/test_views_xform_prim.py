# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD backend tests for FrameView.

Imports the shared contract tests and provides the USD-specific
``view_factory`` fixture.  Also includes USD-only tests for visibility,
prim ordering, xformOp standardization, and Isaac Sim comparison.
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

from pxr import Gf, UsdGeom  # noqa: E402

try:
    from isaacsim.core.experimental.prims import XformPrim as _IsaacSimXformPrimView
except (ModuleNotFoundError, ImportError):
    _IsaacSimXformPrimView = None

from frame_view_contract_utils import *  # noqa: F401, F403, E402
from frame_view_contract_utils import CHILD_OFFSET, ViewBundle  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.views import UsdFrameView as FrameView  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402

PARENT_POS = (0.0, 0.0, 1.0)


@pytest.fixture(autouse=True)
def test_setup_teardown():
    sim_utils.create_new_stage()
    sim_utils.update_stage()
    yield
    sim_utils.clear_stage()
    sim_utils.SimulationContext.clear_instance()


# ------------------------------------------------------------------
# Contract fixture
# ------------------------------------------------------------------


def _get_parent_positions(num_envs, device="cpu"):
    """Read parent Xform positions from USD."""
    stage = sim_utils.get_current_stage()
    xform_cache = UsdGeom.XformCache()
    positions = []
    for i in range(num_envs):
        prim = stage.GetPrimAtPath(f"/World/Parent_{i}")
        tf = xform_cache.GetLocalToWorldTransform(prim)
        t = tf.ExtractTranslation()
        positions.append([float(t[0]), float(t[1]), float(t[2])])
    return torch.tensor(positions, dtype=torch.float32, device=device)


def _set_parent_positions(positions, num_envs):
    """Write parent Xform positions to USD."""
    from pxr import Sdf  # noqa: PLC0415

    stage = sim_utils.get_current_stage()
    with Sdf.ChangeBlock():
        for i in range(num_envs):
            prim = stage.GetPrimAtPath(f"/World/Parent_{i}")
            pos = positions[i]
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))


@pytest.fixture
def view_factory():
    """USD factory: parent Xform at PARENT_POS + child Xform at CHILD_OFFSET."""

    def factory(num_envs: int, device: str) -> ViewBundle:
        stage = sim_utils.get_current_stage()
        for i in range(num_envs):
            sim_utils.create_prim(f"/World/Parent_{i}", "Xform", translation=PARENT_POS, stage=stage)
            sim_utils.create_prim(f"/World/Parent_{i}/Child", "Xform", translation=CHILD_OFFSET, stage=stage)

        view = FrameView("/World/Parent_.*/Child", device=device)
        return ViewBundle(
            view=view,
            get_parent_pos=_get_parent_positions,
            set_parent_pos=_set_parent_positions,
            teardown=lambda: None,
        )

    return factory


# ==================================================================
# USD-only: Visibility
# ==================================================================


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_visibility_toggle(device):
    """Test toggling visibility multiple times."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()
    num_prims = 3
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Object_{i}", "Xform", stage=stage)

    view = FrameView("/World/Object_.*", device=device)

    assert torch.all(view.get_visibility())

    view.set_visibility(torch.zeros(num_prims, dtype=torch.bool, device=device))
    assert not torch.any(view.get_visibility())

    view.set_visibility(torch.ones(num_prims, dtype=torch.bool, device=device))
    assert torch.all(view.get_visibility())

    view.set_visibility(
        torch.tensor([False], dtype=torch.bool, device=device), indices=wp.array([1], dtype=wp.int32, device=device)
    )
    vis = view.get_visibility()
    assert vis[0] and not vis[1] and vis[2]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_visibility_parent_inheritance(device):
    """Making a parent invisible hides all children."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/Parent", "Xform", stage=stage)
    for i in range(4):
        sim_utils.create_prim(f"/World/Parent/Child_{i}", "Xform", stage=stage)

    parent_view = FrameView("/World/Parent", device=device)
    children_view = FrameView("/World/Parent/Child_.*", device=device)

    parent_view.set_visibility(torch.tensor([False], dtype=torch.bool, device=device))
    assert not torch.any(children_view.get_visibility())

    parent_view.set_visibility(torch.tensor([True], dtype=torch.bool, device=device))
    assert torch.all(children_view.get_visibility())


# ==================================================================
# USD-only: Prim ordering
# ==================================================================


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prim_ordering_follows_creation_order(device):
    """Prims are returned in USD creation order (DFS), not alphabetical."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()
    num_envs = 3
    for i in range(num_envs):
        sim_utils.create_prim(f"/World/Env_{i}/Object_1", "Xform", stage=stage)
        sim_utils.create_prim(f"/World/Env_{i}/Object_0", "Xform", stage=stage)
        sim_utils.create_prim(f"/World/Env_{i}/Object_A", "Xform", stage=stage)

    view = FrameView("/World/Env_.*/Object_.*", device=device)
    expected = []
    for i in range(num_envs):
        expected += [f"/World/Env_{i}/Object_1", f"/World/Env_{i}/Object_0", f"/World/Env_{i}/Object_A"]

    assert view.prim_paths == expected


# ==================================================================
# USD-only: xformOp standardization
# ==================================================================


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_standardize_transform_op(device):
    """FrameView standardizes a prim with xformOp:transform to translate/orient/scale."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    expected_pos = (3.0, -1.0, 0.5)
    matrix = Gf.Matrix4d(1.0)
    matrix.SetTranslateOnly(Gf.Vec3d(*expected_pos))

    stage = sim_utils.get_current_stage()
    prim = stage.DefinePrim("/World/TransformPrim", "Xform")
    UsdGeom.Xformable(prim).AddTransformOp().Set(matrix)

    view = FrameView("/World/TransformPrim", device=device)
    assert sim_utils.validate_standard_xform_ops(view.prims[0])

    ordered_ops = UsdGeom.Xformable(view.prims[0]).GetOrderedXformOps()
    op_names = [op.GetOpName() for op in ordered_ops]
    assert op_names == ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    assert ordered_ops[0].Get() == Gf.Vec3d(*expected_pos)


# ==================================================================
# USD-only: Nested hierarchy (frame + target)
# ==================================================================


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nested_hierarchy_world_poses(device):
    """World pose of nested child == sum of parent + child translations."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()
    frame_positions = [(0.0, 0.0, 0.0), (0.0, 10.0, 5.0), (0.0, 3.0, 5.0)]
    target_positions = [(0.0, 20.0, 10.0), (0.0, 30.0, 20.0), (0.0, 50.0, 10.0)]

    for i in range(3):
        sim_utils.create_prim(f"/World/Frame_{i}", "Xform", translation=frame_positions[i], stage=stage)
        sim_utils.create_prim(f"/World/Frame_{i}/Target", "Xform", translation=target_positions[i], stage=stage)

    frames_view = FrameView("/World/Frame_.*", device=device)
    targets_view = FrameView("/World/Frame_.*/Target", device=device)

    frames_view.set_local_poses(translations=torch.tensor(frame_positions, device=device))
    targets_view.set_local_poses(translations=torch.tensor(target_positions, device=device))

    world_pos = wp.to_torch(targets_view.get_world_poses()[0])
    expected = torch.tensor(
        [[f[j] + t[j] for j in range(3)] for f, t in zip(frame_positions, target_positions)],
        device=device,
    )
    torch.testing.assert_close(world_pos, expected, atol=1e-5, rtol=0)


# ==================================================================
# USD-only: Comparison with Isaac Sim
# ==================================================================


def test_compare_get_world_poses_with_isaacsim():
    """Compare get_world_poses with Isaac Sim's implementation."""
    if _IsaacSimXformPrimView is None:
        pytest.skip("Isaac Sim is not available")

    stage = sim_utils.get_current_stage()
    num_prims = 10
    for i in range(num_prims):
        pos = (i * 2.0, i * 0.5, i * 1.5)
        quat = (0.0, 0.0, 0.0, 1.0) if i % 2 == 0 else (0.0, 0.0, 0.7071068, 0.7071068)
        sim_utils.create_prim(f"/World/Env_{i}/Object", "Xform", translation=pos, orientation=quat, stage=stage)

    pattern = "/World/Env_.*/Object"
    isaaclab_view = FrameView(pattern, device="cpu")
    isaacsim_view = _IsaacSimXformPrimView(pattern, reset_xform_properties=False)

    isaaclab_pos = wp.to_torch(isaaclab_view.get_world_poses()[0])
    isaacsim_pos, isaacsim_quat = isaacsim_view.get_world_poses()
    if not isinstance(isaacsim_pos, torch.Tensor):
        isaacsim_pos = torch.tensor(isaacsim_pos, dtype=torch.float32)

    torch.testing.assert_close(isaaclab_pos, isaacsim_pos, atol=1e-5, rtol=0)


# ==================================================================
# USD-only: Franka integration
# ==================================================================


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_with_franka_robots(device):
    """Verify FrameView works with real Franka robot USD assets."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()
    franka_usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"

    sim_utils.create_prim("/World/Franka_1", "Xform", usd_path=franka_usd_path, stage=stage)
    sim_utils.create_prim("/World/Franka_2", "Xform", usd_path=franka_usd_path, stage=stage)

    view = FrameView("/World/Franka_.*", device=device)
    assert view.count == 2

    positions = wp.to_torch(view.get_world_poses()[0])
    torch.testing.assert_close(positions, torch.zeros(2, 3, device=device), atol=1e-5, rtol=0)

    new_pos = torch.tensor([[10.0, 10.0, 0.0], [-40.0, -40.0, 0.0]], device=device)
    new_quat = torch.tensor([[0.0, 0.0, 0.7071068, 0.7071068], [0.0, 0.0, -0.7071068, 0.7071068]], device=device)
    view.set_world_poses(positions=new_pos, orientations=new_quat)

    ret_pos = wp.to_torch(view.get_world_poses()[0])
    torch.testing.assert_close(ret_pos, new_pos, atol=1e-5, rtol=0)
