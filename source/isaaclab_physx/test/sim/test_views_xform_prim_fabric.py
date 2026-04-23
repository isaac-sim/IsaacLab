# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX Fabric backend tests for FrameView.

Imports the shared contract tests and provides the Fabric-specific
``view_factory`` fixture (SimulationContext with use_fabric=True,
Camera prim type for Fabric SelectPrims compatibility).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "isaaclab" / "test" / "sim"))

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from frame_view_contract_utils import *  # noqa: F401, F403, E402
from frame_view_contract_utils import CHILD_OFFSET, ViewBundle, test_set_world_updates_local  # noqa: E402
from isaaclab_physx.sim.views import FabricFrameView as FrameView  # noqa: E402

from pxr import Gf, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

PARENT_POS = (0.0, 0.0, 1.0)


@pytest.fixture(autouse=True)
def test_setup_teardown():
    sim_utils.create_new_stage()
    sim_utils.update_stage()
    yield
    sim_utils.clear_stage()
    sim_utils.SimulationContext.clear_instance()


def _skip_if_unavailable(device: str):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device == "cpu":
        pytest.skip("Warp fabricarray operations on CPU have known issues")


# ------------------------------------------------------------------
# Parent position helpers (via USD xformOps)
# ------------------------------------------------------------------


def _get_parent_positions(num_envs, device="cpu"):
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
    from pxr import Sdf  # noqa: PLC0415

    stage = sim_utils.get_current_stage()
    with Sdf.ChangeBlock():
        for i in range(num_envs):
            prim = stage.GetPrimAtPath(f"/World/Parent_{i}")
            pos = positions[i]
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))


# ------------------------------------------------------------------
# Contract fixture
# ------------------------------------------------------------------


@pytest.fixture
def view_factory():
    """Fabric factory: Camera child at CHILD_OFFSET under parent Xforms, with Fabric enabled."""

    def factory(num_envs: int, device: str) -> ViewBundle:
        _skip_if_unavailable(device)

        stage = sim_utils.get_current_stage()
        for i in range(num_envs):
            sim_utils.create_prim(f"/World/Parent_{i}", "Xform", translation=PARENT_POS, stage=stage)
            sim_utils.create_prim(f"/World/Parent_{i}/Child", "Camera", translation=CHILD_OFFSET, stage=stage)

        sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=device, use_fabric=True))
        view = FrameView("/World/Parent_.*/Child", device=device)
        return ViewBundle(
            view=view,
            get_parent_pos=_get_parent_positions,
            set_parent_pos=_set_parent_positions,
            teardown=lambda: None,
        )

    return factory


# ------------------------------------------------------------------
# Override shared contract test with expected failure for Fabric.
# FabricFrameView.set_world_poses writes to Fabric worldMatrix only; the local
# pose (read via USD) does not reflect the change because there is no
# Fabric → USD writeback for local poses.  This is tracked as Issue #5
# (localMatrix: set_local_poses falls back to USD).
# ------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "Issue #5: FabricFrameView.set_world_poses writes to Fabric worldMatrix only. "
        "get_local_poses reads from stale USD because there is no Fabric→USD "
        "writeback for local poses."
    ),
    strict=True,
)
def test_set_world_updates_local(device, view_factory):  # noqa: F811
    """Override the shared test to mark it as expected failure."""
    from frame_view_contract_utils import test_set_world_updates_local as _impl  # noqa: PLC0415

    _impl(device, view_factory)


# ------------------------------------------------------------------
# Fabric-specific tests (not in shared contract)
# ------------------------------------------------------------------


@wp.kernel
def _fill_position(out: wp.array(dtype=wp.float32, ndim=2), x: float, y: float, z: float):
    i = wp.tid()
    out[i, 0] = wp.float32(x)
    out[i, 1] = wp.float32(y)
    out[i, 2] = wp.float32(z)


@pytest.mark.parametrize("device", ["cuda:0"])
def test_fabric_set_world_does_not_write_back_to_usd(device, view_factory):
    """Verify that set_world_poses in Fabric mode does NOT sync back to USD.

    This confirms the removal of sync_usd_on_fabric_write.  After calling
    set_world_poses, the USD prim's xformOps should still contain the
    original (stale) values.
    """
    bundle = view_factory(1, device)
    view = bundle.view

    # Capture the original USD world position BEFORE any Fabric write
    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(view.prim_paths[0])
    xform_cache = UsdGeom.XformCache()
    usd_tf_before = xform_cache.GetLocalToWorldTransform(prim)
    usd_t_before = usd_tf_before.ExtractTranslation()
    orig_usd_pos = torch.tensor([float(usd_t_before[0]), float(usd_t_before[1]), float(usd_t_before[2])])

    # Write to Fabric — move to (99, 99, 99)
    new_pos = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_pos, 99.0, 99.0, 99.0], device=device)
    view.set_world_poses(positions=new_pos)

    # Verify Fabric has the new position
    fab_pos, _ = view.get_world_poses()
    pos_torch = wp.to_torch(fab_pos)
    assert torch.allclose(pos_torch, torch.tensor([[99.0, 99.0, 99.0]], device=device), atol=0.1), (
        f"Fabric should have new position, got {pos_torch}"
    )

    # Verify USD still has the ORIGINAL position (no writeback)
    xform_cache_after = UsdGeom.XformCache()
    usd_tf_after = xform_cache_after.GetLocalToWorldTransform(prim)
    usd_t_after = usd_tf_after.ExtractTranslation()
    usd_pos_after = torch.tensor([float(usd_t_after[0]), float(usd_t_after[1]), float(usd_t_after[2])])
    assert torch.allclose(usd_pos_after, orig_usd_pos, atol=0.1), (
        f"USD should still have original position {orig_usd_pos}, but got {usd_pos_after}. "
        f"sync_usd_on_fabric_write may not have been fully removed."
    )


@pytest.mark.parametrize("device", ["cuda:0"])
def test_prepare_for_reuse_detects_topology_change(device, view_factory):
    """Verify PrepareForReuse() is callable and returns a bool.

    When no topology change has occurred, it should return False.
    """
    bundle = view_factory(1, device)
    view = bundle.view
    view.get_world_poses()  # trigger Fabric init

    assert view._fabric_selection is not None, "Fabric selection not initialized"
    result = view._fabric_selection.PrepareForReuse()
    assert isinstance(result, bool), f"PrepareForReuse should return bool, got {type(result)}"
    assert not result, "PrepareForReuse should return False when no topology change"


@pytest.mark.parametrize("device", ["cuda:0"])
def test_get_scales_fabric_path(device, view_factory):
    """Exercise the Fabric-native get_scales path."""
    bundle = view_factory(num_envs=1, device=device)
    view = bundle.view

    # Trigger Fabric init
    view.get_world_poses()

    scales = view.get_scales()
    scales_t = wp.to_torch(scales)
    # Default scale should be (1, 1, 1)
    expected = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32, device=device)
    torch.testing.assert_close(scales_t, expected, atol=1e-4, rtol=0)
