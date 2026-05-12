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
from frame_view_contract_utils import CHILD_OFFSET, ViewBundle  # noqa: E402
from isaaclab_physx.sim.views import FabricFrameView as FrameView  # noqa: E402

from pxr import Gf, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

pytestmark = pytest.mark.isaacsim_ci
PARENT_POS = (0.0, 0.0, 1.0)


@pytest.fixture(autouse=True)
def test_setup_teardown():
    sim_utils.create_new_stage()
    sim_utils.update_stage()
    yield
    sim_utils.clear_stage()
    sim_utils.SimulationContext.clear_instance()
    # Each test creates a fresh stage; drop cached IFabricHierarchy handles so
    # the next test does not reuse a handle attached to the disposed stage.
    FrameView.clear_static_caches()


def _skip_if_unavailable(device: str):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")


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
# Override: ensure the shared contract test runs without xfail now that
# get_local_poses computes local from Fabric world matrices.
# ------------------------------------------------------------------
# (No override needed — the shared test_set_world_updates_local from
#  frame_view_contract_utils is imported via wildcard and will run as-is.)


# ------------------------------------------------------------------
# Fabric-specific tests (not in shared contract)
# ------------------------------------------------------------------


@wp.kernel
def _fill_position(out: wp.array(dtype=wp.float32, ndim=2), x: float, y: float, z: float):
    i = wp.tid()
    out[i, 0] = wp.float32(x)
    out[i, 1] = wp.float32(y)
    out[i, 2] = wp.float32(z)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
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

    # Verify USD still has the ORIGINAL position (no writeback). Equality, not
    # approximate — USD should literally not have moved, so any drift would
    # indicate a residual writeback path.
    xform_cache_after = UsdGeom.XformCache()
    usd_tf_after = xform_cache_after.GetLocalToWorldTransform(prim)
    usd_t_after = usd_tf_after.ExtractTranslation()
    usd_pos_after = torch.tensor([float(usd_t_after[0]), float(usd_t_after[1]), float(usd_t_after[2])])
    assert torch.allclose(usd_pos_after, orig_usd_pos, atol=0.0), (
        f"USD should still have original position {orig_usd_pos}, but got {usd_pos_after}. "
        f"sync_usd_on_fabric_write may not have been fully removed."
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fabric_rebuild_after_topology_change(device, view_factory):
    """A simulated topology change rebuilds the indexed fabric arrays and leaves
    the view in a state where subsequent writes/reads still produce correct data.

    Real ``PrimSelection.PrepareForReuse`` reports topology change only when Fabric
    reallocates internally, which is hard to provoke from a unit test.  Instead we
    invoke :meth:`FabricFrameView._compute_fabric_indices` and rebuild the indexed
    arrays manually, mimicking what ``_get_*_array`` would do on a real topology
    event, then verify a roundtrip still works.
    """
    bundle = view_factory(2, device)
    view = bundle.view

    # First write — initializes Fabric.
    initial = wp.zeros((2, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=2, inputs=[initial, 1.0, 2.0, 3.0], device=device)
    view.set_world_poses(positions=initial)

    # Simulate topology change: recompute per-selection fabric indices and rebuild
    # every indexed array, mirroring the lazy paths in the ``_get_*_array`` accessors.
    view._rebuild_trans_ro_arrays()
    view._world_rw_fabric_indices = view._compute_fabric_indices(view._world_sel_rw)
    view._world_ifa_rw = view._build_indexed_array(
        view._world_sel_rw, view._WORLD_MATRIX_NAME, view._world_rw_fabric_indices
    )
    view._local_rw_fabric_indices = view._compute_fabric_indices(view._local_sel_rw)
    view._local_ifa_rw = view._build_indexed_array(
        view._local_sel_rw, view._LOCAL_MATRIX_NAME, view._local_rw_fabric_indices
    )

    # Trigger another write through the rebuilt arrays.
    new = wp.zeros((2, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=2, inputs=[new, 4.0, 5.0, 6.0], device=device)
    view.set_world_poses(positions=new)

    ret_pos, _ = view.get_world_poses()
    pos_torch = wp.to_torch(ret_pos)
    expected = torch.tensor([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]], device=device)
    # 1e-5 ≈ 20 ULP at magnitudes ~4-6; absorbs float32 SRT compose/decompose drift.
    assert torch.allclose(pos_torch, expected, atol=1e-5), f"Read after rebuild failed on {device}: {pos_torch}"


@pytest.mark.parametrize("device", ["cuda:0"])
def test_prepare_for_reuse_detects_topology_change(device, view_factory):
    """Each persistent ``PrimSelection`` exposes ``PrepareForReuse`` and returns a
    bool.  When the underlying Fabric topology is unchanged it returns False.
    """
    bundle = view_factory(1, device)
    view = bundle.view
    view.get_world_poses()  # trigger Fabric init

    assert view._trans_sel_ro is not None, "trans_sel_ro selection not initialized"
    for selection in (view._trans_sel_ro, view._world_sel_rw, view._local_sel_rw):
        result = selection.PrepareForReuse()
        assert isinstance(result, bool), f"PrepareForReuse should return bool, got {type(result)}"
        assert not result, "PrepareForReuse should return False when no topology change"


@pytest.mark.parametrize("device", ["cuda:0"])
def test_set_local_via_fabric_path(device, view_factory):
    """Exercise the Fabric-native set_local_poses path.

    Ensures set_local_poses computes child_world = parent_world * local
    entirely within Fabric (not falling back to USD) by first triggering
    the Fabric sync via get_world_poses.
    """
    bundle = view_factory(num_envs=1, device=device)
    view = bundle.view

    # Trigger lazy `_initialize_fabric()` so subsequent calls take the Fabric path.
    view.get_world_poses()

    # Now set_local_poses should take the Fabric path
    new_local_pos = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_local_pos, 1.0, 2.0, 3.0], device=device)
    ori = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    new_local_ori = wp.from_torch(ori)

    view.set_local_poses(translations=new_local_pos, orientations=new_local_ori)

    # Verify: world = parent(0,0,1) + local(1,2,3) = (1,2,4)
    world_pos, _ = view.get_world_poses()
    expected = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32, device=device)
    torch.testing.assert_close(world_pos.torch, expected, atol=1e-4, rtol=0)

    # Verify get_local_poses returns the local offset
    local_pos, _ = view.get_local_poses()
    expected_local = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, device=device)
    torch.testing.assert_close(local_pos.torch, expected_local, atol=1e-4, rtol=0)


@pytest.mark.parametrize("device", ["cuda:0"])
def test_get_scales_fabric_path(device, view_factory):
    """Exercise the Fabric-native get_scales path."""
    bundle = view_factory(num_envs=1, device=device)
    view = bundle.view

    # Trigger lazy `_initialize_fabric()` so the get_scales call below uses Fabric.
    view.get_world_poses()

    scales = view.get_scales()
    scales_t = wp.to_torch(scales)
    # Default scale should be (1, 1, 1)
    expected = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32, device=device)
    torch.testing.assert_close(scales_t, expected, atol=1e-4, rtol=0)


# ------------------------------------------------------------------
# Transpose-convention verification: world ↔ local kernels rely on the
# identity ``(A·B)ᵀ = Bᵀ·Aᵀ`` to drop explicit transposes when operating
# on Fabric's column-transposed matrix storage.  The translation-only
# parents used by the standard fixture cannot distinguish the right
# convention from the wrong one — the rotation block is identity and
# equals its own transpose.  These tests use a parent rotated 90° around
# Z so that an incorrect storage convention would produce a clearly
# wrong child pose.
# ------------------------------------------------------------------


# Parent at (0, 0, 1) rotated +90° around Z (so the parent X axis points
# along world +Y).  Quaternion components in (x, y, z, w) order.
_ROTATED_PARENT_POS = (0.0, 0.0, 1.0)
_ROTATED_PARENT_QUAT_XYZW = (0.0, 0.0, 0.70710678, 0.70710678)


def _build_rotated_parent_view(device: str) -> "FrameView":
    """Build a 1-env FabricFrameView whose parent is rotated 90° around Z."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim(
        "/World/Parent_0",
        "Xform",
        translation=_ROTATED_PARENT_POS,
        orientation=_ROTATED_PARENT_QUAT_XYZW,
        stage=stage,
    )
    sim_utils.create_prim("/World/Parent_0/Child", "Camera", translation=(0.0, 0.0, 0.0), stage=stage)
    sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=device, use_fabric=True))
    view = FrameView("/World/Parent_.*/Child", device=device)
    view.get_world_poses()  # force Fabric init and USD→Fabric seed
    return view


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_set_local_then_get_world_with_rotated_parent(device):
    """Verify ``update_indexed_world_matrix_from_local`` under non-identity parent rotation.

    With parent rotated +90° around Z, a child local translation of (1, 0, 0)
    must produce world translation (0, 1, 1) — parent_pos + R · local.  If the
    transpose convention in the kernel were wrong, the rotation would flip
    direction and the world position would land at (0, -1, 1) instead.
    """
    _skip_if_unavailable(device)
    view = _build_rotated_parent_view(device)

    new_local = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_local, 1.0, 0.0, 0.0], device=device)
    identity_quat = wp.from_torch(torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device))
    view.set_local_poses(translations=new_local, orientations=identity_quat)

    world_pos, _ = view.get_world_poses()
    expected = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32, device=device)
    torch.testing.assert_close(world_pos.torch, expected, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_set_world_then_get_local_with_rotated_parent(device):
    """Verify ``update_indexed_local_matrix_from_world`` under non-identity parent rotation.

    With parent rotated +90° around Z and at (0, 0, 1), writing child world
    translation (5, 0, 2) must yield child local translation Rᵀ · (5, 0, 1) =
    (0, -5, 1).  A wrong transpose convention would invert the rotation in the
    wrong direction and produce (0, 5, 1) instead.
    """
    _skip_if_unavailable(device)
    view = _build_rotated_parent_view(device)

    new_world = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_world, 5.0, 0.0, 2.0], device=device)
    view.set_world_poses(positions=new_world)

    local_pos, _ = view.get_local_poses()
    expected = torch.tensor([[0.0, -5.0, 1.0]], dtype=torch.float32, device=device)
    torch.testing.assert_close(local_pos.torch, expected, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_initial_seed_with_scaled_parent(device):
    """Verify the initial USD→Fabric seed handles non-unit scales correctly.

    Sets up a parent with world scale (2, 1, 1) and a child with local scale
    (3, 1, 1) at local translation (1, 0, 0).  Expected world-space values for
    the child:

    * world scale = parent_scale * child_local_scale = (6, 1, 1)
    * world position = parent_pos + parent_scale * child_local_pos
                     = (0, 0, 1) + (2 * 1, 0, 0) = (2, 0, 1)

    If the parent's worldMatrix is seeded with a hardcoded unit scale,
    ``get_scales`` returns (3, 1, 1) instead of (6, 1, 1) and ``get_world_poses``
    returns (1, 0, 1) instead of (2, 0, 1).  If the child's localMatrix is
    seeded without scale, after ``_sync_world_from_local_if_dirty`` the world
    scale collapses to (2, 1, 1).  This test catches both regressions.
    """
    _skip_if_unavailable(device)
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/Parent_0", "Xform", translation=(0.0, 0.0, 1.0), scale=(2.0, 1.0, 1.0), stage=stage)
    sim_utils.create_prim(
        "/World/Parent_0/Child",
        "Camera",
        translation=(1.0, 0.0, 0.0),
        scale=(3.0, 1.0, 1.0),
        stage=stage,
    )
    sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=device, use_fabric=True))
    view = FrameView("/World/Parent_.*/Child", device=device)

    world_pos, _ = view.get_world_poses()
    torch.testing.assert_close(
        world_pos.torch,
        torch.tensor([[2.0, 0.0, 1.0]], dtype=torch.float32, device=device),
        atol=1e-5,
        rtol=0,
    )

    scales = wp.to_torch(view.get_scales())
    torch.testing.assert_close(
        scales,
        torch.tensor([[6.0, 1.0, 1.0]], dtype=torch.float32, device=device),
        atol=1e-5,
        rtol=0,
    )
