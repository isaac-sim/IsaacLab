# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX Fabric backend tests for FrameView.

Imports the shared contract tests and provides the Fabric-specific
``view_factory`` fixture (SimulationContext with use_fabric=True,
Camera prim type for Fabric SelectPrims compatibility).
"""

import os
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


def _skip_if_unavailable(device: str):
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    idx = int(device.split(":")[1]) if ":" in device else 0
    n = torch.cuda.device_count()
    if idx >= n:
        # Always skip rather than fail: the dedicated multi-GPU workflow does its own
        # pre-flight ``torch.cuda.device_count() >= 2`` check before invoking pytest, so
        # a misconfigured multi-GPU runner is already caught there.  Failing here would
        # only break the standard single-GPU CI runners that legitimately can't run
        # ``cuda:1+`` tests.
        pytest.skip(f"{device} not available (device_count={n}) -- multi-GPU test skipped")


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
# (No override needed -- the shared test_set_world_updates_local from
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

    # Write to Fabric -- move to (99, 99, 99)
    new_pos = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_pos, 99.0, 99.0, 99.0], device=device)
    view.set_world_poses(positions=new_pos)

    # Verify Fabric has the new position
    fab_pos, _ = view.get_world_poses()
    pos_torch = torch.as_tensor(fab_pos, device=device)
    assert torch.allclose(pos_torch, torch.tensor([[99.0, 99.0, 99.0]], device=device), atol=0.1), (
        f"Fabric should have new position, got {pos_torch}"
    )

    # Verify USD still has the ORIGINAL position (no writeback). Equality, not
    # approximate -- USD should literally not have moved, so any drift would
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

    # First write -- initializes Fabric.
    initial = wp.zeros((2, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=2, inputs=[initial, 1.0, 2.0, 3.0], device=device)
    view.set_world_poses(positions=initial)

    # Simulate topology change: force rebuild of the selection's indexed arrays.
    view._refresh_if_needed()

    # Trigger another write through the rebuilt arrays.
    new = wp.zeros((2, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=2, inputs=[new, 4.0, 5.0, 6.0], device=device)
    view.set_world_poses(positions=new)

    ret_pos, _ = view.get_world_poses()
    pos_torch = torch.as_tensor(ret_pos, device=device)
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

    assert view._sel is not None, "selection not initialized"
    result = view._sel.PrepareForReuse()
    assert isinstance(result, bool), f"PrepareForReuse should return bool, got {type(result)}"
    assert not result, "PrepareForReuse should return False when no topology change"


def _read_fabric_world_matrix_translation(view, prim_index=0):
    """Read cached Fabric worldMatrix directly, without FrameView getter sync."""
    rt_prim = view._stage.GetPrimAtPath(view.prim_paths[prim_index])
    world_attr = rt_prim.GetAttribute(view._WORLD_MATRIX_NAME)
    matrix = world_attr.Get()
    translation = matrix.ExtractTranslation()
    return torch.tensor(
        [[float(translation[0]), float(translation[1]), float(translation[2])]],
        dtype=torch.float32,
        device=view._device,
    )


def _read_fabric_world_matrix_scale(view, prim_index=0):
    """Read cached Fabric worldMatrix scale directly, without FrameView getter sync."""
    import usdrt  # noqa: PLC0415

    rt_prim = view._stage.GetPrimAtPath(view.prim_paths[prim_index])
    world_attr = rt_prim.GetAttribute(view._WORLD_MATRIX_NAME)
    matrix = world_attr.Get()
    scale = usdrt.Gf.Transform(matrix).GetScale()
    return torch.tensor(
        [[float(scale[0]), float(scale[1]), float(scale[2])]],
        dtype=torch.float32,
        device=view._device,
    )


@pytest.mark.parametrize("device", ["cuda:0"])
def test_set_local_poses_updates_renderer_facing_fabric_world_matrix(device, view_factory):
    """Local pose writes must update cached Fabric worldMatrix immediately.

    The FSD renderer reads Fabric's cached ``omni:fabric:worldMatrix`` directly;
    it does not call ``FrameView.get_world_poses()`` to trigger Isaac Lab's lazy
    local→world sync.  This test intentionally reads the Fabric attribute
    directly after ``set_local_poses`` and before any world getter call.
    """
    bundle = view_factory(num_envs=1, device=device)
    view = bundle.view

    # Initialize Fabric and clear the initial dirty state.
    view.get_world_poses()
    assert view._dirty.name == "NONE"

    new_local_pos = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_local_pos, 1.0, 2.0, 3.0], device=device)
    new_local_ori = wp.from_torch(torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device))

    view.set_local_poses(translations=new_local_pos, orientations=new_local_ori)

    # Parent is at (0, 0, 1), so renderer-facing cached worldMatrix should
    # already contain world translation (1, 2, 4) without get_world_poses().
    expected_world = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32, device=device)
    cached_world = _read_fabric_world_matrix_translation(view)
    torch.testing.assert_close(cached_world, expected_world, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cuda:0"])
def test_set_local_scales_updates_renderer_facing_fabric_world_matrix(device, view_factory):
    """Local scale writes must update cached Fabric worldMatrix immediately.

    This is the scale analogue of the local-pose renderer/FSD contract: read
    cached ``omni:fabric:worldMatrix`` directly after ``set_local_scales`` and
    before any FrameView world getter can repair stale state.
    """
    bundle = view_factory(num_envs=1, device=device)
    view = bundle.view

    # Initialize Fabric and clear the initial dirty state.
    view.get_world_poses()
    assert view._dirty.name == "NONE"

    new_scales = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_scales, 2.0, 3.0, 4.0], device=device)

    view.set_local_scales(new_scales)

    expected_scale = torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float32, device=device)
    cached_scale = _read_fabric_world_matrix_scale(view)
    torch.testing.assert_close(cached_scale, expected_scale, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cuda:0"])
def test_change_block_batches_local_world_matrix_update(device, view_factory, monkeypatch):
    """Local pose+scale writes inside change_block flush worldMatrix once on exit."""
    bundle = view_factory(num_envs=1, device=device)
    view = bundle.view

    view.get_world_poses()
    assert view._dirty.name == "NONE"

    calls = 0
    original_recompute = view._recompute_world_from_local

    def counted_recompute():
        nonlocal calls
        calls += 1
        original_recompute()

    monkeypatch.setattr(view, "_recompute_world_from_local", counted_recompute)

    new_local_pos = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_local_pos, 1.0, 2.0, 3.0], device=device)
    new_local_ori = wp.from_torch(torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device))
    new_scales = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_scales, 2.0, 3.0, 4.0], device=device)

    with view.change_block():
        view.set_local_poses(translations=new_local_pos, orientations=new_local_ori)
        assert view._dirty.name == "WORLD"
        assert calls == 0

        view.set_local_scales(new_scales)
        assert view._dirty.name == "WORLD"
        assert calls == 0

    assert calls == 1
    assert view._dirty.name == "NONE"

    expected_world = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32, device=device)
    cached_world = _read_fabric_world_matrix_translation(view)
    torch.testing.assert_close(cached_world, expected_world, atol=1e-5, rtol=0)

    expected_scale = torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float32, device=device)
    cached_scale = _read_fabric_world_matrix_scale(view)
    torch.testing.assert_close(cached_scale, expected_scale, atol=1e-5, rtol=0)


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
    torch.testing.assert_close(torch.as_tensor(world_pos, device=device), expected, atol=1e-4, rtol=0)

    # Verify get_local_poses returns the local offset
    local_pos, _ = view.get_local_poses()
    expected_local = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, device=device)
    torch.testing.assert_close(torch.as_tensor(local_pos, device=device), expected_local, atol=1e-4, rtol=0)


@pytest.mark.parametrize("device", ["cuda:0"])
def test_get_scales_fabric_path(device, view_factory):
    """Exercise the Fabric-native get_world_scales path."""
    bundle = view_factory(num_envs=1, device=device)
    view = bundle.view

    # Trigger lazy `_initialize_fabric()` so the get_world_scales call below uses Fabric.
    view.get_world_poses()

    scales = view.get_world_scales()
    scales_t = scales.torch
    # Default scale should be (1, 1, 1)
    expected = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32, device=device)
    torch.testing.assert_close(scales_t, expected, atol=1e-4, rtol=0)


@pytest.mark.parametrize("device", ["cuda:0"])
def test_local_scales_roundtrip(device, view_factory):
    """set_local_scales -> get_local_scales roundtrip via localMatrix."""
    bundle = view_factory(num_envs=2, device=device)
    view = bundle.view

    # Force Fabric init
    view.get_world_poses()

    new_scales = wp.zeros((2, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=2, inputs=[new_scales, 2.0, 3.0, 4.0], device=device)
    view.set_local_scales(new_scales)

    # Local writes eagerly update renderer-facing world matrices outside change_block().
    assert view._dirty.name == "NONE"

    ret_scales = view.get_local_scales()
    scales_torch = ret_scales.torch
    expected = torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]], device=device)
    torch.testing.assert_close(scales_torch, expected, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cuda:0"])
def test_world_scales_roundtrip(device, view_factory):
    """set_world_scales -> get_world_scales roundtrip via worldMatrix."""
    bundle = view_factory(num_envs=2, device=device)
    view = bundle.view

    # Force Fabric init
    view.get_world_poses()

    new_scales = wp.zeros((2, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=2, inputs=[new_scales, 5.0, 6.0, 7.0], device=device)
    view.set_world_scales(new_scales)

    # Should have dirtied local
    assert view._dirty.name == "LOCAL"

    ret_scales = view.get_world_scales()
    scales_torch = ret_scales.torch
    expected = torch.tensor([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0]], device=device)
    torch.testing.assert_close(scales_torch, expected, atol=1e-5, rtol=0)


# ------------------------------------------------------------------
# Transpose-convention verification: world ↔ local kernels rely on the
# identity ``(A·B)ᵀ = Bᵀ·Aᵀ`` to drop explicit transposes when operating
# on Fabric's column-transposed matrix storage.  The translation-only
# parents used by the standard fixture cannot distinguish the right
# convention from the wrong one -- the rotation block is identity and
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
    must produce world translation (0, 1, 1) -- parent_pos + R · local.  If the
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
    torch.testing.assert_close(torch.as_tensor(world_pos, device=device), expected, atol=1e-5, rtol=0)


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
    torch.testing.assert_close(torch.as_tensor(local_pos, device=device), expected, atol=1e-5, rtol=0)


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
        torch.as_tensor(world_pos, device=device),
        torch.tensor([[2.0, 0.0, 1.0]], dtype=torch.float32, device=device),
        atol=1e-5,
        rtol=0,
    )

    scales = view.get_world_scales().torch
    torch.testing.assert_close(
        scales,
        torch.tensor([[6.0, 1.0, 1.0]], dtype=torch.float32, device=device),
        atol=1e-5,
        rtol=0,
    )


# ------------------------------------------------------------------
# Multi-view per stage: per-view dirty-flag isolation
# ------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_multi_view_per_view_dirty_isolation(device):
    """Two ``FabricFrameView`` instances on the same stage must not clear each other's
    pending local→world sync.

    Background: an earlier implementation stored the world-dirty flag at the class
    level keyed by ``stage_id``.  With two views on the same stage, view B reading
    worlds would clear the flag set by view A's ``set_local_poses``, leaving A's
    world matrices silently stale because A's per-view sync kernel never fired.

    This test sets up two views over disjoint child prims (under different parent
    sub-trees of the same stage), interleaves their writes and reads, and verifies:

    * view A's ``set_local_poses`` only dirties view A
    * view B's ``get_world_poses`` does not clear view A's flag
    * after both views' world reads, each one's worlds reflect its own latest local
    * neither view's reads/writes corrupt the other view's poses
    """
    _skip_if_unavailable(device)
    stage = sim_utils.get_current_stage()

    # Two disjoint sub-trees under the same stage.  Use different parent names so
    # the regex patterns for the two views don't accidentally overlap.
    sim_utils.create_prim("/World/EnvA_0", "Xform", translation=(0.0, 0.0, 1.0), stage=stage)
    sim_utils.create_prim("/World/EnvA_0/ChildA", "Camera", translation=(0.1, 0.0, 0.0), stage=stage)
    sim_utils.create_prim("/World/EnvB_0", "Xform", translation=(0.0, 0.0, 2.0), stage=stage)
    sim_utils.create_prim("/World/EnvB_0/ChildB", "Camera", translation=(0.2, 0.0, 0.0), stage=stage)

    sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=device, use_fabric=True))
    view_a = FrameView("/World/EnvA_.*/ChildA", device=device)
    view_b = FrameView("/World/EnvB_.*/ChildB", device=device)

    # Initial reads -- triggers Fabric init + the seed-time ``_dirty = WORLD``
    # path on both views, then clears it.
    expected_a0 = torch.tensor([[0.1, 0.0, 1.0]], dtype=torch.float32, device=device)
    expected_b0 = torch.tensor([[0.2, 0.0, 2.0]], dtype=torch.float32, device=device)
    torch.testing.assert_close(
        torch.as_tensor(view_a.get_world_poses()[0], device=device), expected_a0, atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        torch.as_tensor(view_b.get_world_poses()[0], device=device), expected_b0, atol=1e-5, rtol=0
    )
    assert view_a._dirty.name == "NONE"
    assert view_b._dirty.name == "NONE"

    # Write a new local pose on view A only.
    new_local_a = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_local_a, 1.0, 0.0, 0.0], device=device)
    identity_quat = wp.from_torch(torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device))
    view_a.set_local_poses(translations=new_local_a, orientations=identity_quat)

    # Local writes eagerly update renderer-facing world matrices outside change_block(),
    # and they must not dirty another view on the same stage.
    assert view_a._dirty.name == "NONE", "set_local_poses should flush its own view outside change_block"
    assert view_b._dirty.name == "NONE", "set_local_poses on view A must not dirty view B"

    # Read worlds from view B FIRST.  This must not affect view A's already-flushed
    # world matrices.
    torch.testing.assert_close(
        torch.as_tensor(view_b.get_world_poses()[0], device=device), expected_b0, atol=1e-5, rtol=0
    )
    assert view_b._dirty.name == "NONE"
    assert view_a._dirty.name == "NONE"

    # Now read view A's worlds -- world already reflects the new local.
    expected_a1 = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    torch.testing.assert_close(
        torch.as_tensor(view_a.get_world_poses()[0], device=device), expected_a1, atol=1e-5, rtol=0
    )
    assert view_a._dirty.name == "NONE"

    # Symmetric pass: write on B, ensure A is undisturbed.
    new_local_b = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_local_b, 3.0, 0.0, 0.0], device=device)
    view_b.set_local_poses(translations=new_local_b, orientations=identity_quat)
    assert view_a._dirty.name == "NONE"
    assert view_b._dirty.name == "NONE"

    # A's worlds must still read back the post-set-local value from above; no
    # cross-view stomp on the world matrix.
    torch.testing.assert_close(
        torch.as_tensor(view_a.get_world_poses()[0], device=device), expected_a1, atol=1e-5, rtol=0
    )
    expected_b1 = torch.tensor([[3.0, 0.0, 2.0]], dtype=torch.float32, device=device)
    torch.testing.assert_close(
        torch.as_tensor(view_b.get_world_poses()[0], device=device), expected_b1, atol=1e-5, rtol=0
    )
    assert view_a._dirty.name == "NONE"
    assert view_b._dirty.name == "NONE"


# ------------------------------------------------------------------
# Multi-GPU tests (cuda:1) -- skipped automatically on single-GPU workstations
# ------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("ISAACLAB_TEST_MULTI_GPU"),
    reason="Multi-GPU tests disabled (set ISAACLAB_TEST_MULTI_GPU=1 to enable)",
)
@pytest.mark.parametrize("device", ["cuda:1"])
def test_fabric_cuda1_world_pose_roundtrip(device, view_factory):
    """set_world_poses -> get_world_poses roundtrip works on cuda:1.

    Verifies that FabricFrameView operates correctly on a non-primary CUDA
    device without falling back to the USD path.
    """
    bundle = view_factory(2, device)
    view = bundle.view

    new_pos = wp.zeros((2, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=2, inputs=[new_pos, 10.0, 20.0, 30.0], device=device)
    view.set_world_poses(positions=new_pos)

    ret_pos, _ = view.get_world_poses()
    pos_torch = torch.as_tensor(ret_pos, device=device)
    expected = torch.tensor([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]], device=device)
    assert torch.allclose(pos_torch, expected, atol=1e-7), f"Roundtrip failed on {device}: {pos_torch}"


@pytest.mark.skipif(
    not os.environ.get("ISAACLAB_TEST_MULTI_GPU"),
    reason="Multi-GPU tests disabled (set ISAACLAB_TEST_MULTI_GPU=1 to enable)",
)
@pytest.mark.parametrize("device", ["cuda:1"])
def test_fabric_cuda1_no_usd_writeback(device, view_factory):
    """set_world_poses on cuda:1 does not write back to USD.

    Mirrors test_fabric_set_world_does_not_write_back_to_usd for the cuda:1
    device to confirm the no-writeback invariant holds across GPU indices.
    """
    bundle = view_factory(1, device)
    view = bundle.view

    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(view.prim_paths[0])
    xform_cache = UsdGeom.XformCache()
    t_before = xform_cache.GetLocalToWorldTransform(prim).ExtractTranslation()
    orig_usd_pos = torch.tensor([float(t_before[0]), float(t_before[1]), float(t_before[2])])

    new_pos = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_pos, 99.0, 99.0, 99.0], device=device)
    view.set_world_poses(positions=new_pos)

    # USD must not have moved at all -- equality, not approximate.
    t_after = UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation()
    usd_pos_after = torch.tensor([float(t_after[0]), float(t_after[1]), float(t_after[2])])
    assert torch.allclose(usd_pos_after, orig_usd_pos, atol=0.0), (
        f"USD wrote back on {device}: expected {orig_usd_pos}, got {usd_pos_after}"
    )


@pytest.mark.skipif(
    not os.environ.get("ISAACLAB_TEST_MULTI_GPU"),
    reason="Multi-GPU tests disabled (set ISAACLAB_TEST_MULTI_GPU=1 to enable)",
)
@pytest.mark.parametrize("device", ["cuda:1"])
def test_fabric_cuda1_scales_roundtrip(device, view_factory):
    """set_world_scales -> get_world_scales roundtrip works on cuda:1.

    Both write paths (``set_world_poses`` and ``set_world_scales``) call
    ``_prepare_for_reuse`` and launch on ``self._device``; this test covers
    the scales path on the non-primary CUDA device.
    """
    bundle = view_factory(2, device)
    view = bundle.view

    new_scales = wp.zeros((2, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=2, inputs=[new_scales, 2.0, 3.0, 4.0], device=device)
    view.set_world_scales(new_scales)

    ret_scales = view.get_world_scales()
    scales_torch = ret_scales.torch
    expected = torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]], device=device)
    assert torch.allclose(scales_torch, expected, atol=1e-7), f"Scales roundtrip failed on {device}: {scales_torch}"


# ------------------------------------------------------------------
# Interleaved set_world_poses / set_local_poses tests
# ------------------------------------------------------------------


def _build_two_child_view(device: str) -> "FrameView":
    """Build a 2-env FabricFrameView with rotated parent for interleave tests.

    Parent at (0, 0, 1) rotated 90° around Z.  Two child prims at identity local.
    """
    _skip_if_unavailable(device)
    stage = sim_utils.get_current_stage()
    for i in range(2):
        sim_utils.create_prim(
            f"/World/Parent_{i}",
            "Xform",
            translation=_ROTATED_PARENT_POS,
            orientation=_ROTATED_PARENT_QUAT_XYZW,
            stage=stage,
        )
        sim_utils.create_prim(f"/World/Parent_{i}/Child", "Camera", translation=(0.0, 0.0, 0.0), stage=stage)
    sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=device, use_fabric=True))
    view = FrameView("/World/Parent_.*/Child", device=device)
    view.get_world_poses()  # force init
    return view


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_interleaved_set_world_then_set_local_partial_indices(device):
    """set_world_poses on index 0, then set_local_poses on index 1 -- both must be correct.

    This exercises the dirty-flag flush: after set_world_poses marks _dirty == LOCAL,
    set_local_poses must flush stale locals before writing index 1, ensuring index 0's
    local is correctly derived from its new world pose.
    """
    view = _build_two_child_view(device)

    # Step 1: set world pose on index 0 only
    new_world_pos = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_world_pos, 5.0, 0.0, 2.0], device=device)
    idx0 = wp.from_torch(torch.tensor([0], dtype=torch.int32, device=device))
    view.set_world_poses(positions=new_world_pos, indices=idx0)

    # Step 2: set local pose on index 1 only
    new_local_pos = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_local_pos, 1.0, 0.0, 0.0], device=device)
    identity_quat = wp.from_torch(torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device))
    idx1 = wp.from_torch(torch.tensor([1], dtype=torch.int32, device=device))
    view.set_local_poses(translations=new_local_pos, orientations=identity_quat, indices=idx1)

    # Verify index 0's world pose is still (5, 0, 2)
    world_pos, _ = view.get_world_poses(indices=idx0)
    torch.testing.assert_close(
        torch.as_tensor(world_pos, device=device),
        torch.tensor([[5.0, 0.0, 2.0]], dtype=torch.float32, device=device),
        atol=1e-5,
        rtol=0,
    )

    # Verify index 0's local pose (derived from world):
    # local = Rᵀ · (child_world_pos - parent_pos) = Rz(-90)·(5, 0, 1) = (0, -5, 1)
    local_pos_0, _ = view.get_local_poses(indices=idx0)
    torch.testing.assert_close(
        torch.as_tensor(local_pos_0, device=device),
        torch.tensor([[0.0, -5.0, 1.0]], dtype=torch.float32, device=device),
        atol=1e-5,
        rtol=0,
    )

    # Verify index 1's world pose (derived from local):
    # world = parent_world * local = Rz(90)·(1, 0, 0) + parent_pos = (0, 1, 0) + (0, 0, 1) = (0, 1, 1)
    world_pos_1, _ = view.get_world_poses(indices=idx1)
    torch.testing.assert_close(
        torch.as_tensor(world_pos_1, device=device),
        torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32, device=device),
        atol=1e-5,
        rtol=0,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_interleaved_set_local_then_set_world_partial_indices(device):
    """set_local_poses on index 0, then set_world_poses on index 1 -- both must be correct.

    The reverse direction of the above: after set_local_poses marks _dirty = WORLD,
    set_world_poses must flush stale worlds before writing index 1.
    """
    view = _build_two_child_view(device)

    # Step 1: set local pose on index 0 only
    new_local_pos = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_local_pos, 2.0, 3.0, 0.0], device=device)
    identity_quat = wp.from_torch(torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device))
    idx0 = wp.from_torch(torch.tensor([0], dtype=torch.int32, device=device))
    view.set_local_poses(translations=new_local_pos, orientations=identity_quat, indices=idx0)

    # Step 2: set world pose on index 1 only
    new_world_pos = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_world_pos, 10.0, 20.0, 30.0], device=device)
    idx1 = wp.from_torch(torch.tensor([1], dtype=torch.int32, device=device))
    view.set_world_poses(positions=new_world_pos, indices=idx1)

    # Verify index 0's world pose (derived from local):
    # world = Rz(90)·(2, 3, 0) + (0, 0, 1) = (-3, 2, 0) + (0, 0, 1) = (-3, 2, 1)
    world_pos_0, _ = view.get_world_poses(indices=idx0)
    torch.testing.assert_close(
        torch.as_tensor(world_pos_0, device=device),
        torch.tensor([[-3.0, 2.0, 1.0]], dtype=torch.float32, device=device),
        atol=1e-5,
        rtol=0,
    )

    # Verify index 1's world pose is still (10, 20, 30)
    world_pos_1, _ = view.get_world_poses(indices=idx1)
    torch.testing.assert_close(
        torch.as_tensor(world_pos_1, device=device),
        torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32, device=device),
        atol=1e-5,
        rtol=0,
    )

    # Verify index 1's local (derived from world):
    # local = Rᵀ·(world - parent) = Rz(-90)·(10, 20, 29) = (20, -10, 29)
    local_pos_1, _ = view.get_local_poses(indices=idx1)
    torch.testing.assert_close(
        torch.as_tensor(local_pos_1, device=device),
        torch.tensor([[20.0, -10.0, 29.0]], dtype=torch.float32, device=device),
        atol=1e-5,
        rtol=0,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_interleaved_set_emits_warning(device, caplog):
    """Interleaving set_world_poses and set_local_poses logs a one-time warning."""
    view = _build_two_child_view(device)

    # First set_world_poses -- no warning (first user setter)
    new_world = wp.zeros((2, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=2, inputs=[new_world, 1.0, 2.0, 3.0], device=device)
    view.set_world_poses(positions=new_world)

    # Now set_local_poses -- should trigger warning about interleaving
    new_local = wp.zeros((2, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=2, inputs=[new_local, 0.0, 0.0, 0.0], device=device)
    identity_quat = wp.from_torch(torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=torch.float32, device=device))

    with caplog.at_level("WARNING", logger="isaaclab_physx.sim.views.fabric_frame_view"):
        caplog.clear()
        view.set_local_poses(translations=new_local, orientations=identity_quat)

    assert any("interleaving" in r.message.lower() for r in caplog.records), (
        f"Expected interleave warning, got: {[r.message for r in caplog.records]}"
    )

    # Second interleave -- warning should NOT repeat (one-time only)
    caplog.clear()
    view.set_world_poses(positions=new_world)
    assert not any("interleaving" in r.message.lower() for r in caplog.records), (
        "Warning should only fire once per view instance"
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_no_warning_when_using_single_setter(device, caplog):
    """Calling only set_world_poses (or only set_local_poses) should never warn."""
    view = _build_two_child_view(device)

    new_world = wp.zeros((2, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=2, inputs=[new_world, 1.0, 2.0, 3.0], device=device)

    with caplog.at_level("WARNING", logger="isaaclab_physx.sim.views.fabric_frame_view"):
        caplog.clear()
        view.set_world_poses(positions=new_world)
        view.set_world_poses(positions=new_world)
        view.set_world_poses(positions=new_world)

    assert not any("interleaving" in r.message.lower() for r in caplog.records), (
        f"Unexpected interleave warning with single setter: {[r.message for r in caplog.records]}"
    )
