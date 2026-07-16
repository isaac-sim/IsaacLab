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
from isaaclab.test.utils import DeviceScope, resolve_test_sim_device, test_devices

simulation_app = AppLauncher(headless=True, device=resolve_test_sim_device()).app

import pytest  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from frame_view_contract_utils import *  # noqa: F401, F403, E402
from frame_view_contract_utils import CHILD_OFFSET, ViewBundle  # noqa: E402, F401
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


@pytest.mark.parametrize("device", test_devices())
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
    with view.xform_world_space_writer() as w:
        w.set_poses(positions=new_pos)

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


@pytest.mark.parametrize("device", test_devices())
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
    with view.xform_world_space_writer() as w:
        w.set_poses(positions=initial)

    # Simulate topology change: rebuild both selection bundles, mirroring the
    # lazy paths in the ``_refresh_active_bundle_if_needed`` accessor.
    view._rebuild_ro_arrays()
    view._rebuild_rw_arrays()

    # Trigger another write through the rebuilt arrays.
    new = wp.zeros((2, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=2, inputs=[new, 4.0, 5.0, 6.0], device=device)
    with view.xform_world_space_writer() as w:
        w.set_poses(positions=new)

    ret_pos, _ = view.get_world_poses()
    pos_torch = torch.as_tensor(ret_pos, device=device)
    expected = torch.tensor([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]], device=device)
    # 1e-5 ≈ 20 ULP at magnitudes ~4-6; absorbs float32 SRT compose/decompose drift.
    assert torch.allclose(pos_torch, expected, atol=1e-5), f"Read after rebuild failed on {device}: {pos_torch}"


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_writer_scope_exception_recovers_state(device, view_factory):
    """An exception raised inside a writer scope must still:

    1. Restore ``IFabricHierarchy.track_*_xform_changes`` to their pre-scope state.
    2. Flip the view's ``_is_rw`` flag back to ``False``.
    3. Leave ``worldMatrix`` and ``localMatrix`` mutually consistent prim-by-prim
       on whatever partial-write state Fabric currently holds (best-effort).

    Simulates an interactive-notebook scenario where a user-code exception
    fires after a ``set_poses`` call but before the scope closes.
    """
    bundle = view_factory(2, device)
    view = bundle.view
    view.get_world_poses()  # ensure Fabric is initialized

    # Snapshot pre-scope tracking state.
    h = view._fabric_hierarchy
    was_tracking_local = h.tracking_local_xform_changes
    was_tracking_world = h.tracking_world_xform_changes

    positions = wp.zeros((2, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=2, inputs=[positions, 7.0, 8.0, 9.0], device=device)

    with pytest.raises(RuntimeError, match="user-code failure"):
        with view.xform_world_space_writer() as w:
            w.set_poses(positions=positions)
            raise RuntimeError("user-code failure")

    # Tracking state restored.
    assert h.tracking_local_xform_changes == was_tracking_local
    assert h.tracking_world_xform_changes == was_tracking_world
    # _is_rw flipped back.
    assert view._is_rw is False
    # World/local mutually consistent: re-reading both spaces succeeds and the
    # world positions reflect the partial write we made before the exception.
    world_pos, _ = view.get_world_poses()
    local_pos, _ = view.get_local_poses()
    world_t = torch.as_tensor(world_pos, device=device)
    local_t = torch.as_tensor(local_pos, device=device)
    expected_world = torch.tensor([[7.0, 8.0, 9.0], [7.0, 8.0, 9.0]], device=device)
    assert torch.allclose(world_t, expected_world, atol=1e-5), f"world not updated on {device}: {world_t}"
    # Parents at PARENT_POS, so local = world - parent_world (parents identity-rotated).
    expected_local = expected_world - torch.tensor(PARENT_POS, device=device)
    assert torch.allclose(local_t, expected_local, atol=1e-5), (
        f"local not derived from partial world write on {device}: got {local_t}, expected {expected_local}"
    )

    # The view is still usable for further writer scopes after recovery.
    next_pos = wp.zeros((2, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=2, inputs=[next_pos, 10.0, 11.0, 12.0], device=device)
    with view.xform_world_space_writer() as w:
        w.set_poses(positions=next_pos)
    follow_up, _ = view.get_world_poses()
    follow_up_t = torch.as_tensor(follow_up, device=device)
    assert torch.allclose(follow_up_t, torch.tensor([[10.0, 11.0, 12.0]] * 2, device=device), atol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0"])
def test_prepare_for_reuse_detects_topology_change(device, view_factory):
    """Each persistent ``PrimSelection`` exposes ``PrepareForReuse`` and returns a
    bool.  When the underlying Fabric topology is unchanged it returns False.
    """
    bundle = view_factory(1, device)
    view = bundle.view
    view.get_world_poses()  # trigger Fabric init

    assert view._sel_ro is not None, "RO selection not initialized"
    assert view._sel_rw is not None, "RW selection not initialized"
    for selection in (view._sel_ro, view._sel_rw):
        result = selection.PrepareForReuse()
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


def _read_fabric_local_matrix_translation(view, prim_index=0):
    """Read cached Fabric localMatrix directly, without FrameView getter sync."""
    rt_prim = view._stage.GetPrimAtPath(view.prim_paths[prim_index])
    local_attr = rt_prim.GetAttribute(view._LOCAL_MATRIX_NAME)
    matrix = local_attr.Get()
    translation = matrix.ExtractTranslation()
    return torch.tensor(
        [[float(translation[0]), float(translation[1]), float(translation[2])]],
        dtype=torch.float32,
        device=view._device,
    )


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

    # Now write via the writer scope (Fabric path).
    new_local_pos = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_local_pos, 1.0, 2.0, 3.0], device=device)
    ori = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    new_local_ori = wp.from_torch(ori)

    with view.xform_local_space_writer() as w:
        w.set_poses(positions=new_local_pos, orientations=new_local_ori)

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
    """Writing scales through the local-space writer roundtrips via ``localMatrix``."""
    bundle = view_factory(num_envs=2, device=device)
    view = bundle.view

    # Force Fabric init
    view.get_world_poses()

    new_scales = wp.zeros((2, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=2, inputs=[new_scales, 2.0, 3.0, 4.0], device=device)
    with view.xform_local_space_writer() as w:
        w.set_scales(new_scales)

    ret_scales = view.get_local_scales()
    scales_torch = ret_scales.torch
    expected = torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]], device=device)
    torch.testing.assert_close(scales_torch, expected, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cuda:0"])
def test_world_scales_roundtrip(device, view_factory):
    """Writing scales through the world-space writer roundtrips via ``worldMatrix``."""
    bundle = view_factory(num_envs=2, device=device)
    view = bundle.view

    # Force Fabric init
    view.get_world_poses()

    new_scales = wp.zeros((2, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=2, inputs=[new_scales, 5.0, 6.0, 7.0], device=device)
    with view.xform_world_space_writer() as w:
        w.set_scales(new_scales)

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
    with view.xform_local_space_writer() as w:
        w.set_poses(positions=new_local, orientations=identity_quat)

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
    with view.xform_world_space_writer() as w:
        w.set_poses(positions=new_world)

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
    seeded without scale, the derived world scale collapses to (2, 1, 1).
    This test catches both regressions.
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
# Multi-view per stage: per-view single-writer isolation
# ------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_multi_view_writer_isolation(device):
    """Two ``FabricFrameView`` instances on the same stage have independent writer scopes.

    Each view's ``_active_writer`` is per-instance, so view B's read or writer
    scope must not interfere with view A's pending writes.  Verifies that
    writes through one view do not corrupt the other view's poses, and that
    each view can hold its own writer scope concurrently with reads on the
    other view.
    """
    _skip_if_unavailable(device)
    stage = sim_utils.get_current_stage()

    sim_utils.create_prim("/World/EnvA_0", "Xform", translation=(0.0, 0.0, 1.0), stage=stage)
    sim_utils.create_prim("/World/EnvA_0/ChildA", "Camera", translation=(0.1, 0.0, 0.0), stage=stage)
    sim_utils.create_prim("/World/EnvB_0", "Xform", translation=(0.0, 0.0, 2.0), stage=stage)
    sim_utils.create_prim("/World/EnvB_0/ChildB", "Camera", translation=(0.2, 0.0, 0.0), stage=stage)

    sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=device, use_fabric=True))
    view_a = FrameView("/World/EnvA_.*/ChildA", device=device)
    view_b = FrameView("/World/EnvB_.*/ChildB", device=device)

    expected_a0 = torch.tensor([[0.1, 0.0, 1.0]], dtype=torch.float32, device=device)
    expected_b0 = torch.tensor([[0.2, 0.0, 2.0]], dtype=torch.float32, device=device)
    torch.testing.assert_close(
        torch.as_tensor(view_a.get_world_poses()[0], device=device), expected_a0, atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        torch.as_tensor(view_b.get_world_poses()[0], device=device), expected_b0, atol=1e-5, rtol=0
    )

    # Write a new local pose on view A only via a writer scope.
    new_local_a = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_local_a, 1.0, 0.0, 0.0], device=device)
    identity_quat = wp.from_torch(torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device))
    with view_a.xform_local_space_writer() as w:
        w.set_poses(positions=new_local_a, orientations=identity_quat)

    # View B remains undisturbed.
    torch.testing.assert_close(
        torch.as_tensor(view_b.get_world_poses()[0], device=device), expected_b0, atol=1e-5, rtol=0
    )

    # View A's world reflects the new local.
    expected_a1 = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    torch.testing.assert_close(
        torch.as_tensor(view_a.get_world_poses()[0], device=device), expected_a1, atol=1e-5, rtol=0
    )

    # Write a new local pose on view B; view A unaffected.
    new_local_b = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_local_b, 3.0, 0.0, 0.0], device=device)
    with view_b.xform_local_space_writer() as w:
        w.set_poses(positions=new_local_b, orientations=identity_quat)
    torch.testing.assert_close(
        torch.as_tensor(view_a.get_world_poses()[0], device=device), expected_a1, atol=1e-5, rtol=0
    )
    expected_b1 = torch.tensor([[3.0, 0.0, 2.0]], dtype=torch.float32, device=device)
    torch.testing.assert_close(
        torch.as_tensor(view_b.get_world_poses()[0], device=device), expected_b1, atol=1e-5, rtol=0
    )

    # Single-active-writer is per-view: opening a writer on A leaves B free.
    with view_a.xform_world_space_writer():
        assert view_a._active_writer is not None
        assert view_b._active_writer is None
        # B can still open its own writer concurrently.
        with view_b.xform_world_space_writer():
            assert view_b._active_writer is not None
    assert view_a._active_writer is None
    assert view_b._active_writer is None


# ------------------------------------------------------------------
# Multi-GPU tests (cuda:1) -- skipped automatically on single-GPU workstations
# ------------------------------------------------------------------


@pytest.mark.parametrize("device", test_devices(DeviceScope.NON_DEFAULT_CUDA))
def test_fabric_cuda1_world_pose_roundtrip(device, view_factory):
    """set_world_poses -> get_world_poses roundtrip works on cuda:1.

    Verifies that FabricFrameView operates correctly on a non-primary CUDA
    device without falling back to the USD path.
    """
    bundle = view_factory(2, device)
    view = bundle.view

    new_pos = wp.zeros((2, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=2, inputs=[new_pos, 10.0, 20.0, 30.0], device=device)
    with view.xform_world_space_writer() as w:
        w.set_poses(positions=new_pos)

    ret_pos, _ = view.get_world_poses()
    pos_torch = torch.as_tensor(ret_pos, device=device)
    expected = torch.tensor([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]], device=device)
    assert torch.allclose(pos_torch, expected, atol=1e-7), f"Roundtrip failed on {device}: {pos_torch}"


@pytest.mark.parametrize("device", test_devices(DeviceScope.NON_DEFAULT_CUDA))
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
    with view.xform_world_space_writer() as w:
        w.set_poses(positions=new_pos)

    # USD must not have moved at all -- equality, not approximate.
    t_after = UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation()
    usd_pos_after = torch.tensor([float(t_after[0]), float(t_after[1]), float(t_after[2])])
    assert torch.allclose(usd_pos_after, orig_usd_pos, atol=0.0), (
        f"USD wrote back on {device}: expected {orig_usd_pos}, got {usd_pos_after}"
    )


@pytest.mark.parametrize("device", test_devices(DeviceScope.NON_DEFAULT_CUDA))
def test_fabric_cuda1_scales_roundtrip(device, view_factory):
    """World-space scale writes roundtrip via ``worldMatrix`` on ``cuda:1``.

    Covers the scales path on the non-primary CUDA device: the writer
    launches its compose kernel on ``self._device`` and the indexed-fabric
    arrays are rebuilt on demand from the corresponding selection.
    """
    bundle = view_factory(2, device)
    view = bundle.view

    new_scales = wp.zeros((2, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=2, inputs=[new_scales, 2.0, 3.0, 4.0], device=device)
    with view.xform_world_space_writer() as w:
        w.set_scales(new_scales)

    ret_scales = view.get_world_scales()
    scales_torch = ret_scales.torch
    expected = torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]], device=device)
    assert torch.allclose(scales_torch, expected, atol=1e-7), f"Scales roundtrip failed on {device}: {scales_torch}"


# ------------------------------------------------------------------
# Sequential writer scopes (interleaved world / local writes via two scopes)
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
def test_sequential_world_then_local_scopes_partial_indices(device):
    """A world writer scope (idx 0), then a local writer scope (idx 1).  Both correct."""
    view = _build_two_child_view(device)

    new_world_pos = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_world_pos, 5.0, 0.0, 2.0], device=device)
    idx0 = wp.from_torch(torch.tensor([0], dtype=torch.int32, device=device))
    with view.xform_world_space_writer() as w:
        w.set_poses(positions=new_world_pos, indices=idx0)

    new_local_pos = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_local_pos, 1.0, 0.0, 0.0], device=device)
    identity_quat = wp.from_torch(torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device))
    idx1 = wp.from_torch(torch.tensor([1], dtype=torch.int32, device=device))
    with view.xform_local_space_writer() as w:
        w.set_poses(positions=new_local_pos, orientations=identity_quat, indices=idx1)

    # Verify index 0's world pose is still (5, 0, 2) -- index 1's local-scope write
    # derives world from the just-written local; index 0 was outside the derived
    # set on entry but the world-scope already wrote it and the second scope
    # re-derives world from local for all prims (including idx 0).  After the
    # second scope, idx 0's local was derived from its world (= (0, -5, 1)),
    # so re-deriving world = parent * local lands back on (5, 0, 2).
    world_pos, _ = view.get_world_poses(indices=idx0)
    torch.testing.assert_close(
        torch.as_tensor(world_pos, device=device),
        torch.tensor([[5.0, 0.0, 2.0]], dtype=torch.float32, device=device),
        atol=1e-5,
        rtol=0,
    )

    # Index 0's local (derived from its world):
    # local = Rᵀ · (child_world_pos - parent_pos) = Rz(-90)·(5, 0, 1) = (0, -5, 1)
    local_pos_0, _ = view.get_local_poses(indices=idx0)
    torch.testing.assert_close(
        torch.as_tensor(local_pos_0, device=device),
        torch.tensor([[0.0, -5.0, 1.0]], dtype=torch.float32, device=device),
        atol=1e-5,
        rtol=0,
    )

    # Index 1's world (derived from local):
    # world = parent_world * local = Rz(90)·(1, 0, 0) + parent_pos = (0, 1, 0) + (0, 0, 1) = (0, 1, 1)
    world_pos_1, _ = view.get_world_poses(indices=idx1)
    torch.testing.assert_close(
        torch.as_tensor(world_pos_1, device=device),
        torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32, device=device),
        atol=1e-5,
        rtol=0,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_sequential_local_then_world_scopes_partial_indices(device):
    """A local writer scope (idx 0), then a world writer scope (idx 1).  Both correct."""
    view = _build_two_child_view(device)

    new_local_pos = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_local_pos, 2.0, 3.0, 0.0], device=device)
    identity_quat = wp.from_torch(torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device))
    idx0 = wp.from_torch(torch.tensor([0], dtype=torch.int32, device=device))
    with view.xform_local_space_writer() as w:
        w.set_poses(positions=new_local_pos, orientations=identity_quat, indices=idx0)

    new_world_pos = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_world_pos, 10.0, 20.0, 30.0], device=device)
    idx1 = wp.from_torch(torch.tensor([1], dtype=torch.int32, device=device))
    with view.xform_world_space_writer() as w:
        w.set_poses(positions=new_world_pos, indices=idx1)

    # Index 0's world (derived from local):
    # world = Rz(90)·(2, 3, 0) + (0, 0, 1) = (-3, 2, 0) + (0, 0, 1) = (-3, 2, 1)
    world_pos_0, _ = view.get_world_poses(indices=idx0)
    torch.testing.assert_close(
        torch.as_tensor(world_pos_0, device=device),
        torch.tensor([[-3.0, 2.0, 1.0]], dtype=torch.float32, device=device),
        atol=1e-5,
        rtol=0,
    )

    # Index 1's world is still (10, 20, 30).
    world_pos_1, _ = view.get_world_poses(indices=idx1)
    torch.testing.assert_close(
        torch.as_tensor(world_pos_1, device=device),
        torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32, device=device),
        atol=1e-5,
        rtol=0,
    )

    # Index 1's local (derived from world):
    # local = Rᵀ·(world - parent) = Rz(-90)·(10, 20, 29) = (20, -10, 29)
    local_pos_1, _ = view.get_local_poses(indices=idx1)
    torch.testing.assert_close(
        torch.as_tensor(local_pos_1, device=device),
        torch.tensor([[20.0, -10.0, 29.0]], dtype=torch.float32, device=device),
        atol=1e-5,
        rtol=0,
    )


# ------------------------------------------------------------------
# FrameViewSpaceWriterBase contract tests
# ------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_world_writer_writes_world_and_derives_local(device, view_factory):
    """A world writer's set_poses + set_scales updates cached Fabric world AND derives local on exit."""
    bundle = view_factory(num_envs=1, device=device)
    view = bundle.view
    view.get_world_poses()  # trigger Fabric init

    new_pos = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_pos, 1.0, 2.0, 4.0], device=device)
    new_scales = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_scales, 2.0, 3.0, 4.0], device=device)

    with view.xform_world_space_writer() as w:
        w.set_poses(positions=new_pos)
        w.set_scales(new_scales)

    # Cached world reflects the writes (parent is at (0, 0, 1) so child world pos
    # is whatever we wrote).
    expected_world = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32, device=device)
    cached_world = _read_fabric_world_matrix_translation(view)
    torch.testing.assert_close(cached_world, expected_world, atol=1e-5, rtol=0)

    expected_scale = torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float32, device=device)
    cached_scale = _read_fabric_world_matrix_scale(view)
    torch.testing.assert_close(cached_scale, expected_scale, atol=1e-5, rtol=0)

    # Local derived: with parent at (0, 0, 1) (identity rotation, unit scale),
    # local translation = world - parent = (1, 2, 3).
    expected_local = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, device=device)
    cached_local = _read_fabric_local_matrix_translation(view)
    torch.testing.assert_close(cached_local, expected_local, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_local_writer_writes_local_and_derives_world(device, view_factory):
    """A local writer's set_poses updates cached Fabric local AND derives world on exit."""
    bundle = view_factory(num_envs=1, device=device)
    view = bundle.view
    view.get_world_poses()

    new_pos = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_pos, 1.0, 2.0, 3.0], device=device)
    identity_quat = wp.from_torch(torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device))

    with view.xform_local_space_writer() as w:
        w.set_poses(positions=new_pos, orientations=identity_quat)

    expected_local = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, device=device)
    cached_local = _read_fabric_local_matrix_translation(view)
    torch.testing.assert_close(cached_local, expected_local, atol=1e-5, rtol=0)

    # World derived: with parent at (0, 0, 1), world = parent + local = (1, 2, 4).
    expected_world = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32, device=device)
    cached_world = _read_fabric_world_matrix_translation(view)
    torch.testing.assert_close(cached_world, expected_world, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cuda:0"])
def test_writer_single_derivation_per_scope(device, view_factory, monkeypatch):
    """Multiple set_* calls inside one scope produce exactly one derive-kernel launch."""
    bundle = view_factory(num_envs=1, device=device)
    view = bundle.view
    view.get_world_poses()

    calls = 0
    original = view._recompute_local_from_world_all

    def counted():
        nonlocal calls
        calls += 1
        original()

    monkeypatch.setattr(view, "_recompute_local_from_world_all", counted)

    new_pos = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_pos, 1.0, 2.0, 4.0], device=device)
    new_scales = wp.zeros((1, 3), dtype=wp.float32, device=device)
    wp.launch(kernel=_fill_position, dim=1, inputs=[new_scales, 2.0, 3.0, 4.0], device=device)

    with view.xform_world_space_writer() as w:
        w.set_poses(positions=new_pos)
        w.set_scales(new_scales)
        assert calls == 0  # no derive yet

    assert calls == 1  # exactly one derive on exit


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_writer_single_active_invariant(device, view_factory):
    """Only one writer scope may be active per view at a time."""
    bundle = view_factory(num_envs=1, device=device)
    view = bundle.view
    view.get_world_poses()

    with view.xform_world_space_writer():
        with pytest.raises(RuntimeError, match="already has an active writer"):
            view.xform_world_space_writer().__enter__()
        with pytest.raises(RuntimeError, match="already has an active writer"):
            view.xform_local_space_writer().__enter__()
    # After the outer scope exits, the lock is released and a new scope succeeds.
    with view.xform_local_space_writer():
        pass


@pytest.mark.parametrize("device", ["cuda:0"])
def test_writer_restores_hierarchy_change_tracking(device, view_factory):
    """``__exit__`` restores the prior ``track_*_xform_changes`` state (don't re-enable paused listeners)."""
    bundle = view_factory(num_envs=1, device=device)
    view = bundle.view
    view.get_world_poses()
    h = view._fabric_hierarchy

    # Case 1: pre-paused local stays paused after exit.
    h.track_local_xform_changes(False)
    assert not h.tracking_local_xform_changes
    with view.xform_world_space_writer():
        pass
    assert not h.tracking_local_xform_changes, "writer must not re-enable a pre-paused local listener"

    # Case 2: pre-enabled local stays enabled after exit.
    h.track_local_xform_changes(True)
    with view.xform_world_space_writer():
        pass
    assert h.tracking_local_xform_changes, "writer must restore the pre-enabled local listener"


@pytest.mark.parametrize("device", ["cuda:0"])
def test_writer_empty_scope_does_no_derivation(device, view_factory, monkeypatch):
    """Entering and exiting a writer scope without any ``set_*`` call must not launch the derive kernel."""
    bundle = view_factory(num_envs=1, device=device)
    view = bundle.view
    view.get_world_poses()

    calls = 0
    original = view._recompute_local_from_world_all

    def counted():
        nonlocal calls
        calls += 1
        original()

    monkeypatch.setattr(view, "_recompute_local_from_world_all", counted)

    with view.xform_world_space_writer():
        pass

    assert calls == 0


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_view_getter_inside_scope_raises(device, view_factory):
    """View-level getters raise ``RuntimeError`` while a writer scope is active."""
    bundle = view_factory(num_envs=1, device=device)
    view = bundle.view
    view.get_world_poses()

    with view.xform_world_space_writer():
        with pytest.raises(RuntimeError, match="while a writer scope is active"):
            view.get_world_poses()
        with pytest.raises(RuntimeError, match="while a writer scope is active"):
            view.get_local_poses()
        with pytest.raises(RuntimeError, match="while a writer scope is active"):
            view.get_world_scales()
        with pytest.raises(RuntimeError, match="while a writer scope is active"):
            view.get_local_scales()
    # After the scope exits, view getters work again.
    view.get_world_poses()
