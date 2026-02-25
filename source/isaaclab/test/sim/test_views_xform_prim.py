# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest  # noqa: E402
import torch  # noqa: E402

try:
    from isaacsim.core.prims import XFormPrim as _IsaacSimXformPrimView
except (ModuleNotFoundError, ImportError):
    _IsaacSimXformPrimView = None


import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.views import XformPrimView as XformPrimView  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402


@pytest.fixture(autouse=True)
def test_setup_teardown():
    """Create a blank new stage for each test."""
    # Setup: Create a new stage
    sim_utils.create_new_stage()
    sim_utils.update_stage()

    # Yield for the test
    yield

    # Teardown: Clear stage after each test
    sim_utils.clear_stage()
    sim_utils.SimulationContext.clear_instance()


"""
Helper functions.
"""


def _prepare_indices(index_type, target_indices, num_prims, device):
    """Helper function to prepare indices based on type."""
    if index_type == "list":
        return target_indices, target_indices
    elif index_type == "torch_tensor":
        return torch.tensor(target_indices, dtype=torch.int64, device=device), target_indices
    elif index_type == "slice_none":
        return slice(None), list(range(num_prims))
    else:
        raise ValueError(f"Unknown index type: {index_type}")


def _skip_if_backend_unavailable(backend: str, device: str):
    """Skip tests when the requested backend is unavailable."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if backend == "fabric" and device == "cpu":
        pytest.skip("Warp fabricarray operations on CPU have known issues")


def _prim_type_for_backend(backend: str) -> str:
    """Return a prim type that is compatible with the backend."""
    return "Camera" if backend == "fabric" else "Xform"


def _create_view(pattern: str, device: str, backend: str) -> XformPrimView:
    """Create an XformPrimView for the requested backend."""
    if backend == "fabric":
        sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=device, use_fabric=True))
    return XformPrimView(pattern, device=device)


"""
Tests - Initialization.
"""


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_xform_prim_view_initialization_single_prim(device):
    """Test XformPrimView initialization with a single prim."""
    # check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create a single xform prim
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/Object", "Xform", translation=(1.0, 2.0, 3.0), stage=stage)

    # Create view
    view = XformPrimView("/World/Object", device=device)

    # Verify properties
    assert view.count == 1
    assert view.prim_paths == ["/World/Object"]
    assert view.device == device
    assert len(view.prims) == 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_xform_prim_view_initialization_multiple_prims(device):
    """Test XformPrimView initialization with multiple prims using pattern matching."""
    # check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create multiple prims
    num_prims = 10
    stage = sim_utils.get_current_stage()
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Env_{i}/Object", "Xform", translation=(i * 2.0, 0.0, 1.0), stage=stage)

    # Create view with pattern
    view = XformPrimView("/World/Env_.*/Object", device=device)

    # Verify properties
    assert view.count == num_prims
    assert view.device == device
    assert len(view.prims) == num_prims
    assert view.prim_paths == [f"/World/Env_{i}/Object" for i in range(num_prims)]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_xform_prim_view_initialization_multiple_prims_order(device):
    """Test XformPrimView initialization with multiple prims using pattern matching with multiple objects per prim.

    This test validates that XformPrimView respects USD stage traversal order, which is based on
    creation order (depth-first search), NOT alphabetical/lexical sorting. This is an important
    edge case that ensures deterministic prim ordering that matches USD's internal representation.

    The test creates prims in a deliberately non-alphabetical order (1, 0, A, a, 2) and verifies
    that they are retrieved in creation order, not sorted order (0, 1, 2, A, a).
    """
    # check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create multiple prims
    num_prims = 10
    stage = sim_utils.get_current_stage()

    # NOTE: Prims are created in a specific order to test that XformPrimView respects
    # USD stage traversal order (DFS based on creation order), NOT alphabetical/lexical order.
    # This is an important edge case: children under the same parent are returned in the
    # order they were created, not sorted by name.

    # First batch: Create Object_1, Object_0, Object_A for each environment
    # (intentionally non-alphabetical: 1, 0, A instead of 0, 1, A)
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Env_{i}/Object_1", "Xform", translation=(i * 2.0, -2.0, 1.0), stage=stage)
        sim_utils.create_prim(f"/World/Env_{i}/Object_0", "Xform", translation=(i * 2.0, 2.0, 1.0), stage=stage)
        sim_utils.create_prim(f"/World/Env_{i}/Object_A", "Xform", translation=(i * 2.0, 0.0, -1.0), stage=stage)

    # Second batch: Create Object_a, Object_2 for each environment
    # (created after the first batch to verify traversal is depth-first per environment)
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Env_{i}/Object_a", "Xform", translation=(i * 2.0, 2.0, -1.0), stage=stage)
        sim_utils.create_prim(f"/World/Env_{i}/Object_2", "Xform", translation=(i * 2.0, 2.0, 1.0), stage=stage)

    # Create view with pattern
    view = XformPrimView("/World/Env_.*/Object_.*", device=device)

    # Expected ordering: DFS traversal by environment, with children in creation order
    # For each Env_i, we expect: Object_1, Object_0, Object_A, Object_a, Object_2
    # (matches creation order, NOT alphabetical: would be 0, 1, 2, A, a if sorted)
    expected_prim_paths_ordering = []
    for i in range(num_prims):
        expected_prim_paths_ordering.append(f"/World/Env_{i}/Object_1")
        expected_prim_paths_ordering.append(f"/World/Env_{i}/Object_0")
        expected_prim_paths_ordering.append(f"/World/Env_{i}/Object_A")
        expected_prim_paths_ordering.append(f"/World/Env_{i}/Object_a")
        expected_prim_paths_ordering.append(f"/World/Env_{i}/Object_2")

    # Verify properties
    assert view.count == num_prims * 5
    assert view.device == device
    assert len(view.prims) == num_prims * 5
    assert view.prim_paths == expected_prim_paths_ordering

    # Additional validation: Verify ordering is NOT alphabetical
    # If it were alphabetical, Object_0 would come before Object_1
    alphabetical_order = []
    for i in range(num_prims):
        alphabetical_order.append(f"/World/Env_{i}/Object_0")
        alphabetical_order.append(f"/World/Env_{i}/Object_1")
        alphabetical_order.append(f"/World/Env_{i}/Object_2")
        alphabetical_order.append(f"/World/Env_{i}/Object_A")
        alphabetical_order.append(f"/World/Env_{i}/Object_a")

    assert view.prim_paths != alphabetical_order, (
        "Prim paths should follow creation order, not alphabetical order. "
        "This test validates that USD stage traversal respects creation order."
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_xform_prim_view_standardizes_transform_op(device):
    """Test that XformPrimView standardizes a prim with xformOp:transform to translate/orient/scale."""
    from pxr import Gf, UsdGeom

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    expected_pos = (3.0, -1.0, 0.5)
    matrix = Gf.Matrix4d(1.0)
    matrix.SetTranslateOnly(Gf.Vec3d(*expected_pos))

    stage = sim_utils.get_current_stage()
    prim = stage.DefinePrim("/World/TransformPrim", "Xform")
    UsdGeom.Xformable(prim).AddTransformOp().Set(matrix)

    view = XformPrimView("/World/TransformPrim", device=device)

    assert view.count == 1
    assert sim_utils.validate_standard_xform_ops(view.prims[0])

    xformable = UsdGeom.Xformable(view.prims[0])
    ordered_ops = xformable.GetOrderedXformOps()
    op_names = [op.GetOpName() for op in ordered_ops]
    assert op_names == ["xformOp:translate", "xformOp:orient", "xformOp:scale"]

    assert ordered_ops[0].Get() == Gf.Vec3d(*expected_pos)
    assert ordered_ops[1].Get() == Gf.Quatd(1.0, 0.0, 0.0, 0.0)
    assert ordered_ops[2].Get() == Gf.Vec3d(1.0, 1.0, 1.0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_xform_prim_view_initialization_empty_pattern(device):
    """Test XformPrimView initialization with pattern that matches no prims."""
    # check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    sim_utils.create_new_stage()

    # Create view with pattern that matches nothing
    view = XformPrimView("/World/NonExistent_.*", device=device)

    # Should have zero count
    assert view.count == 0
    assert len(view.prims) == 0


"""
Tests - Getters.
"""


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["usd", "fabric"])
def test_get_world_poses(device, backend):
    """Test getting world poses from XformPrimView."""
    _skip_if_backend_unavailable(backend, device)

    stage = sim_utils.get_current_stage()
    prim_type = _prim_type_for_backend(backend)

    # Create prims with known world poses
    expected_positions = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
    expected_orientations = [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.7071068, 0.7071068), (0.7071068, 0.0, 0.0, 0.7071068)]

    for i, (pos, quat) in enumerate(zip(expected_positions, expected_orientations)):
        sim_utils.create_prim(f"/World/Object_{i}", prim_type, translation=pos, orientation=quat, stage=stage)

    # Create view
    view = _create_view("/World/Object_.*", device=device, backend=backend)

    # Convert expected values to tensors
    expected_positions_tensor = torch.tensor(expected_positions, dtype=torch.float32, device=device)
    expected_orientations_tensor = torch.tensor(expected_orientations, dtype=torch.float32, device=device)

    # Get world poses
    positions, orientations = view.get_world_poses()

    # Verify shapes
    assert positions.shape == (3, 3)
    assert orientations.shape == (3, 4)

    # Verify positions
    torch.testing.assert_close(positions, expected_positions_tensor, atol=1e-5, rtol=0)

    # Verify orientations (allow for quaternion sign ambiguity)
    try:
        torch.testing.assert_close(orientations, expected_orientations_tensor, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(orientations, -expected_orientations_tensor, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["usd", "fabric"])
def test_get_local_poses(device, backend):
    """Test getting local poses from XformPrimView."""
    _skip_if_backend_unavailable(backend, device)

    stage = sim_utils.get_current_stage()
    prim_type = _prim_type_for_backend(backend)

    # Create parent and child prims
    sim_utils.create_prim("/World/Parent", "Xform", translation=(10.0, 0.0, 0.0), stage=stage)

    # Children with different local poses
    expected_local_positions = [(1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0)]
    expected_local_orientations = [
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.7071068, 0.7071068),
        (0.7071068, 0.0, 0.0, 0.7071068),
    ]

    for i, (pos, quat) in enumerate(zip(expected_local_positions, expected_local_orientations)):
        sim_utils.create_prim(f"/World/Parent/Child_{i}", prim_type, translation=pos, orientation=quat, stage=stage)

    # Create view
    view = _create_view("/World/Parent/Child_.*", device=device, backend=backend)

    # Get local poses
    translations, orientations = view.get_local_poses()

    # Verify shapes
    assert translations.shape == (3, 3)
    assert orientations.shape == (3, 4)

    # Convert expected values to tensors
    expected_translations_tensor = torch.tensor(expected_local_positions, dtype=torch.float32, device=device)
    expected_orientations_tensor = torch.tensor(expected_local_orientations, dtype=torch.float32, device=device)

    # Verify translations
    torch.testing.assert_close(translations, expected_translations_tensor, atol=1e-5, rtol=0)

    # Verify orientations (allow for quaternion sign ambiguity)
    try:
        torch.testing.assert_close(orientations, expected_orientations_tensor, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(orientations, -expected_orientations_tensor, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["usd", "fabric"])
def test_get_scales(device, backend):
    """Test getting scales from XformPrimView."""
    _skip_if_backend_unavailable(backend, device)

    stage = sim_utils.get_current_stage()
    prim_type = _prim_type_for_backend(backend)

    # Create prims with different scales
    expected_scales = [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (1.0, 2.0, 3.0)]

    for i, scale in enumerate(expected_scales):
        sim_utils.create_prim(f"/World/Object_{i}", prim_type, scale=scale, stage=stage)

    # Create view
    view = _create_view("/World/Object_.*", device=device, backend=backend)

    expected_scales_tensor = torch.tensor(expected_scales, dtype=torch.float32, device=device)

    # Get scales
    scales = view.get_scales()

    # Verify shape and values
    assert scales.shape == (3, 3)
    torch.testing.assert_close(scales, expected_scales_tensor, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_get_visibility(device):
    """Test getting visibility when all prims are visible."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Create prims (default is visible)
    num_prims = 5
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Object_{i}", "Xform", translation=(float(i), 0.0, 0.0), stage=stage)

    # Create view
    view = XformPrimView("/World/Object_.*", device=device)

    # Get visibility
    visibility = view.get_visibility()

    # Verify shape and values
    assert visibility.shape == (num_prims,)
    assert visibility.dtype == torch.bool
    assert torch.all(visibility), "All prims should be visible by default"


"""
Tests - Setters.
"""


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["usd", "fabric"])
def test_set_world_poses(device, backend):
    """Test setting world poses in XformPrimView."""
    _skip_if_backend_unavailable(backend, device)

    stage = sim_utils.get_current_stage()
    prim_type = _prim_type_for_backend(backend)

    # Create prims
    num_prims = 5
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Object_{i}", prim_type, translation=(0.0, 0.0, 0.0), stage=stage)

    # Create view
    view = _create_view("/World/Object_.*", device=device, backend=backend)

    # Set new world poses
    new_positions = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]], device=device
    )
    new_orientations = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.7071068, 0.7071068],
            [0.7071068, 0.0, 0.0, 0.7071068],
            [0.3826834, 0.0, 0.0, 0.9238795],
            [0.0, 0.7071068, 0.0, 0.7071068],
        ],
        device=device,
    )

    view.set_world_poses(new_positions, new_orientations)

    # Get the poses back
    retrieved_positions, retrieved_orientations = view.get_world_poses()

    # Verify they match
    torch.testing.assert_close(retrieved_positions, new_positions, atol=1e-5, rtol=0)
    # Check quaternions (allow sign flip)
    try:
        torch.testing.assert_close(retrieved_orientations, new_orientations, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(retrieved_orientations, -new_orientations, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["usd", "fabric"])
def test_set_world_poses_only_positions(device, backend):
    """Test setting only positions, leaving orientations unchanged."""
    _skip_if_backend_unavailable(backend, device)

    stage = sim_utils.get_current_stage()
    prim_type = _prim_type_for_backend(backend)

    # Create prims with specific orientations
    initial_quat = (0.0, 0.0, 0.7071068, 0.7071068)  # 90 deg around Z
    for i in range(3):
        sim_utils.create_prim(
            f"/World/Object_{i}", prim_type, translation=(0.0, 0.0, 0.0), orientation=initial_quat, stage=stage
        )

    # Create view
    view = _create_view("/World/Object_.*", device=device, backend=backend)

    # Get initial orientations
    _, initial_orientations = view.get_world_poses()

    # Set only positions
    new_positions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], device=device)
    view.set_world_poses(positions=new_positions, orientations=None)

    # Get poses back
    retrieved_positions, retrieved_orientations = view.get_world_poses()

    # Positions should be updated
    torch.testing.assert_close(retrieved_positions, new_positions, atol=1e-5, rtol=0)

    # Orientations should be unchanged
    try:
        torch.testing.assert_close(retrieved_orientations, initial_orientations, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(retrieved_orientations, -initial_orientations, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["usd", "fabric"])
def test_set_world_poses_only_orientations(device, backend):
    """Test setting only orientations, leaving positions unchanged."""
    _skip_if_backend_unavailable(backend, device)

    stage = sim_utils.get_current_stage()
    prim_type = _prim_type_for_backend(backend)

    # Create prims with specific positions
    for i in range(3):
        sim_utils.create_prim(f"/World/Object_{i}", prim_type, translation=(float(i), 0.0, 0.0), stage=stage)

    # Create view
    view = _create_view("/World/Object_.*", device=device, backend=backend)

    # Get initial positions
    initial_positions, _ = view.get_world_poses()

    # Set only orientations
    new_orientations = torch.tensor(
        [[0.0, 0.0, 0.7071068, 0.7071068], [0.7071068, 0.0, 0.0, 0.7071068], [0.3826834, 0.0, 0.0, 0.9238795]],
        device=device,
    )
    view.set_world_poses(positions=None, orientations=new_orientations)

    # Get poses back
    retrieved_positions, retrieved_orientations = view.get_world_poses()

    # Positions should be unchanged
    torch.testing.assert_close(retrieved_positions, initial_positions, atol=1e-5, rtol=0)

    # Orientations should be updated
    try:
        torch.testing.assert_close(retrieved_orientations, new_orientations, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(retrieved_orientations, -new_orientations, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["usd", "fabric"])
def test_set_world_poses_with_hierarchy(device, backend):
    """Test setting world poses correctly handles parent transformations."""
    _skip_if_backend_unavailable(backend, device)

    stage = sim_utils.get_current_stage()
    child_prim_type = _prim_type_for_backend(backend)

    # Create parent prims
    for i in range(3):
        parent_pos = (i * 10.0, 0.0, 0.0)
        parent_quat = (0.0, 0.0, 0.7071068, 0.7071068)  # 90 deg around Z
        sim_utils.create_prim(
            f"/World/Parent_{i}", "Xform", translation=parent_pos, orientation=parent_quat, stage=stage
        )
        # Create child prims
        sim_utils.create_prim(f"/World/Parent_{i}/Child", child_prim_type, translation=(0.0, 0.0, 0.0), stage=stage)

    # Create view for children
    view = _create_view("/World/Parent_.*/Child", device=device, backend=backend)

    # Set world poses for children
    desired_world_positions = torch.tensor([[5.0, 5.0, 0.0], [15.0, 5.0, 0.0], [25.0, 5.0, 0.0]], device=device)
    desired_world_orientations = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], device=device
    )

    view.set_world_poses(desired_world_positions, desired_world_orientations)

    # Get world poses back
    retrieved_positions, retrieved_orientations = view.get_world_poses()

    # Should match desired world poses
    torch.testing.assert_close(retrieved_positions, desired_world_positions, atol=1e-4, rtol=0)
    try:
        torch.testing.assert_close(retrieved_orientations, desired_world_orientations, atol=1e-4, rtol=0)
    except AssertionError:
        torch.testing.assert_close(retrieved_orientations, -desired_world_orientations, atol=1e-4, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["usd", "fabric"])
def test_set_local_poses(device, backend):
    """Test setting local poses in XformPrimView."""
    _skip_if_backend_unavailable(backend, device)

    stage = sim_utils.get_current_stage()
    prim_type = _prim_type_for_backend(backend)

    # Create parent
    sim_utils.create_prim("/World/Parent", "Xform", translation=(5.0, 5.0, 5.0), stage=stage)

    # Create children
    num_prims = 4
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Parent/Child_{i}", prim_type, translation=(0.0, 0.0, 0.0), stage=stage)

    # Create view
    view = _create_view("/World/Parent/Child_.*", device=device, backend=backend)

    # Set new local poses
    new_translations = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [4.0, 4.0, 4.0]], device=device)
    new_orientations = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.7071068, 0.7071068],
            [0.7071068, 0.0, 0.0, 0.7071068],
            [0.3826834, 0.0, 0.0, 0.9238795],
        ],
        device=device,
    )

    view.set_local_poses(new_translations, new_orientations)

    # Get local poses back
    retrieved_translations, retrieved_orientations = view.get_local_poses()

    # Verify they match
    torch.testing.assert_close(retrieved_translations, new_translations, atol=1e-5, rtol=0)
    try:
        torch.testing.assert_close(retrieved_orientations, new_orientations, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(retrieved_orientations, -new_orientations, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["usd", "fabric"])
def test_set_local_poses_only_translations(device, backend):
    """Test setting only local translations."""
    _skip_if_backend_unavailable(backend, device)

    stage = sim_utils.get_current_stage()
    prim_type = _prim_type_for_backend(backend)

    # Create parent and children with specific orientations
    sim_utils.create_prim("/World/Parent", "Xform", translation=(0.0, 0.0, 0.0), stage=stage)
    initial_quat = (0.0, 0.0, 0.7071068, 0.7071068)

    for i in range(3):
        sim_utils.create_prim(
            f"/World/Parent/Child_{i}",
            prim_type,
            translation=(0.0, 0.0, 0.0),
            orientation=initial_quat,
            stage=stage,
        )

    # Create view
    view = _create_view("/World/Parent/Child_.*", device=device, backend=backend)

    # Get initial orientations
    _, initial_orientations = view.get_local_poses()

    # Set only translations
    new_translations = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], device=device)
    view.set_local_poses(translations=new_translations, orientations=None)

    # Get poses back
    retrieved_translations, retrieved_orientations = view.get_local_poses()

    # Translations should be updated
    torch.testing.assert_close(retrieved_translations, new_translations, atol=1e-5, rtol=0)

    # Orientations should be unchanged
    try:
        torch.testing.assert_close(retrieved_orientations, initial_orientations, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(retrieved_orientations, -initial_orientations, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["usd", "fabric"])
def test_set_scales(device, backend):
    """Test setting scales in XformPrimView."""
    _skip_if_backend_unavailable(backend, device)

    stage = sim_utils.get_current_stage()
    prim_type = _prim_type_for_backend(backend)

    # Create prims
    num_prims = 5
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Object_{i}", prim_type, scale=(1.0, 1.0, 1.0), stage=stage)

    # Create view
    view = _create_view("/World/Object_.*", device=device, backend=backend)

    # Set new scales
    new_scales = torch.tensor(
        [[2.0, 2.0, 2.0], [1.0, 2.0, 3.0], [0.5, 0.5, 0.5], [3.0, 1.0, 2.0], [1.5, 1.5, 1.5]], device=device
    )

    view.set_scales(new_scales)

    # Get scales back
    retrieved_scales = view.get_scales()

    # Verify they match
    torch.testing.assert_close(retrieved_scales, new_scales, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_set_visibility(device):
    """Test toggling visibility multiple times."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Create prims
    num_prims = 3
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Object_{i}", "Xform", stage=stage)

    # Create view
    view = XformPrimView("/World/Object_.*", device=device)

    # Initial state: all visible
    visibility = view.get_visibility()
    assert torch.all(visibility), "All should be visible initially"

    # Make all invisible
    view.set_visibility(torch.zeros(num_prims, dtype=torch.bool, device=device))
    visibility = view.get_visibility()
    assert not torch.any(visibility), "All should be invisible"

    # Make all visible again
    view.set_visibility(torch.ones(num_prims, dtype=torch.bool, device=device))
    visibility = view.get_visibility()
    assert torch.all(visibility), "All should be visible again"

    # Toggle individual prims
    view.set_visibility(torch.tensor([False], dtype=torch.bool, device=device), indices=[1])
    visibility = view.get_visibility()
    assert visibility[0] and not visibility[1] and visibility[2], "Only middle prim should be invisible"


"""
Tests - Index Handling.
"""


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("index_type", ["list", "torch_tensor", "slice_none"])
@pytest.mark.parametrize("method", ["world_poses", "local_poses", "scales", "visibility"])
def test_index_types_get_methods(device, index_type, method):
    """Test that getter methods work with different index types."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Create prims based on method type
    num_prims = 10
    if method == "local_poses":
        # Create parent and children for local poses
        sim_utils.create_prim("/World/Parent", "Xform", translation=(10.0, 0.0, 0.0), stage=stage)
        for i in range(num_prims):
            sim_utils.create_prim(
                f"/World/Parent/Child_{i}", "Xform", translation=(float(i), float(i) * 0.5, 0.0), stage=stage
            )
        view = XformPrimView("/World/Parent/Child_.*", device=device)
    elif method == "scales":
        # Create prims with different scales
        for i in range(num_prims):
            scale = (1.0 + i * 0.5, 1.0 + i * 0.3, 1.0 + i * 0.2)
            sim_utils.create_prim(f"/World/Object_{i}", "Xform", scale=scale, stage=stage)
        view = XformPrimView("/World/Object_.*", device=device)
    else:  # world_poses
        # Create prims with different positions
        for i in range(num_prims):
            sim_utils.create_prim(f"/World/Object_{i}", "Xform", translation=(float(i), 0.0, 0.0), stage=stage)
        view = XformPrimView("/World/Object_.*", device=device)

    # Get all data as reference
    if method == "world_poses":
        all_data1, all_data2 = view.get_world_poses()
    elif method == "local_poses":
        all_data1, all_data2 = view.get_local_poses()
    elif method == "scales":
        all_data1 = view.get_scales()
        all_data2 = None
    else:  # visibility
        all_data1 = view.get_visibility()
        all_data2 = None

    # Prepare indices
    target_indices_base = [2, 5, 7]
    indices, target_indices = _prepare_indices(index_type, target_indices_base, num_prims, device)

    # Get subset
    if method == "world_poses":
        subset_data1, subset_data2 = view.get_world_poses(indices=indices)  # type: ignore[arg-type]
    elif method == "local_poses":
        subset_data1, subset_data2 = view.get_local_poses(indices=indices)  # type: ignore[arg-type]
    elif method == "scales":
        subset_data1 = view.get_scales(indices=indices)  # type: ignore[arg-type]
        subset_data2 = None
    else:  # visibility
        subset_data1 = view.get_visibility(indices=indices)  # type: ignore[arg-type]
        subset_data2 = None

    # Verify shapes
    expected_count = len(target_indices)
    if method == "visibility":
        assert subset_data1.shape == (expected_count,)
    else:
        assert subset_data1.shape == (expected_count, 3)
    if subset_data2 is not None:
        assert subset_data2.shape == (expected_count, 4)

    # Verify values
    target_indices_tensor = torch.tensor(target_indices, dtype=torch.int64, device=device)
    torch.testing.assert_close(subset_data1, all_data1[target_indices_tensor], atol=1e-5, rtol=0)
    if subset_data2 is not None and all_data2 is not None:
        torch.testing.assert_close(subset_data2, all_data2[target_indices_tensor], atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("index_type", ["list", "torch_tensor", "slice_none"])
@pytest.mark.parametrize("method", ["world_poses", "local_poses", "scales", "visibility"])
def test_index_types_set_methods(device, index_type, method):
    """Test that setter methods work with different index types."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Create prims based on method type
    num_prims = 10
    if method == "local_poses":
        # Create parent and children for local poses
        sim_utils.create_prim("/World/Parent", "Xform", translation=(5.0, 5.0, 0.0), stage=stage)
        for i in range(num_prims):
            sim_utils.create_prim(f"/World/Parent/Child_{i}", "Xform", translation=(float(i), 0.0, 0.0), stage=stage)
        view = XformPrimView("/World/Parent/Child_.*", device=device)
    else:  # world_poses or scales
        for i in range(num_prims):
            sim_utils.create_prim(f"/World/Object_{i}", "Xform", translation=(0.0, 0.0, 0.0), stage=stage)
        view = XformPrimView("/World/Object_.*", device=device)

    # Get initial data
    if method == "world_poses":
        initial_data1, initial_data2 = view.get_world_poses()
    elif method == "local_poses":
        initial_data1, initial_data2 = view.get_local_poses()
    elif method == "scales":
        initial_data1 = view.get_scales()
        initial_data2 = None
    else:  # visibility
        initial_data1 = view.get_visibility()
        initial_data2 = None

    # Prepare indices
    target_indices_base = [2, 5, 7]
    indices, target_indices = _prepare_indices(index_type, target_indices_base, num_prims, device)

    # Prepare new data
    num_to_set = len(target_indices)
    if method in ["world_poses", "local_poses"]:
        new_data1 = torch.randn(num_to_set, 3, device=device) * 10.0
        new_data2 = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * num_to_set, dtype=torch.float32, device=device)
    elif method == "scales":
        new_data1 = torch.rand(num_to_set, 3, device=device) * 2.0 + 0.5
        new_data2 = None
    else:  # visibility
        # Set to False to test change (default is True)
        new_data1 = torch.zeros(num_to_set, dtype=torch.bool, device=device)
        new_data2 = None

    # Set data
    if method == "world_poses":
        view.set_world_poses(positions=new_data1, orientations=new_data2, indices=indices)  # type: ignore[arg-type]
    elif method == "local_poses":
        view.set_local_poses(translations=new_data1, orientations=new_data2, indices=indices)  # type: ignore[arg-type]
    elif method == "scales":
        view.set_scales(scales=new_data1, indices=indices)  # type: ignore[arg-type]
    else:  # visibility
        view.set_visibility(visibility=new_data1, indices=indices)  # type: ignore[arg-type]

    # Get all data after update
    if method == "world_poses":
        updated_data1, updated_data2 = view.get_world_poses()
    elif method == "local_poses":
        updated_data1, updated_data2 = view.get_local_poses()
    elif method == "scales":
        updated_data1 = view.get_scales()
        updated_data2 = None
    else:  # visibility
        updated_data1 = view.get_visibility()
        updated_data2 = None

    # Verify that specified indices were updated
    for i, target_idx in enumerate(target_indices):
        torch.testing.assert_close(updated_data1[target_idx], new_data1[i], atol=1e-5, rtol=0)
        if new_data2 is not None and updated_data2 is not None:
            try:
                torch.testing.assert_close(updated_data2[target_idx], new_data2[i], atol=1e-5, rtol=0)
            except AssertionError:
                # Account for quaternion sign ambiguity
                torch.testing.assert_close(updated_data2[target_idx], -new_data2[i], atol=1e-5, rtol=0)

    # Verify that other indices were NOT updated (only for non-slice(None) cases)
    if index_type != "slice_none":
        for i in range(num_prims):
            if i not in target_indices:
                torch.testing.assert_close(updated_data1[i], initial_data1[i], atol=1e-5, rtol=0)
                if initial_data2 is not None and updated_data2 is not None:
                    try:
                        torch.testing.assert_close(updated_data2[i], initial_data2[i], atol=1e-5, rtol=0)
                    except AssertionError:
                        # Account for quaternion sign ambiguity
                        torch.testing.assert_close(updated_data2[i], -initial_data2[i], atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["usd", "fabric"])
def test_indices_single_element(device, backend):
    """Test with a single index."""
    _skip_if_backend_unavailable(backend, device)

    stage = sim_utils.get_current_stage()
    prim_type = _prim_type_for_backend(backend)

    # Create prims
    num_prims = 5
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Object_{i}", prim_type, translation=(float(i), 0.0, 0.0), stage=stage)

    # Create view
    view = _create_view("/World/Object_.*", device=device, backend=backend)

    # Test with single index
    indices = [3]
    positions, orientations = view.get_world_poses(indices=indices)

    # Verify shapes
    assert positions.shape == (1, 3)
    assert orientations.shape == (1, 4)

    # Set pose for single index
    new_position = torch.tensor([[100.0, 200.0, 300.0]], device=device)
    view.set_world_poses(positions=new_position, indices=indices)

    # Verify it was set
    retrieved_positions, _ = view.get_world_poses(indices=indices)
    torch.testing.assert_close(retrieved_positions, new_position, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["usd", "fabric"])
def test_indices_out_of_order(device, backend):
    """Test with indices provided in non-sequential order."""
    _skip_if_backend_unavailable(backend, device)

    stage = sim_utils.get_current_stage()
    prim_type = _prim_type_for_backend(backend)

    # Create prims
    num_prims = 10
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Object_{i}", prim_type, translation=(0.0, 0.0, 0.0), stage=stage)

    # Create view
    view = _create_view("/World/Object_.*", device=device, backend=backend)

    # Use out-of-order indices
    indices = [7, 2, 9, 0, 5]
    new_positions = torch.tensor(
        [[7.0, 0.0, 0.0], [2.0, 0.0, 0.0], [9.0, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], device=device
    )

    # Set poses with out-of-order indices
    view.set_world_poses(positions=new_positions, indices=indices)

    # Get all poses
    all_positions, _ = view.get_world_poses()

    # Verify each index got the correct value
    expected_x_values = [0.0, 0.0, 2.0, 0.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0]
    for i in range(num_prims):
        assert abs(all_positions[i, 0].item() - expected_x_values[i]) < 1e-5


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["usd", "fabric"])
def test_indices_with_only_positions_or_orientations(device, backend):
    """Test indices work correctly when setting only positions or only orientations."""
    _skip_if_backend_unavailable(backend, device)

    stage = sim_utils.get_current_stage()
    prim_type = _prim_type_for_backend(backend)

    # Create prims
    num_prims = 5
    for i in range(num_prims):
        sim_utils.create_prim(
            f"/World/Object_{i}",
            prim_type,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            stage=stage,
        )

    # Create view
    view = _create_view("/World/Object_.*", device=device, backend=backend)

    # Get initial poses
    initial_positions, initial_orientations = view.get_world_poses()

    # Set only positions for specific indices
    indices = [1, 3]
    new_positions = torch.tensor([[10.0, 0.0, 0.0], [30.0, 0.0, 0.0]], device=device)
    view.set_world_poses(positions=new_positions, orientations=None, indices=indices)

    # Get updated poses
    updated_positions, updated_orientations = view.get_world_poses()

    # Verify positions updated for indices 1 and 3, others unchanged
    torch.testing.assert_close(updated_positions[1], new_positions[0], atol=1e-5, rtol=0)
    torch.testing.assert_close(updated_positions[3], new_positions[1], atol=1e-5, rtol=0)
    torch.testing.assert_close(updated_positions[0], initial_positions[0], atol=1e-5, rtol=0)

    # Verify all orientations unchanged
    try:
        torch.testing.assert_close(updated_orientations, initial_orientations, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(updated_orientations, -initial_orientations, atol=1e-5, rtol=0)

    # Now set only orientations for different indices
    indices2 = [0, 4]
    new_orientations = torch.tensor([[0.0, 0.0, 0.7071068, 0.7071068], [0.7071068, 0.0, 0.0, 0.7071068]], device=device)
    view.set_world_poses(positions=None, orientations=new_orientations, indices=indices2)

    # Get final poses
    final_positions, final_orientations = view.get_world_poses()

    # Verify positions unchanged from previous step
    torch.testing.assert_close(final_positions, updated_positions, atol=1e-5, rtol=0)

    # Verify orientations updated for indices 0 and 4
    try:
        torch.testing.assert_close(final_orientations[0], new_orientations[0], atol=1e-5, rtol=0)
        torch.testing.assert_close(final_orientations[4], new_orientations[1], atol=1e-5, rtol=0)
    except AssertionError:
        # Account for quaternion sign ambiguity
        torch.testing.assert_close(final_orientations[0], -new_orientations[0], atol=1e-5, rtol=0)
        torch.testing.assert_close(final_orientations[4], -new_orientations[1], atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_index_type_none_equivalent_to_all(device):
    """Test that indices=None is equivalent to getting/setting all prims."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Create prims
    num_prims = 6
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Object_{i}", "Xform", translation=(float(i), 0.0, 0.0), stage=stage)

    # Create view
    view = XformPrimView("/World/Object_.*", device=device)

    # Get poses with indices=None
    pos_none, quat_none = view.get_world_poses(indices=None)

    # Get poses with no argument (default)
    pos_default, quat_default = view.get_world_poses()

    # Get poses with slice(None)
    pos_slice, quat_slice = view.get_world_poses(indices=slice(None))  # type: ignore[arg-type]

    # All should be equivalent
    torch.testing.assert_close(pos_none, pos_default, atol=1e-10, rtol=0)
    torch.testing.assert_close(quat_none, quat_default, atol=1e-10, rtol=0)
    torch.testing.assert_close(pos_none, pos_slice, atol=1e-10, rtol=0)
    torch.testing.assert_close(quat_none, quat_slice, atol=1e-10, rtol=0)

    # Test the same for set operations
    new_positions = torch.randn(num_prims, 3, device=device) * 10.0
    new_orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * num_prims, dtype=torch.float32, device=device)

    # Set with indices=None
    view.set_world_poses(positions=new_positions, orientations=new_orientations, indices=None)
    pos_after_none, quat_after_none = view.get_world_poses()

    # Reset
    view.set_world_poses(positions=torch.zeros(num_prims, 3, device=device), indices=None)

    # Set with slice(None)
    view.set_world_poses(
        positions=new_positions,
        orientations=new_orientations,
        indices=slice(None),  # type: ignore[arg-type]
    )
    pos_after_slice, quat_after_slice = view.get_world_poses()

    # Should be equivalent
    torch.testing.assert_close(pos_after_none, pos_after_slice, atol=1e-5, rtol=0)
    torch.testing.assert_close(quat_after_none, quat_after_slice, atol=1e-5, rtol=0)


"""
Tests - Integration.
"""


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_with_franka_robots(device):
    """Test XformPrimView with real Franka robot USD assets."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Load Franka robot assets
    franka_usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"

    # Add two Franka robots to the stage
    sim_utils.create_prim("/World/Franka_1", "Xform", usd_path=franka_usd_path, stage=stage)
    sim_utils.create_prim("/World/Franka_2", "Xform", usd_path=franka_usd_path, stage=stage)

    # Create view for both Frankas
    frankas_view = XformPrimView("/World/Franka_.*", device=device)

    # Verify count
    assert frankas_view.count == 2

    # Get initial world poses (should be at origin)
    initial_positions, initial_orientations = frankas_view.get_world_poses()

    # Verify initial positions are at origin
    expected_initial_positions = torch.zeros(2, 3, device=device)
    torch.testing.assert_close(initial_positions, expected_initial_positions, atol=1e-5, rtol=0)

    # Verify initial orientations are identity
    expected_initial_orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], device=device)
    try:
        torch.testing.assert_close(initial_orientations, expected_initial_orientations, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(initial_orientations, -expected_initial_orientations, atol=1e-5, rtol=0)

    # Set new world poses
    new_positions = torch.tensor([[10.0, 10.0, 0.0], [-40.0, -40.0, 0.0]], device=device)
    # 90° rotation around Z axis for first, -90° for second
    new_orientations = torch.tensor(
        [[0.0, 0.0, 0.7071068, 0.7071068], [0.0, 0.0, -0.7071068, 0.7071068]], device=device
    )

    frankas_view.set_world_poses(positions=new_positions, orientations=new_orientations)

    # Get poses back and verify
    retrieved_positions, retrieved_orientations = frankas_view.get_world_poses()

    torch.testing.assert_close(retrieved_positions, new_positions, atol=1e-5, rtol=0)
    try:
        torch.testing.assert_close(retrieved_orientations, new_orientations, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(retrieved_orientations, -new_orientations, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_with_nested_targets(device):
    """Test with nested frame/target structure similar to Isaac Sim tests."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Create frames and targets
    for i in range(1, 4):
        sim_utils.create_prim(f"/World/Frame_{i}", "Xform", stage=stage)
        sim_utils.create_prim(f"/World/Frame_{i}/Target", "Xform", stage=stage)

    # Create views
    frames_view = XformPrimView("/World/Frame_.*", device=device)
    targets_view = XformPrimView("/World/Frame_.*/Target", device=device)

    assert frames_view.count == 3
    assert targets_view.count == 3

    # Set local poses for frames
    frame_translations = torch.tensor([[0.0, 0.0, 0.0], [0.0, 10.0, 5.0], [0.0, 3.0, 5.0]], device=device)
    frames_view.set_local_poses(translations=frame_translations)

    # Set local poses for targets
    target_translations = torch.tensor([[0.0, 20.0, 10.0], [0.0, 30.0, 20.0], [0.0, 50.0, 10.0]], device=device)
    targets_view.set_local_poses(translations=target_translations)

    # Get world poses of targets
    world_positions, _ = targets_view.get_world_poses()

    # Expected world positions are frame_translation + target_translation
    expected_positions = torch.tensor([[0.0, 20.0, 10.0], [0.0, 40.0, 25.0], [0.0, 53.0, 15.0]], device=device)

    torch.testing.assert_close(world_positions, expected_positions, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_visibility_with_hierarchy(device):
    """Test visibility with parent-child hierarchy and inheritance."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Create parent and children
    sim_utils.create_prim("/World/Parent", "Xform", stage=stage)

    num_children = 4
    for i in range(num_children):
        sim_utils.create_prim(f"/World/Parent/Child_{i}", "Xform", stage=stage)

    # Create views for both parent and children
    parent_view = XformPrimView("/World/Parent", device=device)
    children_view = XformPrimView("/World/Parent/Child_.*", device=device)

    # Verify parent and all children are visible initially
    parent_visibility = parent_view.get_visibility()
    children_visibility = children_view.get_visibility()
    assert parent_visibility[0], "Parent should be visible initially"
    assert torch.all(children_visibility), "All children should be visible initially"

    # Make some children invisible directly
    new_visibility = torch.tensor([True, False, True, False], dtype=torch.bool, device=device)
    children_view.set_visibility(new_visibility)

    # Verify the visibility changes
    retrieved_visibility = children_view.get_visibility()
    torch.testing.assert_close(retrieved_visibility, new_visibility)

    # Make all children visible again
    children_view.set_visibility(torch.ones(num_children, dtype=torch.bool, device=device))
    all_visible = children_view.get_visibility()
    assert torch.all(all_visible), "All children should be visible again"

    # Now test parent visibility inheritance:
    # Make parent invisible
    parent_view.set_visibility(torch.tensor([False], dtype=torch.bool, device=device))

    # Verify parent is invisible
    parent_visibility = parent_view.get_visibility()
    assert not parent_visibility[0], "Parent should be invisible"

    # Verify children are also invisible (due to parent being invisible)
    children_visibility = children_view.get_visibility()
    assert not torch.any(children_visibility), "All children should be invisible when parent is invisible"

    # Make parent visible again
    parent_view.set_visibility(torch.tensor([True], dtype=torch.bool, device=device))

    # Verify parent is visible
    parent_visibility = parent_view.get_visibility()
    assert parent_visibility[0], "Parent should be visible again"

    # Verify children are also visible again
    children_visibility = children_view.get_visibility()
    assert torch.all(children_visibility), "All children should be visible again when parent is visible"


"""
Tests - Comparison with Isaac Sim Implementation.
"""


def test_compare_get_world_poses_with_isaacsim():
    """Compare get_world_poses with Isaac Sim's implementation."""
    stage = sim_utils.get_current_stage()

    # Check if Isaac Sim is available
    if _IsaacSimXformPrimView is None:
        pytest.skip("Isaac Sim is not available")

    # Create prims with various poses
    num_prims = 10
    for i in range(num_prims):
        pos = (i * 2.0, i * 0.5, i * 1.5)
        # Vary orientations
        if i % 3 == 0:
            quat = (0.0, 0.0, 0.0, 1.0)  # Identity
        elif i % 3 == 1:
            quat = (0.0, 0.0, 0.7071068, 0.7071068)  # 90 deg around Z
        else:
            quat = (0.7071068, 0.0, 0.0, 0.7071068)  # 90 deg around X
        sim_utils.create_prim(f"/World/Env_{i}/Object", "Xform", translation=pos, orientation=quat, stage=stage)

    pattern = "/World/Env_.*/Object"

    # Create both views
    isaaclab_view = XformPrimView(pattern, device="cpu")
    isaacsim_view = _IsaacSimXformPrimView(pattern, reset_xform_properties=False)

    # Get world poses from both
    isaaclab_pos, isaaclab_quat = isaaclab_view.get_world_poses()  # xyzw
    isaacsim_pos, isaacsim_quat = isaacsim_view.get_world_poses()  # wxyz

    # Convert Isaac Sim results to torch tensors if needed
    if not isinstance(isaacsim_pos, torch.Tensor):
        isaacsim_pos = torch.tensor(isaacsim_pos, dtype=torch.float32)
    if not isinstance(isaacsim_quat, torch.Tensor):
        isaacsim_quat = torch.tensor(isaacsim_quat, dtype=torch.float32).roll(-1, dims=1)

    # Compare results
    torch.testing.assert_close(isaaclab_pos, isaacsim_pos, atol=1e-5, rtol=0)

    # Compare quaternions (account for sign ambiguity)
    try:
        torch.testing.assert_close(isaaclab_quat, isaacsim_quat, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(isaaclab_quat, -isaacsim_quat, atol=1e-5, rtol=0)


def test_compare_set_world_poses_with_isaacsim():
    """Compare set_world_poses with Isaac Sim's implementation."""
    stage = sim_utils.get_current_stage()

    # Check if Isaac Sim is available
    if _IsaacSimXformPrimView is None:
        pytest.skip("Isaac Sim is not available")

    # Create prims
    num_prims = 8
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Env_{i}/Object", "Xform", translation=(0.0, 0.0, 0.0), stage=stage)

    pattern = "/World/Env_.*/Object"

    # Create both views
    isaaclab_view = XformPrimView(pattern, device="cpu")
    isaacsim_view = _IsaacSimXformPrimView(pattern, reset_xform_properties=False)

    # Generate new poses
    new_positions = torch.randn(num_prims, 3) * 10.0
    new_orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * num_prims, dtype=torch.float32)

    # Set poses using both implementations
    isaaclab_view.set_world_poses(new_positions.clone(), new_orientations.clone())  # xyzw
    isaacsim_view.set_world_poses(new_positions.clone(), new_orientations.clone().roll(1, dims=1))  # wxyz

    # Get poses back from both
    isaaclab_pos, isaaclab_quat = isaaclab_view.get_world_poses()  # xyzw
    isaacsim_pos, isaacsim_quat = isaacsim_view.get_world_poses()  # wxyz

    # Convert Isaac Sim results to torch tensors if needed
    if not isinstance(isaacsim_pos, torch.Tensor):
        isaacsim_pos = torch.tensor(isaacsim_pos, dtype=torch.float32)
    if not isinstance(isaacsim_quat, torch.Tensor):
        isaacsim_quat = torch.tensor(isaacsim_quat, dtype=torch.float32).roll(-1, dims=1)

    # Compare results - both implementations should produce the same world poses
    torch.testing.assert_close(isaaclab_pos, isaacsim_pos, atol=1e-4, rtol=0)
    try:
        torch.testing.assert_close(isaaclab_quat, isaacsim_quat, atol=1e-4, rtol=0)
    except AssertionError:
        torch.testing.assert_close(isaaclab_quat, -isaacsim_quat, atol=1e-4, rtol=0)


def test_compare_get_local_poses_with_isaacsim():
    """Compare get_local_poses with Isaac Sim's implementation."""
    stage = sim_utils.get_current_stage()

    # Check if Isaac Sim is available
    if _IsaacSimXformPrimView is None:
        pytest.skip("Isaac Sim is not available")

    # Create hierarchical prims
    num_prims = 5
    for i in range(num_prims):
        # Create parent
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 5.0, 0.0, 0.0), stage=stage)
        # Create child with local pose
        local_pos = (1.0, float(i), 0.0)
        local_quat = (0.0, 0.0, 0.0, 1.0) if i % 2 == 0 else (0.0, 0.0, 0.7071068, 0.7071068)
        sim_utils.create_prim(
            f"/World/Env_{i}/Object", "Xform", translation=local_pos, orientation=local_quat, stage=stage
        )

    pattern = "/World/Env_.*/Object"

    # Create both views
    isaaclab_view = XformPrimView(pattern, device="cpu")
    isaacsim_view = _IsaacSimXformPrimView(pattern, reset_xform_properties=False)

    # Get local poses from both
    isaaclab_trans, isaaclab_quat = isaaclab_view.get_local_poses()
    isaacsim_trans, isaacsim_quat = isaacsim_view.get_local_poses()

    # Convert Isaac Sim results to torch tensors if needed
    if not isinstance(isaacsim_trans, torch.Tensor):
        isaacsim_trans = torch.tensor(isaacsim_trans, dtype=torch.float32)
    if not isinstance(isaacsim_quat, torch.Tensor):
        isaacsim_quat = torch.tensor(isaacsim_quat, dtype=torch.float32).roll(-1, dims=1)

    # Compare results
    torch.testing.assert_close(isaaclab_trans, isaacsim_trans, atol=1e-5, rtol=0)
    try:
        torch.testing.assert_close(isaaclab_quat, isaacsim_quat, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(isaaclab_quat, -isaacsim_quat, atol=1e-5, rtol=0)


def test_compare_set_local_poses_with_isaacsim():
    """Compare set_local_poses with Isaac Sim's implementation."""
    stage = sim_utils.get_current_stage()

    # Check if Isaac Sim is available
    if _IsaacSimXformPrimView is None:
        pytest.skip("Isaac Sim is not available")

    # Create hierarchical prims
    num_prims = 6
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 3.0, 0.0, 0.0), stage=stage)
        sim_utils.create_prim(f"/World/Env_{i}/Object", "Xform", translation=(0.0, 0.0, 0.0), stage=stage)

    pattern = "/World/Env_.*/Object"

    # Create both views
    isaaclab_view = XformPrimView(pattern, device="cpu")
    isaacsim_view = _IsaacSimXformPrimView(pattern, reset_xform_properties=False)

    # Generate new local poses
    new_translations = torch.randn(num_prims, 3) * 5.0
    new_orientations = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.7071068, 0.7071068]] * (num_prims // 2), dtype=torch.float32
    )

    # Set local poses using both implementations
    isaaclab_view.set_local_poses(new_translations.clone(), new_orientations.clone())
    isaacsim_view.set_local_poses(new_translations.clone(), new_orientations.clone().roll(1, dims=1))

    # Get local poses back from both
    isaaclab_trans, isaaclab_quat = isaaclab_view.get_local_poses()
    isaacsim_trans, isaacsim_quat = isaacsim_view.get_local_poses()

    # Convert Isaac Sim results to torch tensors if needed
    if not isinstance(isaacsim_trans, torch.Tensor):
        isaacsim_trans = torch.tensor(isaacsim_trans, dtype=torch.float32)
    if not isinstance(isaacsim_quat, torch.Tensor):
        isaacsim_quat = torch.tensor(isaacsim_quat, dtype=torch.float32).roll(-1, dims=1)

    # Compare results
    torch.testing.assert_close(isaaclab_trans, isaacsim_trans, atol=1e-4, rtol=0)
    try:
        torch.testing.assert_close(isaaclab_quat, isaacsim_quat, atol=1e-4, rtol=0)
    except AssertionError:
        torch.testing.assert_close(isaaclab_quat, -isaacsim_quat, atol=1e-4, rtol=0)


"""
Tests - Fabric Operations.
"""


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fabric_initialization(device):
    """Test XformPrimView initialization with Fabric enabled."""
    _skip_if_backend_unavailable("fabric", device)

    stage = sim_utils.get_current_stage()

    # Create camera prims (Boundable prims that support Fabric)
    num_prims = 5
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Cam_{i}", "Camera", translation=(i * 1.0, 0.0, 1.0), stage=stage)

    # Create view with Fabric enabled
    view = _create_view("/World/Cam_.*", device=device, backend="fabric")

    # Verify properties
    assert view.count == num_prims
    assert view.device == device
    assert len(view.prims) == num_prims


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fabric_usd_consistency(device):
    """Test that Fabric round-trip (write→read) is consistent, matching Isaac Sim's design.

    Note: This does NOT test Fabric vs USD reads on initialization, as Fabric is designed
    for write-first workflows. Instead, it tests that:
    1. Fabric write→read round-trip works correctly
    2. This matches Isaac Sim's Fabric behavior
    """
    _skip_if_backend_unavailable("fabric", device)

    stage = sim_utils.get_current_stage()

    # Create prims
    num_prims = 5
    for i in range(num_prims):
        sim_utils.create_prim(
            f"/World/Cam_{i}",
            "Camera",
            translation=(i * 1.0, 2.0, 3.0),
            orientation=(0.0, 0.0, 0.7071068, 0.7071068),
            stage=stage,
        )

    # Create Fabric view
    view_fabric = _create_view("/World/Cam_.*", device=device, backend="fabric")

    # Test Fabric write→read round-trip (Isaac Sim's intended workflow)
    # Initialize Fabric state by WRITING first
    init_positions = torch.zeros((num_prims, 3), dtype=torch.float32, device=device)
    init_positions[:, 0] = torch.arange(num_prims, dtype=torch.float32, device=device)
    init_positions[:, 1] = 2.0
    init_positions[:, 2] = 3.0
    init_orientations = torch.tensor([[0.0, 0.0, 0.7071068, 0.7071068]] * num_prims, dtype=torch.float32, device=device)

    view_fabric.set_world_poses(init_positions, init_orientations)

    # Read back from Fabric (should match what we wrote)
    pos_fabric, quat_fabric = view_fabric.get_world_poses()
    torch.testing.assert_close(pos_fabric, init_positions, atol=1e-4, rtol=0)
    torch.testing.assert_close(quat_fabric, init_orientations, atol=1e-4, rtol=0)

    # Test another round-trip with different values
    new_positions = torch.rand((num_prims, 3), dtype=torch.float32, device=device) * 10.0
    new_orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * num_prims, dtype=torch.float32, device=device)

    view_fabric.set_world_poses(new_positions, new_orientations)

    # Read back from Fabric (should match)
    pos_fabric_after, quat_fabric_after = view_fabric.get_world_poses()
    torch.testing.assert_close(pos_fabric_after, new_positions, atol=1e-4, rtol=0)
    torch.testing.assert_close(quat_fabric_after, new_orientations, atol=1e-4, rtol=0)
