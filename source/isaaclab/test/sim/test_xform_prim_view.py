# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch

import pytest

try:
    from isaacsim.core.prims import XFormPrim as _IsaacSimXFormPrimView
except (ModuleNotFoundError, ImportError):
    _IsaacSimXFormPrimView = None

import isaaclab.sim as sim_utils
from isaaclab.sim.views import XFormPrimView as XFormPrimView


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


def assert_tensors_close(
    t1: torch.Tensor, t2: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-5, check_quat_sign: bool = False
):
    """Assert two tensors are close, with optional quaternion sign handling."""
    if check_quat_sign:
        # For quaternions, q and -q represent the same rotation
        # Try both normal and sign-flipped comparison
        try:
            torch.testing.assert_close(t1, t2, rtol=rtol, atol=atol)
        except AssertionError:
            # Try with sign flipped
            torch.testing.assert_close(t1, -t2, rtol=rtol, atol=atol)
    else:
        torch.testing.assert_close(t1, t2, rtol=rtol, atol=atol)


"""
Tests - Initialization.
"""


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_xform_prim_view_initialization_single_prim(device):
    """Test XFormPrimView initialization with a single prim."""
    # check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create a single xform prim
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/Object", "Xform", translation=(1.0, 2.0, 3.0), stage=stage)

    # Create view
    view = XFormPrimView("/World/Object", device=device)

    # Verify properties
    assert view.count == 1
    assert view.prim_path == "/World/Object"
    assert view.device == device
    assert len(view.prims) == 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_xform_prim_view_initialization_multiple_prims(device):
    """Test XFormPrimView initialization with multiple prims using pattern matching."""
    # check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create multiple prims
    num_prims = 10
    stage = sim_utils.get_current_stage()
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Env_{i}/Object", "Xform", translation=(i * 2.0, 0.0, 1.0), stage=stage)

    # Create view with pattern
    view = XFormPrimView("/World/Env_.*/Object", device=device)

    # Verify properties
    assert view.count == num_prims
    assert view.device == device
    assert len(view.prims) == num_prims


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_xform_prim_view_initialization_invalid_prim(device):
    """Test XFormPrimView initialization fails for non-xformable prims."""
    # check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Create a prim with non-standard xform operations
    stage.DefinePrim("/World/InvalidPrim", "Xform")

    # XFormPrimView should raise ValueError because prim doesn't have standard operations
    with pytest.raises(ValueError, match="not a xformable prim"):
        XFormPrimView("/World/InvalidPrim", device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_xform_prim_view_initialization_empty_pattern(device):
    """Test XFormPrimView initialization with pattern that matches no prims."""
    # check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    sim_utils.create_new_stage()

    # Create view with pattern that matches nothing
    view = XFormPrimView("/World/NonExistent_.*", device=device)

    # Should have zero count
    assert view.count == 0
    assert len(view.prims) == 0


"""
Tests - Get/Set World Poses.
"""


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_get_world_poses(device):
    """Test getting world poses from XFormPrimView."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Create prims with known world poses
    expected_positions = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
    expected_orientations = [(1.0, 0.0, 0.0, 0.0), (0.7071068, 0.0, 0.0, 0.7071068), (0.7071068, 0.7071068, 0.0, 0.0)]

    for i, (pos, quat) in enumerate(zip(expected_positions, expected_orientations)):
        sim_utils.create_prim(f"/World/Object_{i}", "Xform", translation=pos, orientation=quat, stage=stage)

    # Create view
    view = XFormPrimView("/World/Object_.*", device=device)

    # Get world poses
    positions, orientations = view.get_world_poses()

    # Verify shapes
    assert positions.shape == (3, 3)
    assert orientations.shape == (3, 4)

    # Convert expected values to tensors
    expected_positions_tensor = torch.tensor(expected_positions, dtype=torch.float32, device=device)
    expected_orientations_tensor = torch.tensor(expected_orientations, dtype=torch.float32, device=device)

    # Verify positions
    torch.testing.assert_close(positions, expected_positions_tensor, atol=1e-5, rtol=0)

    # Verify orientations (allow for quaternion sign ambiguity)
    try:
        torch.testing.assert_close(orientations, expected_orientations_tensor, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(orientations, -expected_orientations_tensor, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_set_world_poses(device):
    """Test setting world poses in XFormPrimView."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Create prims
    num_prims = 5
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Object_{i}", "Xform", translation=(0.0, 0.0, 0.0), stage=stage)

    # Create view
    view = XFormPrimView("/World/Object_.*", device=device)

    # Set new world poses
    new_positions = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]], device=device
    )
    new_orientations = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.7071068, 0.0, 0.0, 0.7071068],
            [0.7071068, 0.7071068, 0.0, 0.0],
            [0.9238795, 0.3826834, 0.0, 0.0],
            [0.7071068, 0.0, 0.7071068, 0.0],
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
def test_set_world_poses_only_positions(device):
    """Test setting only positions, leaving orientations unchanged."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Create prims with specific orientations
    initial_quat = (0.7071068, 0.0, 0.0, 0.7071068)  # 90 deg around Z
    for i in range(3):
        sim_utils.create_prim(
            f"/World/Object_{i}", "Xform", translation=(0.0, 0.0, 0.0), orientation=initial_quat, stage=stage
        )

    # Create view
    view = XFormPrimView("/World/Object_.*", device=device)

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
def test_set_world_poses_only_orientations(device):
    """Test setting only orientations, leaving positions unchanged."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Create prims with specific positions
    for i in range(3):
        sim_utils.create_prim(f"/World/Object_{i}", "Xform", translation=(float(i), 0.0, 0.0), stage=stage)

    # Create view
    view = XFormPrimView("/World/Object_.*", device=device)

    # Get initial positions
    initial_positions, _ = view.get_world_poses()

    # Set only orientations
    new_orientations = torch.tensor(
        [[0.7071068, 0.0, 0.0, 0.7071068], [0.7071068, 0.7071068, 0.0, 0.0], [0.9238795, 0.3826834, 0.0, 0.0]],
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
def test_set_world_poses_with_hierarchy(device):
    """Test setting world poses correctly handles parent transformations."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Create parent prims
    for i in range(3):
        parent_pos = (i * 10.0, 0.0, 0.0)
        parent_quat = (0.7071068, 0.0, 0.0, 0.7071068)  # 90 deg around Z
        sim_utils.create_prim(
            f"/World/Parent_{i}", "Xform", translation=parent_pos, orientation=parent_quat, stage=stage
        )
        # Create child prims
        sim_utils.create_prim(f"/World/Parent_{i}/Child", "Xform", translation=(0.0, 0.0, 0.0), stage=stage)

    # Create view for children
    view = XFormPrimView("/World/Parent_.*/Child", device=device)

    # Set world poses for children
    desired_world_positions = torch.tensor([[5.0, 5.0, 0.0], [15.0, 5.0, 0.0], [25.0, 5.0, 0.0]], device=device)
    desired_world_orientations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device
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


"""
Tests - Get/Set Local Poses.
"""


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_get_local_poses(device):
    """Test getting local poses from XFormPrimView."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Create parent and child prims
    sim_utils.create_prim("/World/Parent", "Xform", translation=(10.0, 0.0, 0.0), stage=stage)

    # Children with different local poses
    expected_local_positions = [(1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0)]
    expected_local_orientations = [
        (1.0, 0.0, 0.0, 0.0),
        (0.7071068, 0.0, 0.0, 0.7071068),
        (0.7071068, 0.7071068, 0.0, 0.0),
    ]

    for i, (pos, quat) in enumerate(zip(expected_local_positions, expected_local_orientations)):
        sim_utils.create_prim(f"/World/Parent/Child_{i}", "Xform", translation=pos, orientation=quat, stage=stage)

    # Create view
    view = XFormPrimView("/World/Parent/Child_.*", device=device)

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
def test_set_local_poses(device):
    """Test setting local poses in XFormPrimView."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Create parent
    sim_utils.create_prim("/World/Parent", "Xform", translation=(5.0, 5.0, 5.0), stage=stage)

    # Create children
    num_prims = 4
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Parent/Child_{i}", "Xform", translation=(0.0, 0.0, 0.0), stage=stage)

    # Create view
    view = XFormPrimView("/World/Parent/Child_.*", device=device)

    # Set new local poses
    new_translations = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [4.0, 4.0, 4.0]], device=device)
    new_orientations = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.7071068, 0.0, 0.0, 0.7071068],
            [0.7071068, 0.7071068, 0.0, 0.0],
            [0.9238795, 0.3826834, 0.0, 0.0],
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
def test_set_local_poses_only_translations(device):
    """Test setting only local translations."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Create parent and children with specific orientations
    sim_utils.create_prim("/World/Parent", "Xform", translation=(0.0, 0.0, 0.0), stage=stage)
    initial_quat = (0.7071068, 0.0, 0.0, 0.7071068)

    for i in range(3):
        sim_utils.create_prim(
            f"/World/Parent/Child_{i}", "Xform", translation=(0.0, 0.0, 0.0), orientation=initial_quat, stage=stage
        )

    # Create view
    view = XFormPrimView("/World/Parent/Child_.*", device=device)

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


"""
Tests - Get/Set Scales.
"""


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_get_scales(device):
    """Test getting scales from XFormPrimView."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Create prims with different scales
    expected_scales = [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (1.0, 2.0, 3.0)]

    for i, scale in enumerate(expected_scales):
        sim_utils.create_prim(f"/World/Object_{i}", "Xform", scale=scale, stage=stage)

    # Create view
    view = XFormPrimView("/World/Object_.*", device=device)

    # Get scales
    scales = view.get_scales()

    # Verify shape and values
    assert scales.shape == (3, 3)
    expected_scales_tensor = torch.tensor(expected_scales, dtype=torch.float32, device=device)
    torch.testing.assert_close(scales, expected_scales_tensor, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_set_scales(device):
    """Test setting scales in XFormPrimView."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Create prims
    num_prims = 5
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Object_{i}", "Xform", scale=(1.0, 1.0, 1.0), stage=stage)

    # Create view
    view = XFormPrimView("/World/Object_.*", device=device)

    # Set new scales
    new_scales = torch.tensor(
        [[2.0, 2.0, 2.0], [1.0, 2.0, 3.0], [0.5, 0.5, 0.5], [3.0, 1.0, 2.0], [1.5, 1.5, 1.5]], device=device
    )

    view.set_scales(new_scales)

    # Get scales back
    retrieved_scales = view.get_scales()

    # Verify they match
    torch.testing.assert_close(retrieved_scales, new_scales, atol=1e-5, rtol=0)


"""
Tests - Comparison with Isaac Sim Implementation.
"""


def test_compare_get_world_poses_with_isaacsim():
    """Compare get_world_poses with Isaac Sim's implementation."""
    stage = sim_utils.get_current_stage()

    # Check if Isaac Sim is available
    if _IsaacSimXFormPrimView is None:
        pytest.skip("Isaac Sim is not available")

    # Create prims with various poses
    num_prims = 10
    for i in range(num_prims):
        pos = (i * 2.0, i * 0.5, i * 1.5)
        # Vary orientations
        if i % 3 == 0:
            quat = (1.0, 0.0, 0.0, 0.0)  # Identity
        elif i % 3 == 1:
            quat = (0.7071068, 0.0, 0.0, 0.7071068)  # 90 deg around Z
        else:
            quat = (0.7071068, 0.7071068, 0.0, 0.0)  # 90 deg around X
        sim_utils.create_prim(f"/World/Env_{i}/Object", "Xform", translation=pos, orientation=quat, stage=stage)

    pattern = "/World/Env_.*/Object"

    # Create both views
    isaaclab_view = XFormPrimView(pattern, device="cpu")
    isaacsim_view = _IsaacSimXFormPrimView(pattern, reset_xform_properties=False)

    # Get world poses from both
    isaaclab_pos, isaaclab_quat = isaaclab_view.get_world_poses()
    isaacsim_pos, isaacsim_quat = isaacsim_view.get_world_poses()

    # Convert Isaac Sim results to torch tensors if needed
    if not isinstance(isaacsim_pos, torch.Tensor):
        isaacsim_pos = torch.tensor(isaacsim_pos, dtype=torch.float32)
    if not isinstance(isaacsim_quat, torch.Tensor):
        isaacsim_quat = torch.tensor(isaacsim_quat, dtype=torch.float32)

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
    if _IsaacSimXFormPrimView is None:
        pytest.skip("Isaac Sim is not available")

    # Create prims
    num_prims = 8
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Env_{i}/Object", "Xform", translation=(0.0, 0.0, 0.0), stage=stage)

    pattern = "/World/Env_.*/Object"

    # Create both views
    isaaclab_view = XFormPrimView(pattern, device="cpu")
    isaacsim_view = _IsaacSimXFormPrimView(pattern, reset_xform_properties=False)

    # Generate new poses
    new_positions = torch.randn(num_prims, 3) * 10.0
    new_orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_prims, dtype=torch.float32)

    # Set poses using both implementations
    isaaclab_view.set_world_poses(new_positions.clone(), new_orientations.clone())
    isaacsim_view.set_world_poses(new_positions.clone(), new_orientations.clone())

    # Get poses back from both
    isaaclab_pos, isaaclab_quat = isaaclab_view.get_world_poses()
    isaacsim_pos, isaacsim_quat = isaacsim_view.get_world_poses()

    # Convert Isaac Sim results to torch tensors if needed
    if not isinstance(isaacsim_pos, torch.Tensor):
        isaacsim_pos = torch.tensor(isaacsim_pos, dtype=torch.float32)
    if not isinstance(isaacsim_quat, torch.Tensor):
        isaacsim_quat = torch.tensor(isaacsim_quat, dtype=torch.float32)

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
    if _IsaacSimXFormPrimView is None:
        pytest.skip("Isaac Sim is not available")

    # Create hierarchical prims
    num_prims = 5
    for i in range(num_prims):
        # Create parent
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 5.0, 0.0, 0.0), stage=stage)
        # Create child with local pose
        local_pos = (1.0, float(i), 0.0)
        local_quat = (1.0, 0.0, 0.0, 0.0) if i % 2 == 0 else (0.7071068, 0.0, 0.0, 0.7071068)
        sim_utils.create_prim(
            f"/World/Env_{i}/Object", "Xform", translation=local_pos, orientation=local_quat, stage=stage
        )

    pattern = "/World/Env_.*/Object"

    # Create both views
    isaaclab_view = XFormPrimView(pattern, device="cpu")
    isaacsim_view = _IsaacSimXFormPrimView(pattern, reset_xform_properties=False)

    # Get local poses from both
    isaaclab_trans, isaaclab_quat = isaaclab_view.get_local_poses()
    isaacsim_trans, isaacsim_quat = isaacsim_view.get_local_poses()

    # Convert Isaac Sim results to torch tensors if needed
    if not isinstance(isaacsim_trans, torch.Tensor):
        isaacsim_trans = torch.tensor(isaacsim_trans, dtype=torch.float32)
    if not isinstance(isaacsim_quat, torch.Tensor):
        isaacsim_quat = torch.tensor(isaacsim_quat, dtype=torch.float32)

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
    if _IsaacSimXFormPrimView is None:
        pytest.skip("Isaac Sim is not available")

    # Create hierarchical prims
    num_prims = 6
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 3.0, 0.0, 0.0), stage=stage)
        sim_utils.create_prim(f"/World/Env_{i}/Object", "Xform", translation=(0.0, 0.0, 0.0), stage=stage)

    pattern = "/World/Env_.*/Object"

    # Create both views
    isaaclab_view = XFormPrimView(pattern, device="cpu")
    isaacsim_view = _IsaacSimXFormPrimView(pattern, reset_xform_properties=False)

    # Generate new local poses
    new_translations = torch.randn(num_prims, 3) * 5.0
    new_orientations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.7071068, 0.0, 0.0, 0.7071068]] * (num_prims // 2), dtype=torch.float32
    )

    # Set local poses using both implementations
    isaaclab_view.set_local_poses(new_translations.clone(), new_orientations.clone())
    isaacsim_view.set_local_poses(new_translations.clone(), new_orientations.clone())

    # Get local poses back from both
    isaaclab_trans, isaaclab_quat = isaaclab_view.get_local_poses()
    isaacsim_trans, isaacsim_quat = isaacsim_view.get_local_poses()

    # Convert Isaac Sim results to torch tensors if needed
    if not isinstance(isaacsim_trans, torch.Tensor):
        isaacsim_trans = torch.tensor(isaacsim_trans, dtype=torch.float32)
    if not isinstance(isaacsim_quat, torch.Tensor):
        isaacsim_quat = torch.tensor(isaacsim_quat, dtype=torch.float32)

    # Compare results
    torch.testing.assert_close(isaaclab_trans, isaacsim_trans, atol=1e-4, rtol=0)
    try:
        torch.testing.assert_close(isaaclab_quat, isaacsim_quat, atol=1e-4, rtol=0)
    except AssertionError:
        torch.testing.assert_close(isaaclab_quat, -isaacsim_quat, atol=1e-4, rtol=0)


"""
Tests - Complex Scenarios.
"""


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_complex_hierarchy_world_local_consistency(device):
    """Test that world and local poses are consistent in complex hierarchies."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stage = sim_utils.get_current_stage()

    # Create complex hierarchy: Grandparent -> Parent -> Child
    num_envs = 3
    for i in range(num_envs):
        # Grandparent
        sim_utils.create_prim(
            f"/World/Grandparent_{i}",
            "Xform",
            translation=(i * 20.0, 0.0, 0.0),
            orientation=(0.7071068, 0.0, 0.0, 0.7071068),
            scale=(2.0, 2.0, 2.0),
            stage=stage,
        )
        # Parent
        sim_utils.create_prim(
            f"/World/Grandparent_{i}/Parent",
            "Xform",
            translation=(5.0, 0.0, 0.0),
            orientation=(0.7071068, 0.7071068, 0.0, 0.0),
            stage=stage,
        )
        # Child
        sim_utils.create_prim(f"/World/Grandparent_{i}/Parent/Child", "Xform", translation=(1.0, 2.0, 3.0), stage=stage)

    # Create view for children
    view = XFormPrimView("/World/Grandparent_.*/Parent/Child", device=device)

    # Get world and local poses
    world_pos, world_quat = view.get_world_poses()
    local_trans, local_quat = view.get_local_poses()

    # Change local poses
    new_local_trans = torch.tensor([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]], device=device)
    view.set_local_poses(translations=new_local_trans, orientations=None)

    # Get world poses again
    new_world_pos, new_world_quat = view.get_world_poses()

    # World poses should have changed when local poses changed
    assert not torch.allclose(world_pos, new_world_pos, atol=1e-5)

    # Now set world poses back to original
    view.set_world_poses(world_pos, world_quat)

    # Get world poses again
    final_world_pos, final_world_quat = view.get_world_poses()

    # Should match original world poses
    torch.testing.assert_close(final_world_pos, world_pos, atol=1e-4, rtol=0)
