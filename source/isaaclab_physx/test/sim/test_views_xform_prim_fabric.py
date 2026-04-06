# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PhysX XformPrimView with Fabric acceleration.

Re-runs all the backend-parametrized tests from the core test suite with
``backend="fabric"`` and the PhysX :class:`XformPrimView`, plus the two
dedicated Fabric-only tests.
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab_physx.sim.views import XformPrimView  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

BACKEND = "fabric"


@pytest.fixture(autouse=True)
def test_setup_teardown():
    """Create a blank new stage for each test."""
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


def _create_view(pattern: str, device: str) -> XformPrimView:
    sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=device, use_fabric=True))
    return XformPrimView(pattern, device=device)


"""
Tests - Getters.
"""


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_get_world_poses(device):
    _skip_if_unavailable(device)
    stage = sim_utils.get_current_stage()

    expected_positions = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
    expected_orientations = [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.7071068, 0.7071068), (0.7071068, 0.0, 0.0, 0.7071068)]

    for i, (pos, quat) in enumerate(zip(expected_positions, expected_orientations)):
        sim_utils.create_prim(f"/World/Object_{i}", "Camera", translation=pos, orientation=quat, stage=stage)

    view = _create_view("/World/Object_.*", device=device)

    expected_positions_tensor = torch.tensor(expected_positions, dtype=torch.float32, device=device)
    expected_orientations_tensor = torch.tensor(expected_orientations, dtype=torch.float32, device=device)

    positions, orientations = view.get_world_poses()
    positions, orientations = wp.to_torch(positions), wp.to_torch(orientations)

    assert positions.shape == (3, 3)
    assert orientations.shape == (3, 4)
    torch.testing.assert_close(positions, expected_positions_tensor, atol=1e-5, rtol=0)

    try:
        torch.testing.assert_close(orientations, expected_orientations_tensor, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(orientations, -expected_orientations_tensor, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_get_local_poses(device):
    _skip_if_unavailable(device)
    stage = sim_utils.get_current_stage()

    sim_utils.create_prim("/World/Parent", "Xform", translation=(10.0, 0.0, 0.0), stage=stage)

    expected_local_positions = [(1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0)]
    expected_local_orientations = [
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.7071068, 0.7071068),
        (0.7071068, 0.0, 0.0, 0.7071068),
    ]

    for i, (pos, quat) in enumerate(zip(expected_local_positions, expected_local_orientations)):
        sim_utils.create_prim(f"/World/Parent/Child_{i}", "Camera", translation=pos, orientation=quat, stage=stage)

    view = _create_view("/World/Parent/Child_.*", device=device)
    translations, orientations = view.get_local_poses()
    translations, orientations = wp.to_torch(translations), wp.to_torch(orientations)

    assert translations.shape == (3, 3)
    assert orientations.shape == (3, 4)

    expected_translations_tensor = torch.tensor(expected_local_positions, dtype=torch.float32, device=device)
    expected_orientations_tensor = torch.tensor(expected_local_orientations, dtype=torch.float32, device=device)

    torch.testing.assert_close(translations, expected_translations_tensor, atol=1e-5, rtol=0)
    try:
        torch.testing.assert_close(orientations, expected_orientations_tensor, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(orientations, -expected_orientations_tensor, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_get_scales(device):
    _skip_if_unavailable(device)
    stage = sim_utils.get_current_stage()

    expected_scales = [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (1.0, 2.0, 3.0)]
    for i, scale in enumerate(expected_scales):
        sim_utils.create_prim(f"/World/Object_{i}", "Camera", scale=scale, stage=stage)

    view = _create_view("/World/Object_.*", device=device)
    scales = wp.to_torch(view.get_scales())

    assert scales.shape == (3, 3)
    torch.testing.assert_close(
        scales, torch.tensor(expected_scales, dtype=torch.float32, device=device), atol=1e-5, rtol=0
    )


"""
Tests - Setters.
"""


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_set_world_poses(device):
    _skip_if_unavailable(device)
    stage = sim_utils.get_current_stage()

    num_prims = 5
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Object_{i}", "Camera", translation=(0.0, 0.0, 0.0), stage=stage)

    view = _create_view("/World/Object_.*", device=device)

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
    retrieved_positions, retrieved_orientations = view.get_world_poses()
    retrieved_positions, retrieved_orientations = wp.to_torch(retrieved_positions), wp.to_torch(retrieved_orientations)

    torch.testing.assert_close(retrieved_positions, new_positions, atol=1e-5, rtol=0)
    try:
        torch.testing.assert_close(retrieved_orientations, new_orientations, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(retrieved_orientations, -new_orientations, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_set_world_poses_only_positions(device):
    _skip_if_unavailable(device)
    stage = sim_utils.get_current_stage()

    initial_quat = (0.0, 0.0, 0.7071068, 0.7071068)
    for i in range(3):
        sim_utils.create_prim(
            f"/World/Object_{i}", "Camera", translation=(0.0, 0.0, 0.0), orientation=initial_quat, stage=stage
        )

    view = _create_view("/World/Object_.*", device=device)
    _, initial_orientations = view.get_world_poses()
    initial_orientations = wp.to_torch(initial_orientations)

    new_positions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], device=device)
    view.set_world_poses(positions=new_positions, orientations=None)

    retrieved_positions, retrieved_orientations = view.get_world_poses()
    retrieved_positions, retrieved_orientations = wp.to_torch(retrieved_positions), wp.to_torch(retrieved_orientations)
    torch.testing.assert_close(retrieved_positions, new_positions, atol=1e-5, rtol=0)
    try:
        torch.testing.assert_close(retrieved_orientations, initial_orientations, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(retrieved_orientations, -initial_orientations, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_set_world_poses_only_orientations(device):
    _skip_if_unavailable(device)
    stage = sim_utils.get_current_stage()

    for i in range(3):
        sim_utils.create_prim(f"/World/Object_{i}", "Camera", translation=(float(i), 0.0, 0.0), stage=stage)

    view = _create_view("/World/Object_.*", device=device)
    initial_positions = wp.to_torch(view.get_world_poses()[0])

    new_orientations = torch.tensor(
        [[0.0, 0.0, 0.7071068, 0.7071068], [0.7071068, 0.0, 0.0, 0.7071068], [0.3826834, 0.0, 0.0, 0.9238795]],
        device=device,
    )
    view.set_world_poses(positions=None, orientations=new_orientations)

    retrieved_positions, retrieved_orientations = view.get_world_poses()
    retrieved_positions, retrieved_orientations = wp.to_torch(retrieved_positions), wp.to_torch(retrieved_orientations)
    torch.testing.assert_close(retrieved_positions, initial_positions, atol=1e-5, rtol=0)
    try:
        torch.testing.assert_close(retrieved_orientations, new_orientations, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(retrieved_orientations, -new_orientations, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_set_world_poses_with_hierarchy(device):
    _skip_if_unavailable(device)
    stage = sim_utils.get_current_stage()

    for i in range(3):
        parent_pos = (i * 10.0, 0.0, 0.0)
        parent_quat = (0.0, 0.0, 0.7071068, 0.7071068)
        sim_utils.create_prim(
            f"/World/Parent_{i}", "Xform", translation=parent_pos, orientation=parent_quat, stage=stage
        )
        sim_utils.create_prim(f"/World/Parent_{i}/Child", "Camera", translation=(0.0, 0.0, 0.0), stage=stage)

    view = _create_view("/World/Parent_.*/Child", device=device)

    desired_world_positions = torch.tensor([[5.0, 5.0, 0.0], [15.0, 5.0, 0.0], [25.0, 5.0, 0.0]], device=device)
    desired_world_orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3, device=device)

    view.set_world_poses(desired_world_positions, desired_world_orientations)
    retrieved_positions, retrieved_orientations = view.get_world_poses()
    retrieved_positions, retrieved_orientations = wp.to_torch(retrieved_positions), wp.to_torch(retrieved_orientations)

    torch.testing.assert_close(retrieved_positions, desired_world_positions, atol=1e-4, rtol=0)
    try:
        torch.testing.assert_close(retrieved_orientations, desired_world_orientations, atol=1e-4, rtol=0)
    except AssertionError:
        torch.testing.assert_close(retrieved_orientations, -desired_world_orientations, atol=1e-4, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_set_local_poses(device):
    _skip_if_unavailable(device)
    stage = sim_utils.get_current_stage()

    sim_utils.create_prim("/World/Parent", "Xform", translation=(5.0, 5.0, 5.0), stage=stage)
    num_prims = 4
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Parent/Child_{i}", "Camera", translation=(0.0, 0.0, 0.0), stage=stage)

    view = _create_view("/World/Parent/Child_.*", device=device)

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
    retrieved_translations, retrieved_orientations = view.get_local_poses()
    retrieved_translations, retrieved_orientations = (
        wp.to_torch(retrieved_translations),
        wp.to_torch(retrieved_orientations),
    )

    torch.testing.assert_close(retrieved_translations, new_translations, atol=1e-5, rtol=0)
    try:
        torch.testing.assert_close(retrieved_orientations, new_orientations, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(retrieved_orientations, -new_orientations, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_set_local_poses_only_translations(device):
    _skip_if_unavailable(device)
    stage = sim_utils.get_current_stage()

    sim_utils.create_prim("/World/Parent", "Xform", translation=(0.0, 0.0, 0.0), stage=stage)
    initial_quat = (0.0, 0.0, 0.7071068, 0.7071068)
    for i in range(3):
        sim_utils.create_prim(
            f"/World/Parent/Child_{i}", "Camera", translation=(0.0, 0.0, 0.0), orientation=initial_quat, stage=stage
        )

    view = _create_view("/World/Parent/Child_.*", device=device)
    initial_orientations = wp.to_torch(view.get_local_poses()[1])

    new_translations = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], device=device)
    view.set_local_poses(translations=new_translations, orientations=None)

    retrieved_translations, retrieved_orientations = view.get_local_poses()
    retrieved_translations, retrieved_orientations = (
        wp.to_torch(retrieved_translations),
        wp.to_torch(retrieved_orientations),
    )
    torch.testing.assert_close(retrieved_translations, new_translations, atol=1e-5, rtol=0)
    try:
        torch.testing.assert_close(retrieved_orientations, initial_orientations, atol=1e-5, rtol=0)
    except AssertionError:
        torch.testing.assert_close(retrieved_orientations, -initial_orientations, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_set_scales(device):
    _skip_if_unavailable(device)
    stage = sim_utils.get_current_stage()

    num_prims = 5
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Object_{i}", "Camera", scale=(1.0, 1.0, 1.0), stage=stage)

    view = _create_view("/World/Object_.*", device=device)
    new_scales = torch.tensor(
        [[2.0, 2.0, 2.0], [1.0, 2.0, 3.0], [0.5, 0.5, 0.5], [3.0, 1.0, 2.0], [1.5, 1.5, 1.5]], device=device
    )
    view.set_scales(new_scales)

    retrieved_scales = wp.to_torch(view.get_scales())
    torch.testing.assert_close(retrieved_scales, new_scales, atol=1e-5, rtol=0)


"""
Tests - Indices.
"""


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_indices_single_element(device):
    _skip_if_unavailable(device)
    stage = sim_utils.get_current_stage()

    num_prims = 5
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Object_{i}", "Camera", translation=(float(i), 0.0, 0.0), stage=stage)

    view = _create_view("/World/Object_.*", device=device)

    indices = [3]
    positions, orientations = view.get_world_poses(indices=indices)
    positions, orientations = wp.to_torch(positions), wp.to_torch(orientations)
    assert positions.shape == (1, 3)
    assert orientations.shape == (1, 4)

    new_position = torch.tensor([[100.0, 200.0, 300.0]], device=device)
    view.set_world_poses(positions=new_position, indices=indices)

    retrieved_positions = wp.to_torch(view.get_world_poses(indices=indices)[0])
    torch.testing.assert_close(retrieved_positions, new_position, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_indices_out_of_order(device):
    _skip_if_unavailable(device)
    stage = sim_utils.get_current_stage()

    num_prims = 10
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Object_{i}", "Camera", translation=(0.0, 0.0, 0.0), stage=stage)

    view = _create_view("/World/Object_.*", device=device)

    indices = [7, 2, 9, 0, 5]
    new_positions = torch.tensor(
        [[7.0, 0.0, 0.0], [2.0, 0.0, 0.0], [9.0, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], device=device
    )
    view.set_world_poses(positions=new_positions, indices=indices)

    all_positions = wp.to_torch(view.get_world_poses()[0])

    expected_x_values = [0.0, 0.0, 2.0, 0.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0]
    for i in range(num_prims):
        assert abs(all_positions[i, 0].item() - expected_x_values[i]) < 1e-5


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_indices_with_only_positions_or_orientations(device):
    _skip_if_unavailable(device)
    stage = sim_utils.get_current_stage()

    num_prims = 5
    for i in range(num_prims):
        sim_utils.create_prim(
            f"/World/Object_{i}", "Camera", translation=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0), stage=stage
        )

    view = _create_view("/World/Object_.*", device=device)
    initial_positions = wp.to_torch(view.get_world_poses()[0]).clone()

    indices = [1, 3]
    new_positions = torch.tensor([[10.0, 0.0, 0.0], [30.0, 0.0, 0.0]], device=device)
    view.set_world_poses(positions=new_positions, orientations=None, indices=indices)

    updated_positions = wp.to_torch(view.get_world_poses()[0])

    torch.testing.assert_close(updated_positions[1], new_positions[0], atol=1e-5, rtol=0)
    torch.testing.assert_close(updated_positions[3], new_positions[1], atol=1e-5, rtol=0)
    torch.testing.assert_close(updated_positions[0], initial_positions[0], atol=1e-5, rtol=0)


"""
Tests - Fabric Operations.
"""


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fabric_initialization(device):
    _skip_if_unavailable(device)
    stage = sim_utils.get_current_stage()

    num_prims = 5
    for i in range(num_prims):
        sim_utils.create_prim(f"/World/Cam_{i}", "Camera", translation=(i * 1.0, 0.0, 1.0), stage=stage)

    view = _create_view("/World/Cam_.*", device=device)

    assert view.count == num_prims
    assert view.device == device
    assert len(view.prims) == num_prims


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fabric_usd_consistency(device):
    _skip_if_unavailable(device)
    stage = sim_utils.get_current_stage()

    num_prims = 5
    for i in range(num_prims):
        sim_utils.create_prim(
            f"/World/Cam_{i}",
            "Camera",
            translation=(i * 1.0, 2.0, 3.0),
            orientation=(0.0, 0.0, 0.7071068, 0.7071068),
            stage=stage,
        )

    view = _create_view("/World/Cam_.*", device=device)

    init_positions = torch.zeros((num_prims, 3), dtype=torch.float32, device=device)
    init_positions[:, 0] = torch.arange(num_prims, dtype=torch.float32, device=device)
    init_positions[:, 1] = 2.0
    init_positions[:, 2] = 3.0
    init_orientations = torch.tensor([[0.0, 0.0, 0.7071068, 0.7071068]] * num_prims, dtype=torch.float32, device=device)

    view.set_world_poses(init_positions, init_orientations)

    pos, quat = view.get_world_poses()
    torch.testing.assert_close(wp.to_torch(pos), init_positions, atol=1e-4, rtol=0)
    torch.testing.assert_close(wp.to_torch(quat), init_orientations, atol=1e-4, rtol=0)

    new_positions = torch.rand((num_prims, 3), dtype=torch.float32, device=device) * 10.0
    new_orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * num_prims, dtype=torch.float32, device=device)
    view.set_world_poses(new_positions, new_orientations)

    pos2, quat2 = view.get_world_poses()
    torch.testing.assert_close(wp.to_torch(pos2), new_positions, atol=1e-4, rtol=0)
    torch.testing.assert_close(wp.to_torch(quat2), new_orientations, atol=1e-4, rtol=0)
