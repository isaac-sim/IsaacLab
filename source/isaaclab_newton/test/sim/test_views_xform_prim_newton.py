# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton XformPrimView backed by sites (body_q + local transform).

Covers the same test surface as the core USD and PhysX Fabric tests where
applicable.  Tests that require USD hierarchy features (local poses with
parents, visibility) are skipped since Newton tracks bodies in world space.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import warp as wp

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics.newton_manager import NewtonManager
from isaaclab_newton.sim.views import XformPrimView

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils import configclass

NEWTON_SIM_CFG = SimulationCfg(
    physics=NewtonCfg(
        solver_cfg=MJWarpSolverCfg(),
    ),
)


@configclass
class SimpleSceneCfg(InteractiveSceneCfg):
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


def _newton_sim_context(device="cuda:0", num_envs=4, **kwargs):
    NEWTON_SIM_CFG.device = device
    return build_simulation_context(device=device, sim_cfg=NEWTON_SIM_CFG, **kwargs)


def _build_scene(num_envs=4, device="cuda:0"):
    """Build scene and return (sim, scene, view)."""
    ctx = _newton_sim_context(device=device, num_envs=num_envs, add_ground_plane=True)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    scene = InteractiveScene(SimpleSceneCfg(num_envs=num_envs, env_spacing=2.0))
    sim.reset()
    view = XformPrimView("/World/envs/env_.*/Cube", device=device)
    return sim, scene, view, ctx


"""
Tests - Initialization.
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_count():
    """Test that Newton XformPrimView reports correct prim count."""
    sim, scene, view, ctx = _build_scene(num_envs=4)
    assert view.count == 4
    ctx.__exit__(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("num_envs", [1, 4, 8])
def test_count_various_sizes(num_envs):
    """Test count with different numbers of environments."""
    sim, scene, view, ctx = _build_scene(num_envs=num_envs)
    assert view.count == num_envs
    ctx.__exit__(None, None, None)


"""
Tests - Getters.
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_world_poses_shape():
    """Test that get_world_poses returns correct shapes."""
    sim, scene, view, ctx = _build_scene(num_envs=4)
    positions, orientations = view.get_world_poses()

    assert positions.shape == (4, 3)
    assert orientations.shape == (4, 4)
    assert positions.device.type == "cuda"
    ctx.__exit__(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_world_poses_matches_body_q():
    """Test that get_world_poses returns values matching body_q."""
    sim, scene, view, ctx = _build_scene(num_envs=4)
    positions, orientations = view.get_world_poses()

    model = NewtonManager.get_model()
    body_labels = list(model.body_label)
    state = NewtonManager.get_state_0()
    body_q = wp.to_torch(state.body_q)

    for i in range(4):
        prim_path = f"/World/envs/env_{i}/Cube"
        if prim_path in body_labels:
            body_idx = body_labels.index(prim_path)
            expected_pos = body_q[body_idx, :3]
            torch.testing.assert_close(positions[i], expected_pos, atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_world_poses_with_indices():
    """Test get_world_poses with a subset of indices."""
    sim, scene, view, ctx = _build_scene(num_envs=4)
    positions_all, orientations_all = view.get_world_poses()

    indices = [0, 2]
    positions_subset, orientations_subset = view.get_world_poses(indices)

    assert positions_subset.shape == (2, 3)
    assert orientations_subset.shape == (2, 4)
    torch.testing.assert_close(positions_subset[0], positions_all[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(positions_subset[1], positions_all[2], atol=1e-6, rtol=0)
    ctx.__exit__(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_world_poses_single_index():
    """Test get_world_poses with a single index."""
    sim, scene, view, ctx = _build_scene(num_envs=5)
    positions_all, _ = view.get_world_poses()

    positions_single, orientations_single = view.get_world_poses([3])
    assert positions_single.shape == (1, 3)
    assert orientations_single.shape == (1, 4)
    torch.testing.assert_close(positions_single[0], positions_all[3], atol=1e-6, rtol=0)
    ctx.__exit__(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_world_poses_out_of_order_indices():
    """Test get_world_poses with out-of-order indices."""
    sim, scene, view, ctx = _build_scene(num_envs=5)
    positions_all, _ = view.get_world_poses()

    indices = [4, 1, 3]
    positions_subset, _ = view.get_world_poses(indices)

    assert positions_subset.shape == (3, 3)
    torch.testing.assert_close(positions_subset[0], positions_all[4], atol=1e-6, rtol=0)
    torch.testing.assert_close(positions_subset[1], positions_all[1], atol=1e-6, rtol=0)
    torch.testing.assert_close(positions_subset[2], positions_all[3], atol=1e-6, rtol=0)
    ctx.__exit__(None, None, None)


"""
Tests - Setters.
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_world_poses():
    """Test setting world poses writes to body_q."""
    sim, scene, view, ctx = _build_scene(num_envs=4)

    new_positions = torch.tensor(
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0], [100.0, 110.0, 120.0]],
        device="cuda:0",
    )
    new_orientations = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.7071068, 0.7071068],
         [0.7071068, 0.0, 0.0, 0.7071068], [0.3826834, 0.0, 0.0, 0.9238795]],
        device="cuda:0",
    )

    view.set_world_poses(new_positions, new_orientations)
    retrieved_positions, retrieved_orientations = view.get_world_poses()

    torch.testing.assert_close(retrieved_positions, new_positions, atol=1e-5, rtol=0)
    torch.testing.assert_close(retrieved_orientations, new_orientations, atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_world_poses_only_positions():
    """Test setting only positions, leaving orientations unchanged."""
    sim, scene, view, ctx = _build_scene(num_envs=3)
    _, initial_orientations = view.get_world_poses()

    new_positions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], device="cuda:0")
    view.set_world_poses(positions=new_positions, orientations=None)

    retrieved_positions, retrieved_orientations = view.get_world_poses()
    torch.testing.assert_close(retrieved_positions, new_positions, atol=1e-5, rtol=0)
    torch.testing.assert_close(retrieved_orientations, initial_orientations, atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_world_poses_only_orientations():
    """Test setting only orientations, leaving positions unchanged."""
    sim, scene, view, ctx = _build_scene(num_envs=3)
    initial_positions, _ = view.get_world_poses()

    new_orientations = torch.tensor(
        [[0.0, 0.0, 0.7071068, 0.7071068], [0.7071068, 0.0, 0.0, 0.7071068], [0.3826834, 0.0, 0.0, 0.9238795]],
        device="cuda:0",
    )
    view.set_world_poses(positions=None, orientations=new_orientations)

    retrieved_positions, retrieved_orientations = view.get_world_poses()
    torch.testing.assert_close(retrieved_positions, initial_positions, atol=1e-5, rtol=0)
    torch.testing.assert_close(retrieved_orientations, new_orientations, atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_world_poses_with_indices():
    """Test setting world poses for a subset of indices."""
    sim, scene, view, ctx = _build_scene(num_envs=5)
    initial_positions, initial_orientations = view.get_world_poses()

    indices = [1, 3]
    new_positions = torch.tensor([[10.0, 0.0, 0.0], [30.0, 0.0, 0.0]], device="cuda:0")
    view.set_world_poses(positions=new_positions, orientations=None, indices=indices)

    updated_positions, _ = view.get_world_poses()

    torch.testing.assert_close(updated_positions[1], new_positions[0], atol=1e-5, rtol=0)
    torch.testing.assert_close(updated_positions[3], new_positions[1], atol=1e-5, rtol=0)
    torch.testing.assert_close(updated_positions[0], initial_positions[0], atol=1e-5, rtol=0)
    torch.testing.assert_close(updated_positions[2], initial_positions[2], atol=1e-5, rtol=0)
    torch.testing.assert_close(updated_positions[4], initial_positions[4], atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


"""
Tests - Round-trip consistency.
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_write_read_roundtrip():
    """Test that write -> read round-trip is consistent."""
    sim, scene, view, ctx = _build_scene(num_envs=4)

    new_positions = torch.rand((4, 3), dtype=torch.float32, device="cuda:0") * 10.0
    new_orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 4, dtype=torch.float32, device="cuda:0")

    view.set_world_poses(new_positions, new_orientations)
    pos, quat = view.get_world_poses()

    torch.testing.assert_close(pos, new_positions, atol=1e-5, rtol=0)
    torch.testing.assert_close(quat, new_orientations, atol=1e-5, rtol=0)

    new_positions2 = torch.rand((4, 3), dtype=torch.float32, device="cuda:0") * 10.0
    view.set_world_poses(new_positions2, new_orientations)
    pos2, quat2 = view.get_world_poses()

    torch.testing.assert_close(pos2, new_positions2, atol=1e-5, rtol=0)
    torch.testing.assert_close(quat2, new_orientations, atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)
