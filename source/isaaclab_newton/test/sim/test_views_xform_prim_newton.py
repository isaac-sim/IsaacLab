# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton XformPrimView backed by sites (body_q + local transform).

Covers the same test surface as the core USD and PhysX Fabric tests where
applicable, plus Newton-specific scenarios: ancestor-walk resolution for
non-physics prims, world-attached (body=-1) prims, and hierarchical prims
(Xform child under a rigid body).
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

from pxr import Gf

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils import configclass

NEWTON_SIM_CFG = SimulationCfg(
    physics=NewtonCfg(
        solver_cfg=MJWarpSolverCfg(),
    ),
)


def _assert_close(a, b, **kwargs):
    """Compare two arrays (wp.array or torch.Tensor) via torch.testing.assert_close."""
    a_t = wp.to_torch(a) if isinstance(a, wp.array) else a
    b_t = wp.to_torch(b) if isinstance(b, wp.array) else b
    torch.testing.assert_close(a_t, b_t, **kwargs)


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
    """Build scene and return (sim, scene, view, ctx)."""
    ctx = _newton_sim_context(device=device, num_envs=num_envs, add_ground_plane=True)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    scene = InteractiveScene(SimpleSceneCfg(num_envs=num_envs, env_spacing=2.0))
    sim.reset()
    view = XformPrimView("/World/envs/env_.*/Cube", device=device)
    return sim, scene, view, ctx


def _build_scene_with_child_xform(num_envs=4, device="cuda:0"):
    """Build scene with a child Xform prim under each Cube (camera mount scenario)."""
    ctx = _newton_sim_context(device=device, num_envs=num_envs, add_ground_plane=True)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    scene = InteractiveScene(SimpleSceneCfg(num_envs=num_envs, env_spacing=2.0))

    stage = sim_utils.get_current_stage()
    for i in range(num_envs):
        xform_path = f"/World/envs/env_{i}/Cube/CameraMount"
        xform_prim = stage.DefinePrim(xform_path, "Xform")
        sim_utils.standardize_xform_ops(xform_prim)
        xform_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.1, 0.0, 0.05))
        xform_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

    sim.reset()
    view = XformPrimView("/World/envs/env_.*/Cube/CameraMount", device=device)
    return sim, scene, view, ctx


def _build_scene_with_world_xform(num_envs=4, device="cuda:0"):
    """Build scene with a plain Xform at the world root (no body ancestor)."""
    ctx = _newton_sim_context(device=device, num_envs=num_envs, add_ground_plane=True)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    scene = InteractiveScene(SimpleSceneCfg(num_envs=num_envs, env_spacing=2.0))

    stage = sim_utils.get_current_stage()
    xform_path = "/World/StaticMarker"
    xform_prim = stage.DefinePrim(xform_path, "Xform")
    sim_utils.standardize_xform_ops(xform_prim)
    xform_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(5.0, 3.0, 1.0))
    xform_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

    sim.reset()
    view = XformPrimView("/World/StaticMarker", device=device)
    return sim, scene, view, ctx


# ======================================================================
# Tests - Initialization
# ======================================================================


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


# ======================================================================
# Tests - Getters (returns wp.array)
# ======================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_world_poses_returns_warp_arrays():
    """Test that get_world_poses returns wp.array, not torch.Tensor."""
    sim, scene, view, ctx = _build_scene(num_envs=4)
    positions, orientations = view.get_world_poses()

    assert isinstance(positions, wp.array)
    assert isinstance(orientations, wp.array)
    assert positions.shape[0] == 4
    assert orientations.shape[0] == 4
    pos_t = wp.to_torch(positions)
    quat_t = wp.to_torch(orientations)
    assert pos_t.shape == (4, 3)
    assert quat_t.shape == (4, 4)
    assert pos_t.device.type == "cuda"
    ctx.__exit__(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_world_poses_matches_body_q():
    """Test that get_world_poses returns values matching body_q."""
    sim, scene, view, ctx = _build_scene(num_envs=4)
    positions, orientations = view.get_world_poses()
    positions_t = wp.to_torch(positions)

    model = NewtonManager.get_model()
    body_labels = list(model.body_label)
    state = NewtonManager.get_state_0()
    body_q = wp.to_torch(state.body_q)

    for i in range(4):
        prim_path = f"/World/envs/env_{i}/Cube"
        if prim_path in body_labels:
            body_idx = body_labels.index(prim_path)
            expected_pos = body_q[body_idx, :3]
            torch.testing.assert_close(positions_t[i], expected_pos, atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_world_poses_with_indices():
    """Test get_world_poses with a subset of indices."""
    sim, scene, view, ctx = _build_scene(num_envs=4)
    pos_all_t = wp.to_torch(view.get_world_poses()[0])

    indices = [0, 2]
    positions_subset, orientations_subset = view.get_world_poses(indices)

    assert isinstance(positions_subset, wp.array)
    pos_sub_t = wp.to_torch(positions_subset)
    quat_sub_t = wp.to_torch(orientations_subset)
    assert pos_sub_t.shape == (2, 3)
    assert quat_sub_t.shape == (2, 4)
    torch.testing.assert_close(pos_sub_t[0], pos_all_t[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(pos_sub_t[1], pos_all_t[2], atol=1e-6, rtol=0)
    ctx.__exit__(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_world_poses_single_index():
    """Test get_world_poses with a single index."""
    sim, scene, view, ctx = _build_scene(num_envs=5)
    pos_all_t = wp.to_torch(view.get_world_poses()[0])

    positions_single, orientations_single = view.get_world_poses([3])
    pos_s_t = wp.to_torch(positions_single)
    quat_s_t = wp.to_torch(orientations_single)
    assert pos_s_t.shape == (1, 3)
    assert quat_s_t.shape == (1, 4)
    torch.testing.assert_close(pos_s_t[0], pos_all_t[3], atol=1e-6, rtol=0)
    ctx.__exit__(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_world_poses_out_of_order_indices():
    """Test get_world_poses with out-of-order indices."""
    sim, scene, view, ctx = _build_scene(num_envs=5)
    pos_all_t = wp.to_torch(view.get_world_poses()[0])

    indices = [4, 1, 3]
    positions_subset, _ = view.get_world_poses(indices)
    pos_sub_t = wp.to_torch(positions_subset)

    assert pos_sub_t.shape == (3, 3)
    torch.testing.assert_close(pos_sub_t[0], pos_all_t[4], atol=1e-6, rtol=0)
    torch.testing.assert_close(pos_sub_t[1], pos_all_t[1], atol=1e-6, rtol=0)
    torch.testing.assert_close(pos_sub_t[2], pos_all_t[3], atol=1e-6, rtol=0)
    ctx.__exit__(None, None, None)


# ======================================================================
# Tests - Setters (wp.array input)
# ======================================================================


def _wp_vec3f(data, device="cuda:0"):
    return wp.array([wp.vec3f(*row) for row in data], dtype=wp.vec3f, device=device)


def _wp_vec4f(data, device="cuda:0"):
    return wp.array([wp.vec4f(*row) for row in data], dtype=wp.vec4f, device=device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_world_poses():
    """Test setting world poses writes to body_q."""
    sim, scene, view, ctx = _build_scene(num_envs=4)

    new_pos = _wp_vec3f([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0], [100.0, 110.0, 120.0]])
    new_quat = _wp_vec4f(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.7071068, 0.7071068],
            [0.7071068, 0.0, 0.0, 0.7071068],
            [0.3826834, 0.0, 0.0, 0.9238795],
        ]
    )

    view.set_world_poses(new_pos, new_quat)
    ret_pos, ret_quat = view.get_world_poses()

    _assert_close(ret_pos, new_pos, atol=1e-5, rtol=0)
    _assert_close(ret_quat, new_quat, atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_world_poses_only_positions():
    """Test setting only positions, leaving orientations unchanged."""
    sim, scene, view, ctx = _build_scene(num_envs=3)
    _, initial_quat = view.get_world_poses()

    new_pos = _wp_vec3f([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    view.set_world_poses(positions=new_pos, orientations=None)

    ret_pos, ret_quat = view.get_world_poses()
    _assert_close(ret_pos, new_pos, atol=1e-5, rtol=0)
    _assert_close(ret_quat, initial_quat, atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_world_poses_only_orientations():
    """Test setting only orientations, leaving positions unchanged."""
    sim, scene, view, ctx = _build_scene(num_envs=3)
    initial_pos, _ = view.get_world_poses()

    new_quat = _wp_vec4f(
        [
            [0.0, 0.0, 0.7071068, 0.7071068],
            [0.7071068, 0.0, 0.0, 0.7071068],
            [0.3826834, 0.0, 0.0, 0.9238795],
        ]
    )
    view.set_world_poses(positions=None, orientations=new_quat)

    ret_pos, ret_quat = view.get_world_poses()
    _assert_close(ret_pos, initial_pos, atol=1e-5, rtol=0)
    _assert_close(ret_quat, new_quat, atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_world_poses_with_indices():
    """Test setting world poses for a subset of indices."""
    sim, scene, view, ctx = _build_scene(num_envs=5)
    init_pos_t = wp.to_torch(view.get_world_poses()[0]).clone()

    indices = [1, 3]
    new_pos = _wp_vec3f([[10.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    view.set_world_poses(positions=new_pos, orientations=None, indices=indices)

    updated_pos_t = wp.to_torch(view.get_world_poses()[0])
    new_pos_t = wp.to_torch(new_pos)

    torch.testing.assert_close(updated_pos_t[1], new_pos_t[0], atol=1e-5, rtol=0)
    torch.testing.assert_close(updated_pos_t[3], new_pos_t[1], atol=1e-5, rtol=0)
    torch.testing.assert_close(updated_pos_t[0], init_pos_t[0], atol=1e-5, rtol=0)
    torch.testing.assert_close(updated_pos_t[2], init_pos_t[2], atol=1e-5, rtol=0)
    torch.testing.assert_close(updated_pos_t[4], init_pos_t[4], atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


# ======================================================================
# Tests - Round-trip consistency
# ======================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_write_read_roundtrip():
    """Test that write -> read round-trip is consistent."""
    sim, scene, view, ctx = _build_scene(num_envs=4)

    new_pos = _wp_vec3f([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    new_quat = _wp_vec4f([[0.0, 0.0, 0.0, 1.0]] * 4)

    view.set_world_poses(new_pos, new_quat)
    pos, quat = view.get_world_poses()

    _assert_close(pos, new_pos, atol=1e-5, rtol=0)
    _assert_close(quat, new_quat, atol=1e-5, rtol=0)

    new_pos2 = _wp_vec3f([[11.0, 12.0, 13.0], [14.0, 15.0, 16.0], [17.0, 18.0, 19.0], [20.0, 21.0, 22.0]])
    view.set_world_poses(new_pos2, new_quat)
    pos2, quat2 = view.get_world_poses()

    _assert_close(pos2, new_pos2, atol=1e-5, rtol=0)
    _assert_close(quat2, new_quat, atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


# ======================================================================
# Tests - Ancestor-walk resolution (non-physics Xform under rigid body)
# ======================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_child_xform_initialization():
    """Test that a child Xform under a rigid body resolves via ancestor walk."""
    sim, scene, view, ctx = _build_scene_with_child_xform(num_envs=4)
    assert view.count == 4
    ctx.__exit__(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_child_xform_world_pose_includes_offset():
    """Test that child Xform world pose includes the local offset from the body.

    The CameraMount is at (0.1, 0.0, 0.05) relative to the Cube body.
    Its world pose should be body_world_pos + rotated_offset.
    """
    sim, scene, view, ctx = _build_scene_with_child_xform(num_envs=4)
    child_positions, child_orientations = view.get_world_poses()
    child_pos_t = wp.to_torch(child_positions)

    body_view = XformPrimView("/World/envs/env_.*/Cube", device="cuda:0")
    body_positions, body_orientations = body_view.get_world_poses()
    body_pos_t = wp.to_torch(body_positions)

    offset = torch.tensor([0.1, 0.0, 0.05], device="cuda:0")
    for i in range(4):
        expected_pos = body_pos_t[i] + offset
        torch.testing.assert_close(child_pos_t[i], expected_pos, atol=1e-4, rtol=0)
    ctx.__exit__(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_child_xform_tracks_body_after_move():
    """Test that child Xform world pose updates when parent body moves."""
    sim, scene, view, ctx = _build_scene_with_child_xform(num_envs=2)

    body_view = XformPrimView("/World/envs/env_.*/Cube", device="cuda:0")
    new_body_pos = _wp_vec3f([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    new_body_orient = _wp_vec4f([[0.0, 0.0, 0.0, 1.0]] * 2)
    body_view.set_world_poses(new_body_pos, new_body_orient)

    child_positions, _ = view.get_world_poses()
    child_pos_t = wp.to_torch(child_positions)
    new_body_pos_t = wp.to_torch(new_body_pos)

    offset = torch.tensor([0.1, 0.0, 0.05], device="cuda:0")
    for i in range(2):
        expected_pos = new_body_pos_t[i] + offset
        torch.testing.assert_close(child_pos_t[i], expected_pos, atol=1e-4, rtol=0)
    ctx.__exit__(None, None, None)


# ======================================================================
# Tests - World-attached prims (body=-1)
# ======================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_world_xform_initialization():
    """Test that a world-rooted Xform resolves with body=-1."""
    sim, scene, view, ctx = _build_scene_with_world_xform(num_envs=2)
    assert view.count == 1
    ctx.__exit__(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_world_xform_returns_correct_pose():
    """Test that a world-rooted Xform returns its USD world transform."""
    sim, scene, view, ctx = _build_scene_with_world_xform(num_envs=2)
    positions, orientations = view.get_world_poses()

    expected_pos = torch.tensor([[5.0, 3.0, 1.0]], device="cuda:0")
    _assert_close(positions, expected_pos, atol=1e-4, rtol=0)
    ctx.__exit__(None, None, None)
