# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton backend tests for FrameView.

Runs the shared contract checks in resettable Newton scenes.  Also includes Newton-only guard tests and
the world-attached prim edge case.
"""

import sys
from pathlib import Path

from isaaclab.test.utils import test_devices

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "isaaclab" / "test" / "sim"))

import frame_view_contract_utils as frame_view_contract
import pytest
import torch
import warp as wp
from frame_view_contract_utils import CHILD_OFFSET, ViewBundle, _wp_vec3f, _wp_vec4f
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics.newton_manager import NewtonManager
from isaaclab_newton.sim.views import NewtonSiteFrameView as FrameView

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.configclass import configclass

NEWTON_SIM_CFG = SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))
WORLD_MARKER_POS = (5.0, 3.0, 1.0)


pytestmark = pytest.mark.ci_only


@configclass
class _SceneCfg(InteractiveSceneCfg):
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


def _sim_context(device, num_envs=4):
    NEWTON_SIM_CFG.device = device
    return build_simulation_context(device=device, sim_cfg=NEWTON_SIM_CFG, add_ground_plane=True)


def _get_body_positions(num_envs, device="cpu"):
    model = NewtonManager.get_model()
    body_labels = list(model.body_label)
    body_q_t = wp.to_torch(NewtonManager.get_state_0().body_q)
    return torch.stack([body_q_t[body_labels.index(f"/World/envs/env_{i}/Cube"), :3] for i in range(num_envs)])


def _set_body_positions(positions, num_envs):
    model = NewtonManager.get_model()
    body_labels = list(model.body_label)
    body_q_t = wp.to_torch(NewtonManager.get_state_0().body_q)
    for i in range(num_envs):
        body_q_t[body_labels.index(f"/World/envs/env_{i}/Cube"), :3] = positions[i]


# ------------------------------------------------------------------
# Contract fixture
# ------------------------------------------------------------------


def _view_factory():
    """Newton factory: CameraMount child Xform at CHILD_OFFSET under each Cube body."""

    def factory(num_envs: int, device: str) -> ViewBundle:
        ctx = _sim_context(device, num_envs=num_envs)
        sim = ctx.__enter__()
        sim._app_control_on_stop_handle = None
        InteractiveScene(_SceneCfg(num_envs=num_envs, env_spacing=2.0))
        sim_utils.create_prim("/World/envs/env_0/Cube/CameraMount", translation=CHILD_OFFSET)
        view = FrameView("/World/envs/env_.*/Cube/CameraMount", device=device)
        sim.reset()

        return ViewBundle(
            view=view,
            get_parent_pos=_get_body_positions,
            set_parent_pos=_set_body_positions,
            teardown=lambda: ctx.__exit__(None, None, None),
        )

    return factory


_CONTRACT_CASES = [
    (
        2,
        [
            frame_view_contract.test_local_differs_from_world,
            frame_view_contract.test_local_stable_after_parent_move,
            frame_view_contract.test_world_tracks_parent_move,
            frame_view_contract.test_set_world_roundtrip,
            frame_view_contract.test_set_local_roundtrip,
            frame_view_contract.test_set_world_does_not_move_parent,
            frame_view_contract.test_set_local_does_not_move_parent,
            frame_view_contract.test_set_world_updates_local,
            frame_view_contract.test_set_local_updates_world,
            frame_view_contract.test_set_world_partial_position_only,
            frame_view_contract.test_set_world_partial_orientation_only,
            frame_view_contract.test_set_local_partial_position_only,
            frame_view_contract.test_return_types_are_torcharray,
            frame_view_contract.test_local_scales_default_identity,
            frame_view_contract.test_world_scales_default_identity,
            frame_view_contract.test_local_scales_roundtrip,
            frame_view_contract.test_world_scales_roundtrip,
            frame_view_contract.test_local_scales_do_not_affect_local_poses,
            frame_view_contract.test_scale_getters_return_proxyarray,
        ],
    ),
    (
        4,
        [
            frame_view_contract.test_world_pose_equals_parent_plus_offset,
            frame_view_contract.test_local_pose_equals_structural_offset,
            frame_view_contract.test_set_world_indexed_only_affects_subset,
        ],
    ),
    (5, [frame_view_contract.test_indexed_get_returns_correct_subset]),
]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("num_envs, contract_tests", _CONTRACT_CASES, ids=["two-env", "four-env", "five-env"])
def test_frame_view_contract_group(device, num_envs, contract_tests):
    """Run compatible FrameView contracts against one resettable initialized scene."""
    bundle = _view_factory()(num_envs=num_envs, device=device)
    initial_parent_pos = bundle.get_parent_pos(num_envs, device).clone()
    reset_bundle = bundle._replace(teardown=lambda: None)
    expected_num_envs = num_envs
    expected_device = device

    def reusable_factory(num_envs, device):
        assert num_envs == expected_num_envs
        assert device == expected_device
        bundle.set_parent_pos(initial_parent_pos, num_envs)
        local_positions = _wp_vec3f([list(CHILD_OFFSET)] * num_envs, device=device)
        local_orientations = _wp_vec4f([[0.0, 0.0, 0.0, 1.0]] * num_envs, device=device)
        local_scales = _wp_vec3f([[1.0, 1.0, 1.0]] * num_envs, device=device)
        with bundle.view.xform_local_space_writer() as writer:
            writer.set_poses(local_positions, local_orientations)
            writer.set_scales(local_scales)
        return reset_bundle

    try:
        for contract_test in contract_tests:
            contract_test(device, reusable_factory)
    finally:
        bundle.teardown()


# ==================================================================
# Newton-only: guard tests
# ==================================================================


@pytest.mark.parametrize("device", test_devices())
def test_reject_physics_paths(device):
    """FrameView rejects Newton physics bodies and collision shapes."""
    ctx = _sim_context(device, num_envs=2)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    InteractiveScene(_SceneCfg(num_envs=2, env_spacing=2.0))
    sim.reset()

    try:
        with pytest.raises(ValueError, match="physics body"):
            FrameView("/World/envs/env_.*/Cube", device=device)

        shape_labels = list(NewtonManager.get_model().shape_label)
        if not shape_labels:
            pytest.skip("No shapes in model")
        with pytest.raises(ValueError, match="collision shape"):
            FrameView(shape_labels[0], device=device)
    finally:
        ctx.__exit__(None, None, None)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_clone_plan_view_resolves_before_and_after_reset(device):
    """Resolve the same cloned body-local frame both before and after simulation reset."""
    num_envs = 3
    ctx = _sim_context(device, num_envs=num_envs)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    InteractiveScene(_SceneCfg(num_envs=num_envs, env_spacing=2.0))

    stage = sim_utils.get_current_stage()
    assert stage.GetPrimAtPath("/World/envs/env_0/Cube").IsValid()
    assert not stage.GetPrimAtPath("/World/envs/env_1/Cube").IsValid()
    sim_utils.create_prim("/World/envs/env_0/Cube/CameraMount", translation=CHILD_OFFSET)

    view_before_reset = FrameView("/World/envs/env_.*/Cube/CameraMount", device=device)
    sim.reset()

    try:
        expected = _get_body_positions(num_envs, device) + torch.tensor(CHILD_OFFSET, device=device)
        assert view_before_reset.count == num_envs
        assert not stage.GetPrimAtPath("/World/envs/env_1/Cube/CameraMount").IsValid()
        torch.testing.assert_close(view_before_reset.get_world_poses()[0].torch, expected, atol=1e-5, rtol=0)

        view_after_reset = FrameView("/World/envs/env_.*/Cube/CameraMount", device=device)
        torch.testing.assert_close(view_after_reset.get_world_poses()[0].torch, expected, atol=1e-5, rtol=0)
    finally:
        ctx.__exit__(None, None, None)


# ==================================================================
# Newton edge case: world-attached prim (body=-1)
# ==================================================================


@pytest.mark.parametrize("device", test_devices())
def test_world_attached_pose_roundtrip(device):
    """Read and reposition a world-attached frame within one initialized scene."""
    ctx = _sim_context(device, num_envs=2)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    InteractiveScene(_SceneCfg(num_envs=2, env_spacing=2.0))

    sim.reset()
    sim_utils.create_prim("/World/StaticMarker", translation=WORLD_MARKER_POS)
    view = FrameView("/World/StaticMarker", device=device)

    try:
        pos = view.get_world_poses()[0].torch
        expected = torch.tensor([list(WORLD_MARKER_POS)], device=device)
        torch.testing.assert_close(pos, expected, atol=1e-5, rtol=0)

        new_pos = _wp_vec3f([[10.0, 20.0, 30.0]], device=device)
        new_quat = _wp_vec4f([[0.0, 0.0, 0.0, 1.0]], device=device)
        with view.xform_world_space_writer() as writer:
            writer.set_poses(new_pos, new_quat)

        ret_pos, ret_quat = view.get_world_poses()
        torch.testing.assert_close(ret_pos.torch, wp.to_torch(new_pos), atol=1e-5, rtol=0)
        torch.testing.assert_close(ret_quat.torch, wp.to_torch(new_quat), atol=1e-5, rtol=0)
    finally:
        ctx.__exit__(None, None, None)
