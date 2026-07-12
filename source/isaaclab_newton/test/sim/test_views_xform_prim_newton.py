# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton backend tests for FrameView.

Imports the shared contract tests and provides the Newton-specific
``view_factory`` fixture.  Also includes Newton-only guard tests, the
world-attached prim edge case, and heterogeneous multi-asset cloning.

Newton frame views register sites that clone with the scene, so every
view is constructed inside the replication session (the same window in
which :class:`~isaaclab.scene.InteractiveScene` constructs sensors).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "isaaclab" / "test" / "sim"))

import pytest
import torch
import warp as wp
from frame_view_contract_utils import *  # noqa: F401, F403 — import all contract tests
from frame_view_contract_utils import CHILD_OFFSET, ViewBundle, _wp_vec3f, _wp_vec4f
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics.newton_manager import NewtonManager
from isaaclab_newton.sim.views import NewtonSiteFrameView as FrameView

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import SimulationCfg, build_simulation_context

NEWTON_SIM_CFG = SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))
WORLD_MARKER_POS = (5.0, 3.0, 1.0)


def _cube_spawn_cfg(size=(0.2, 0.2, 0.2)):
    return sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )


def _cube_cfg(spawn=None):
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=spawn if spawn is not None else _cube_spawn_cfg(),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


def _replicate_cube_scene(num_envs, device, session_body, cube_cfg=None):
    """Clone a cube per env and run ``session_body`` inside the replication session.

    Mirrors what :class:`~isaaclab.scene.InteractiveScene` does: author the env
    namespace, build the clone plan, construct assets inside the session, and
    replicate on exit. Returns whatever ``session_body`` returns.
    """
    stage = sim_utils.get_current_stage()
    stage.DefinePrim("/World/envs/env_0", "Xform")
    env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
    env_origins, _ = cloner.grid_transforms(num_envs, 2.0, device=device)
    with cloner.disabled_fabric_change_notifies(stage, restore=False):
        cloner.usd_replicate(stage, ["/World/envs/env_0"], ["/World/envs/env_{}"], env_ids, positions=env_origins)

    cube_cfg = cube_cfg if cube_cfg is not None else _cube_cfg()
    session = cloner.ReplicateSession([cube_cfg], num_clones=num_envs, env_spacing=2.0, device=device, stage=stage)
    with session:
        cube_cfg.class_type(cube_cfg)
        result = session_body()
    return result, session


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


@pytest.fixture
def view_factory():
    """Newton factory: CameraMount child Xform at CHILD_OFFSET under each Cube body."""

    def factory(num_envs: int, device: str) -> ViewBundle:
        ctx = _sim_context(device, num_envs=num_envs)
        sim = ctx.__enter__()
        sim._app_control_on_stop_handle = None

        def build_view():
            sim_utils.create_prim("/World/envs/env_0/Cube/CameraMount", translation=CHILD_OFFSET)
            return FrameView("/World/envs/env_.*/Cube/CameraMount", device=device)

        view, _ = _replicate_cube_scene(num_envs, device, build_view)
        sim.reset()

        return ViewBundle(
            view=view,
            get_parent_pos=_get_body_positions,
            set_parent_pos=_set_body_positions,
            teardown=lambda: ctx.__exit__(None, None, None),
        )

    return factory


# ==================================================================
# Newton-only: guard tests
# ==================================================================


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_reject_body_path(device):
    """FrameView rejects prim paths that resolve to a Newton physics body."""
    ctx = _sim_context(device, num_envs=2)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None

    def build_view():
        with pytest.raises(ValueError, match="physics body"):
            FrameView("/World/envs/env_.*/Cube", device=device)

    _replicate_cube_scene(2, device, build_view)
    ctx.__exit__(None, None, None)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_clone_plan_view_uses_source_child_without_destination_usd(device):
    """FrameView expands a registered body-local site through the ClonePlan."""
    num_envs = 3
    ctx = _sim_context(device, num_envs=num_envs)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    stage = sim_utils.get_current_stage()

    def build_view():
        assert stage.GetPrimAtPath("/World/envs/env_0/Cube").IsValid()
        assert not stage.GetPrimAtPath("/World/envs/env_1/Cube").IsValid()
        sim_utils.create_prim("/World/envs/env_0/Cube/CameraMount", translation=CHILD_OFFSET)
        return FrameView("/World/envs/env_.*/Cube/CameraMount", device=device)

    view, _ = _replicate_cube_scene(num_envs, device, build_view)
    sim.reset()

    assert view.count == num_envs
    assert not stage.GetPrimAtPath("/World/envs/env_1/Cube/CameraMount").IsValid()
    pos = view.get_world_poses()[0].torch
    expected = _get_body_positions(num_envs, device) + torch.tensor(CHILD_OFFSET, device=device)
    torch.testing.assert_close(pos, expected, atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_view_construction_after_replication_raises(device):
    """FrameView constructed after the scene replicated raises a contract error."""
    num_envs = 3
    ctx = _sim_context(device, num_envs=num_envs)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None

    def create_mount():
        sim_utils.create_prim("/World/envs/env_0/Cube/CameraMount", translation=CHILD_OFFSET)

    _replicate_cube_scene(num_envs, device, create_mount)
    with pytest.raises(RuntimeError, match="after the scene was built"):
        FrameView("/World/envs/env_.*/Cube/CameraMount", device=device)
    ctx.__exit__(None, None, None)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_view_construction_after_reset_raises(device):
    """FrameView constructed after the Newton model is built raises a contract error."""
    num_envs = 3
    ctx = _sim_context(device, num_envs=num_envs)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None

    def create_mount():
        sim_utils.create_prim("/World/envs/env_0/Cube/CameraMount", translation=CHILD_OFFSET)

    _replicate_cube_scene(num_envs, device, create_mount)
    sim.reset()
    with pytest.raises(RuntimeError, match="after the scene was built"):
        FrameView("/World/envs/env_.*/Cube/CameraMount", device=device)
    ctx.__exit__(None, None, None)


# ==================================================================
# Newton-only: heterogeneous multi-asset cloning
# ==================================================================


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_heterogeneous_sources_resolve_per_variant(device):
    """Each env resolves its own variant's frame; count and env order stay intact."""
    num_envs = 4
    variant_offsets = {0: (0.05, 0.0, 0.1), 1: (-0.05, 0.0, 0.2)}
    ctx = _sim_context(device, num_envs=num_envs)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None

    multi_spawn = sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[_cube_spawn_cfg(size=(0.2, 0.2, 0.2)), _cube_spawn_cfg(size=(0.3, 0.3, 0.3))],
        random_choice=False,
    )
    cube_cfg = _cube_cfg(spawn=multi_spawn)

    def build_view():
        source_prims = sorted(
            prim.GetPath().pathString for prim in sim_utils.find_matching_prims("/World/envs/env_.*/Cube")
        )
        assert len(source_prims) == 2, source_prims
        for variant, source in enumerate(source_prims):
            sim_utils.create_prim(f"{source}/CameraMount", translation=variant_offsets[variant])
        return FrameView("/World/envs/env_.*/Cube/CameraMount", device=device)

    view, session = _replicate_cube_scene(num_envs, device, build_view, cube_cfg=cube_cfg)
    cfg_rows = sorted(session.plan.cfg_rows[id(cube_cfg)])
    variant_of_env = session.plan.clone_mask[cfg_rows].to(torch.long).argmax(dim=0)
    sim.reset()

    assert view.count == num_envs
    offsets = torch.stack([torch.tensor(variant_offsets[int(v)]) for v in variant_of_env]).to(device)
    expected = _get_body_positions(num_envs, device) + offsets
    pos = view.get_world_poses()[0].torch
    torch.testing.assert_close(pos, expected, atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


# ==================================================================
# Newton edge case: world-attached prim (body=-1)
# ==================================================================


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_world_attached_returns_initial_pose(device):
    """A world-rooted frame returns its configured position."""
    ctx = _sim_context(device, num_envs=2)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None

    def build_view():
        sim_utils.create_prim("/World/StaticMarker", translation=WORLD_MARKER_POS)
        return FrameView("/World/StaticMarker", device=device)

    view, _ = _replicate_cube_scene(2, device, build_view)
    sim.reset()

    pos = view.get_world_poses()[0].torch
    expected = torch.tensor([list(WORLD_MARKER_POS)], device=device)
    torch.testing.assert_close(pos, expected, atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_world_attached_set_world_roundtrip(device):
    """A world-attached prim can be repositioned via set_world_poses."""
    ctx = _sim_context(device, num_envs=2)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None

    def build_view():
        sim_utils.create_prim("/World/StaticMarker", translation=WORLD_MARKER_POS)
        return FrameView("/World/StaticMarker", device=device)

    view, _ = _replicate_cube_scene(2, device, build_view)
    sim.reset()

    new_pos = _wp_vec3f([[10.0, 20.0, 30.0]], device=device)
    new_quat = _wp_vec4f([[0.0, 0.0, 0.0, 1.0]], device=device)
    with view.xform_world_space_writer() as w:
        w.set_poses(new_pos, new_quat)

    ret_pos, ret_quat = view.get_world_poses()
    torch.testing.assert_close(ret_pos.torch, wp.to_torch(new_pos), atol=1e-5, rtol=0)
    torch.testing.assert_close(ret_quat.torch, wp.to_torch(new_quat), atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)
