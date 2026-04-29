# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton backend tests for FrameView.

Imports the shared contract tests and provides the Newton-specific
``view_factory`` fixture.  Also includes Newton-only guard tests and
the world-attached prim edge case.
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

from pxr import Gf

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils import configclass

NEWTON_SIM_CFG = SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))
WORLD_MARKER_POS = (5.0, 3.0, 1.0)


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


@pytest.fixture
def view_factory():
    """Newton factory: CameraMount child Xform at CHILD_OFFSET under each Cube body."""

    def factory(num_envs: int, device: str) -> ViewBundle:
        ctx = _sim_context(device, num_envs=num_envs)
        sim = ctx.__enter__()
        sim._app_control_on_stop_handle = None
        InteractiveScene(_SceneCfg(num_envs=num_envs, env_spacing=2.0))

        stage = sim_utils.get_current_stage()
        for i in range(num_envs):
            prim = stage.DefinePrim(f"/World/envs/env_{i}/Cube/CameraMount", "Xform")
            sim_utils.standardize_xform_ops(prim)
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*CHILD_OFFSET))
            prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

        sim.reset()
        view = FrameView("/World/envs/env_.*/Cube/CameraMount", device=device)

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
    InteractiveScene(_SceneCfg(num_envs=2, env_spacing=2.0))
    sim.reset()

    with pytest.raises(ValueError, match="physics body"):
        FrameView("/World/envs/env_.*/Cube", device=device)
    ctx.__exit__(None, None, None)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_reject_shape_path(device):
    """FrameView rejects prim paths that resolve to a Newton collision shape."""
    ctx = _sim_context(device, num_envs=2)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    InteractiveScene(_SceneCfg(num_envs=2, env_spacing=2.0))
    sim.reset()

    shape_labels = list(NewtonManager.get_model().shape_label)
    if not shape_labels:
        pytest.skip("No shapes in model")

    with pytest.raises(ValueError, match="collision shape"):
        FrameView(shape_labels[0], device=device)
    ctx.__exit__(None, None, None)


# ==================================================================
# Newton edge case: world-attached prim (body=-1)
# ==================================================================


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_world_attached_returns_initial_pose(device):
    """A world-rooted Xform returns its USD-authored position."""
    ctx = _sim_context(device, num_envs=2)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    InteractiveScene(_SceneCfg(num_envs=2, env_spacing=2.0))

    stage = sim_utils.get_current_stage()
    prim = stage.DefinePrim("/World/StaticMarker", "Xform")
    sim_utils.standardize_xform_ops(prim)
    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*WORLD_MARKER_POS))
    prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

    sim.reset()
    view = FrameView("/World/StaticMarker", device=device)

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
    InteractiveScene(_SceneCfg(num_envs=2, env_spacing=2.0))

    stage = sim_utils.get_current_stage()
    prim = stage.DefinePrim("/World/StaticMarker", "Xform")
    sim_utils.standardize_xform_ops(prim)
    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*WORLD_MARKER_POS))
    prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

    sim.reset()
    view = FrameView("/World/StaticMarker", device=device)

    new_pos = _wp_vec3f([[10.0, 20.0, 30.0]], device=device)
    new_quat = _wp_vec4f([[0.0, 0.0, 0.0, 1.0]], device=device)
    view.set_world_poses(new_pos, new_quat)

    ret_pos, ret_quat = view.get_world_poses()
    torch.testing.assert_close(ret_pos.torch, wp.to_torch(new_pos), atol=1e-5, rtol=0)
    torch.testing.assert_close(ret_quat.torch, wp.to_torch(new_quat), atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)
