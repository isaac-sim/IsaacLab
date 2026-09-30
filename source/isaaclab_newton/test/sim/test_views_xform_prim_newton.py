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

from isaaclab.test.utils import DeviceScope, test_devices

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

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.cloner as cloner
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils import configclass

NEWTON_SIM_CFG = SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))
WORLD_MARKER_POS = (5.0, 3.0, 1.0)
SITE_PATH = "/World/Robot/SiteFrame"
VISUAL_PATH = "/World/Robot/VisualFrame"


@configclass
class _SceneCfg(InteractiveSceneCfg):
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
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
        sim_utils.create_prim("/World/envs/env_0/Cube/CameraMount", translation=CHILD_OFFSET)
        view = FrameView("/World/envs/env_[^/]+/Cube/CameraMount", device=device)
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


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_non_colliding_shapes_after_finalize(device):
    """Non-colliding site and visual shapes remain valid after finalization."""
    ctx = _sim_context(device, num_envs=1)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    body_cfg = sim_utils.CuboidCfg(
        size=(0.2, 0.2, 0.2),
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
        mass_props=sim_utils.MassCfg(mass=1.0),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
    )
    body_cfg.func("/World/Robot", body_cfg)

    site_prim = sim_utils.create_prim(SITE_PATH, prim_type="Sphere", scale=(0.01, 0.01, 0.01))
    site_schemas = Sdf.TokenListOp()
    site_schemas.prependedItems = ["MjcSiteAPI"]
    site_prim.SetMetadata("apiSchemas", site_schemas)
    sim_utils.create_prim(VISUAL_PATH, prim_type="Cube", scale=(0.01, 0.01, 0.01))
    sim.require_visual_shapes()
    assets = AssetBaseCfg(prim_path="/World/defaultGroundPlane"), AssetBaseCfg(prim_path="/World/Robot")
    plan = cloner.clone_plan_from_env_0(cloner.CloneCfg(), assets, 1, 0.0)
    cloner.replicate(plan)
    sim.reset()

    shape_labels = list(NewtonManager.get_model().shape_label)
    assert SITE_PATH in shape_labels
    assert VISUAL_PATH in shape_labels
    FrameView(SITE_PATH, device=device)
    FrameView(VISUAL_PATH, device=device)
    ctx.__exit__(None, None, None)


@pytest.mark.parametrize("device", test_devices())
def test_body_local_frame_resolves_from_body_labels_after_reset(device):
    """A body-local site created after reset resolves from the finalized Newton body labels.

    Only the prototype env authors the child prim on the stage, so the view must expand it through the
    Newton body labels rather than the stage. The ClonePlan path before reset is covered by the shared
    contract tests.
    """
    num_envs = 3
    ctx = _sim_context(device, num_envs=num_envs)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    InteractiveScene(_SceneCfg(num_envs=num_envs, env_spacing=2.0))

    stage = sim_utils.get_current_stage()
    assert stage.GetPrimAtPath("/World/envs/env_0/Cube").IsValid()
    assert not stage.GetPrimAtPath("/World/envs/env_1/Cube").IsValid()
    sim_utils.create_prim("/World/envs/env_0/Cube/CameraMount", translation=CHILD_OFFSET)

    sim.reset()
    label_view = FrameView("/World/envs/env_[^/]+/Cube/CameraMount", device=device)

    assert label_view.count == num_envs
    assert not stage.GetPrimAtPath("/World/envs/env_1/Cube/CameraMount").IsValid()
    expected = _get_body_positions(num_envs, device) + torch.tensor(CHILD_OFFSET, device=device)
    torch.testing.assert_close(label_view.get_world_poses()[0].torch, expected, atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_close_before_reset_cancels_deferred_initialization(device):
    """A view closed before the Newton model exists must not initialize on ``PHYSICS_READY``.

    After reset, new views over Newton bodies or collision shapes are rejected.
    """
    num_envs = 3
    ctx = _sim_context(device, num_envs=num_envs)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    InteractiveScene(_SceneCfg(num_envs=num_envs, env_spacing=2.0))
    sim_utils.create_prim("/World/envs/env_0/Cube/CameraMount", translation=CHILD_OFFSET)

    view = FrameView("/World/envs/env_[^/]+/Cube/CameraMount", device=device)
    assert view.count == 0, "the model already exists; this is not the deferred path"
    view.close()

    sim.reset()

    assert view.count == 0, "a closed view still initialized from the physics-ready callback"

    # FrameView rejects prim paths that resolve to a Newton physics body or collision shape.
    with pytest.raises(ValueError, match="physics body"):
        FrameView("/World/envs/env_[^/]+/Cube", device=device)
    shape_labels = list(NewtonManager.get_model().shape_label)
    assert shape_labels, "scene must contribute at least one collision shape"
    with pytest.raises(ValueError, match="collision shape"):
        FrameView(shape_labels[0], device=device)
    ctx.__exit__(None, None, None)


# ==================================================================
# Newton edge case: world-attached prim (body=-1)
# ==================================================================


@pytest.mark.parametrize("device", test_devices())
def test_world_attached_pose_read_and_write(device):
    """A world-rooted frame returns its configured position and can be repositioned via set_world_poses."""
    ctx = _sim_context(device, num_envs=2)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    InteractiveScene(_SceneCfg(num_envs=2, env_spacing=2.0))

    sim.reset()
    sim_utils.create_prim("/World/StaticMarker", translation=WORLD_MARKER_POS)
    view = FrameView("/World/StaticMarker", device=device)

    pos = view.get_world_poses()[0].torch
    expected = torch.tensor([list(WORLD_MARKER_POS)], device=device)
    torch.testing.assert_close(pos, expected, atol=1e-5, rtol=0)

    new_pos = _wp_vec3f([[10.0, 20.0, 30.0]], device=device)
    new_quat = _wp_vec4f([[0.0, 0.0, 0.0, 1.0]], device=device)
    with view.xform_world_space_writer() as w:
        w.set_poses(new_pos, new_quat)

    ret_pos, ret_quat = view.get_world_poses()
    torch.testing.assert_close(ret_pos.torch, wp.to_torch(new_pos), atol=1e-5, rtol=0)
    torch.testing.assert_close(ret_quat.torch, wp.to_torch(new_quat), atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


# ==================================================================
# Newton edge case: frame below a non-body articulation root
# ==================================================================


def _author_xform_rooted_articulation(usd_path: str) -> None:
    """Author a floating articulation whose ``ArticulationRootAPI`` sits on a plain root Xform.

    Many assets put the API there instead of on the root link. ``Mount`` is a frame below the root
    Xform but outside every rigid body; ``base/Mount`` is a frame on the root link.
    """
    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for name, pos in (("base", (0.0, 0.0, 0.0)), ("link", (0.3, 0.0, 0.0))):
        body = UsdGeom.Xform.Define(stage, f"/Robot/{name}")
        body.AddTranslateOp().Set(Gf.Vec3d(*pos))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(1.0)
        collision = UsdGeom.Cube.Define(stage, f"/Robot/{name}/collision")
        collision.CreateSizeAttr(0.1)
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/joint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/Robot/base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/Robot/link")])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.15, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(-0.15, 0.0, 0.0))
    joint.CreateAxisAttr("Y")
    for path in ("/Robot/Mount", "/Robot/base/Mount"):
        UsdGeom.Xform.Define(stage, path).AddTranslateOp().Set(Gf.Vec3d(*CHILD_OFFSET))
    stage.Save()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_frame_below_non_body_articulation_root_is_static(device, tmp_path):
    """A frame below an ``ArticulationRootAPI`` Xform but outside every rigid body stays in place.

    The root Xform is not simulated, so, as on PhysX, the frame must not follow the robot, while a
    frame on the root link must.
    """
    num_envs = 2
    robot_pos = (0.0, 0.0, 1.0)
    usd_path = str(tmp_path / "xform_rooted_articulation.usda")
    _author_xform_rooted_articulation(usd_path)

    @configclass
    class _RobotSceneCfg(InteractiveSceneCfg):
        robot: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
            init_state=ArticulationCfg.InitialStateCfg(pos=robot_pos),
            actuators={"joint": ImplicitActuatorCfg(joint_names_expr=["joint"], stiffness=0.0, damping=0.0)},
        )

    ctx = _sim_context(device, num_envs=num_envs)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    scene = InteractiveScene(_RobotSceneCfg(num_envs=num_envs, env_spacing=2.0))
    sim.reset()
    # Created after reset, as a camera sensor creates its view once the Newton model exists.
    static_view = FrameView("/World/envs/env_[^/]+/Robot/Mount", device=device)
    body_view = FrameView("/World/envs/env_[^/]+/Robot/base/Mount", device=device)
    for _ in range(20):
        sim.step()

    assert static_view.count == num_envs and body_view.count == num_envs
    offset = torch.tensor(CHILD_OFFSET, device=device)
    env_origins = scene.env_origins.to(device)
    expected_static = env_origins + torch.tensor(robot_pos, device=device) + offset
    torch.testing.assert_close(static_view.get_world_poses()[0].torch, expected_static, atol=1e-5, rtol=0)

    body_labels = list(NewtonManager.get_model().body_label)
    body_q = wp.to_torch(NewtonManager.get_state_0().body_q)
    base_pos = torch.stack([body_q[body_labels.index(f"/World/envs/env_{i}/Robot/base"), :3] for i in range(num_envs)])
    assert torch.all(base_pos[:, 2] < robot_pos[2] - 0.01), "the robot should have fallen under gravity"
    # No rotation is expected for a free fall from rest, so the offset stays axis-aligned.
    torch.testing.assert_close(body_view.get_world_poses()[0].torch, base_pos + offset, atol=1e-4, rtol=0)
    ctx.__exit__(None, None, None)
