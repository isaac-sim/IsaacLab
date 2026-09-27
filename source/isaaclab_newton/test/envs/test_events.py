# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared MDP events against Newton assets and model buffers."""

from collections.abc import Iterator
from copy import copy
from pathlib import Path

import pytest
import torch
import warp as wp
from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonManager

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.envs.mdp import (
    JointEffortActionCfg,
    joint_pos,
    randomize_joint_parameters,
    randomize_physics_scene_gravity,
    randomize_rigid_body_collider_offsets,
    randomize_rigid_body_material,
    randomize_visual_shape,
)
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.renderers import RenderContext, RendererCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg


@configclass
class ActionsCfg:
    effort = JointEffortActionCfg(asset_name="robot", joint_names=[".*"])


@pytest.fixture(scope="module", params=test_devices(DeviceScope.CUDA))
def device(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def event_env(
    request: pytest.FixtureRequest, device: str, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[ManagerBasedEnv]:
    source = Path(__file__).parents[1] / "assets/data/articulation_ordering_branching.usda"
    stage = Usd.Stage.Open(Usd.Stage.Open(str(source)).Flatten())
    body_names = ["base", "left_upper", "left_tip", "right_upper", "right_tip"]
    for body_index, name in enumerate(body_names):
        for shape_index in range(body_index + 1):
            cube = UsdGeom.Cube.Define(stage, f"/Robot/{name}/Collider_{shape_index}")
            cube.CreateSizeAttr(0.05)
            UsdGeom.Xformable(cube).AddTranslateOp().Set((0.0, shape_index * 0.1, 0.0))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    path = tmp_path_factory.mktemp("newton_events") / "robot.usda"
    stage.Export(str(path))
    scene_cfg = InteractiveSceneCfg(num_envs=4, env_spacing=5.0)
    scene_cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(path)),
        actuators={},
        body_ordering=list(reversed(body_names)),
    )
    observations = ObservationGroupCfg()
    observations.joints = ObservationTermCfg(func=joint_pos)
    cfg = ManagerBasedEnvCfg(
        sim=SimulationCfg(
            device=device,
            physics=NewtonCfg(solver_cfg=getattr(request, "param", MJWarpSolverCfg()), use_cuda_graph=False),
        ),
        scene=scene_cfg,
        decimation=1,
        observations={"policy": observations},
        actions=ActionsCfg(),
        events={},
        seed=7,
    )
    env = ManagerBasedEnv(cfg)
    try:
        yield env
    finally:
        env.close()


def test_material_body_selection(event_env: ManagerBasedEnv):
    env = event_env
    model = env.sim.physics_manager.get_model()
    shape_bodies = wp.to_torch(model.shape_body)
    selected = torch.tensor(
        [
            index
            for index, body in enumerate(shape_bodies.tolist())
            if model.body_label[body].endswith("/env_1/Robot/right_tip")
        ],
        device=env.device,
    )
    assert selected.numel() == 5
    params = {
        "asset_cfg": SceneEntityCfg("robot", body_ids=[0]),
        "static_friction_range": (0.4, 0.4),
        "dynamic_friction_range": (0.2, 0.2),
        "restitution_range": (0.1, 0.1),
        "num_buckets": 1,
    }
    friction = wp.to_torch(model.shape_material_mu)
    restitution = wp.to_torch(model.shape_material_restitution)
    expected_friction = friction.clone()
    expected_restitution = restitution.clone()
    expected_friction[selected] = 0.4
    expected_restitution[selected] = 0.1
    term = randomize_rigid_body_material(EventTermCfg(func=randomize_rigid_body_material, params=params), env)
    term(env, torch.tensor([1], device=env.device, dtype=torch.int32), **params)
    torch.testing.assert_close(friction, expected_friction)
    torch.testing.assert_close(restitution, expected_restitution)


def test_collider_offsets_preserve_unselected_environments(event_env: ManagerBasedEnv):
    env = event_env
    model = env.sim.physics_manager.get_model()
    worlds = wp.to_torch(model.shape_world)
    margin = wp.to_torch(model.shape_margin)
    gap = wp.to_torch(model.shape_gap)
    expected_margin = margin.clone()
    expected_gap = gap.clone()
    asset_cfg = SceneEntityCfg("robot")
    term = randomize_rigid_body_collider_offsets(
        EventTermCfg(func=randomize_rigid_body_collider_offsets, params={"asset_cfg": asset_cfg}), env
    )
    term(env, torch.tensor([1], device=env.device), asset_cfg, (0.3, 0.3), (0.4, 0.4))
    expected_margin[worlds == 1] = 0.3
    expected_gap[worlds == 1] = 0.1
    torch.testing.assert_close(margin, expected_margin)
    torch.testing.assert_close(gap, expected_gap)
    term(env, slice(0, 1), asset_cfg, contact_offset_distribution_params=(0.6, 0.6))
    expected_gap[worlds == 0] = 0.6 - expected_margin[worlds == 0]
    torch.testing.assert_close(margin, expected_margin)
    torch.testing.assert_close(gap, expected_gap)
    term(env, torch.tensor([1], device=env.device), asset_cfg, contact_offset_distribution_params=(0.2, 0.2))
    expected_gap[worlds == 1] = 0.0
    torch.testing.assert_close(margin, expected_margin)
    torch.testing.assert_close(gap, expected_gap)


def test_gravity_selectors_preserve_global_world(event_env: ManagerBasedEnv):
    env = event_env
    gravity = wp.to_torch(env.sim.physics_manager.get_model().gravity)
    original = gravity.clone()
    params = {"gravity_distribution_params": ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)), "operation": "abs"}
    term = randomize_physics_scene_gravity(EventTermCfg(func=randomize_physics_scene_gravity, params=params), env)
    for selector, rows in (
        (None, [0, 1, 2, 3]),
        (slice(None), [0, 1, 2, 3]),
        (slice(1, None, 2), [1, 3]),
        (slice(-2, None), [2, 3]),
        (torch.tensor([1, 3], dtype=torch.int32, device=env.device), [1, 3]),
        (slice(0, 0), []),
    ):
        gravity.copy_(original)
        term(env, selector, **params)
        expected = original.clone()
        expected[rows] = torch.tensor([1.0, 2.0, 3.0], device=env.device)
        torch.testing.assert_close(gravity, expected)
    gravity.copy_(original)
    bounds = ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    selection = torch.tensor([1], device=env.device)
    for _ in range(2):
        term(env, selection, bounds, operation="add")
    expected = original.clone()
    expected[1] += torch.tensor([2.0, 4.0, 6.0], device=env.device)
    torch.testing.assert_close(gravity, expected)
    bounds[0][:] = [2.0, 2.0, 2.0]
    bounds[1][:] = [2.0, 2.0, 2.0]
    for _ in range(2):
        term(env, selection, bounds, operation="scale")
    expected[1] *= 4
    torch.testing.assert_close(gravity, expected)
    gravity.copy_(original)


def test_joint_friction_preserves_unselected_environments(event_env: ManagerBasedEnv):
    env = event_env
    robot = env.scene["robot"]
    model = env.sim.physics_manager.get_model()
    friction = wp.to_torch(model.joint_friction)
    damping = wp.to_torch(model.joint_damping)
    expected_friction = friction.clone()
    expected_damping = damping.clone()
    dof_starts = model.joint_qd_start.numpy()
    params = {"asset_cfg": SceneEntityCfg("robot"), "operation": "abs"}
    term = randomize_joint_parameters(EventTermCfg(func=randomize_joint_parameters, params=params), env)
    for selector, row, value in ((torch.tensor([1], device=env.device), 1, 0.5), (slice(2, 3), 2, 0.7)):
        term(env, selector, friction_distribution_params=(value, value), **params)
        selected = [
            dof
            for joint, label in enumerate(model.joint_label)
            if f"/env_{row}/" in label and label.rsplit("/", 1)[-1] in robot.joint_names
            for dof in range(dof_starts[joint], dof_starts[joint + 1])
        ]
        assert len(selected) == robot.num_joints
        expected_friction[selected] = value
        expected_damping[selected] = value
        torch.testing.assert_close(friction, expected_friction)
        torch.testing.assert_close(damping, expected_damping)


@pytest.mark.parametrize("consumer", ["visualizer", "renderer"])
def test_visual_colors_select_bodies_and_rebind(
    event_env: ManagerBasedEnv, monkeypatch: pytest.MonkeyPatch, consumer: str
):
    env = event_env
    model = env.sim.physics_manager.get_model()
    if consumer == "visualizer":
        monkeypatch.setattr(env.sim.cfg, "visualizer_cfgs", [VisualizerCfg(visualizer_type="newton_gl")])
    else:
        monkeypatch.setattr(
            env.sim, "_render_context", RenderContext([(RendererCfg(renderer_type="newton_warp"), None)])
        )
    params = {
        "asset_cfg": SceneEntityCfg("robot", body_ids=[0, 2]),
        "channels": {"color": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))},
    }
    term = randomize_visual_shape(EventTermCfg(func=randomize_visual_shape, params=params), env)
    replacement = copy(model)
    replacement.shape_color = wp.clone(model.shape_color)
    monkeypatch.setattr(NewtonManager.backend, "model", replacement)
    monkeypatch.setattr(env.scene["robot"].root_view, "model", model)
    colors = wp.to_torch(replacement.shape_color)
    before = colors.clone()
    shape_bodies = wp.to_torch(model.shape_body).tolist()
    body_paths = [model.body_label[body] for body in shape_bodies]
    for selector, rows in (
        (torch.tensor([3, 1], device=env.device), [3, 1]),
        (slice(None), [0, 1, 2, 3]),
        (slice(1, None, 2), [1, 3]),
        (slice(0, 0), []),
    ):
        colors.copy_(before)
        term(env, selector, **params)
        changed = torch.zeros(len(colors), dtype=torch.bool, device=env.device)
        for row in rows:
            sampled = []
            for body_name in ("right_tip", "left_tip"):
                indices = [
                    index for index, path in enumerate(body_paths) if path.endswith(f"/env_{row}/Robot/{body_name}")
                ]
                changed[indices] = True
                actual = colors[indices]
                assert torch.all((actual >= 0) & (actual <= 1))
                torch.testing.assert_close(actual, actual[0].expand_as(actual))
                sampled.append(actual[0])
            assert not torch.equal(*sampled)
        torch.testing.assert_close(colors[~changed], before[~changed])
    torch.testing.assert_close(wp.to_torch(model.shape_color), before)
    params["channels"] = {"roughness": (0.0, 1.0)}
    with pytest.raises(NotImplementedError, match="only the 'color' channel"):
        randomize_visual_shape(EventTermCfg(func=randomize_visual_shape, params=params), env)


@pytest.mark.parametrize("event_env", [KaminoPADMMSolverCfg()], indirect=True)
def test_kamino_material_groups_persist(event_env: ManagerBasedEnv):
    env = event_env
    model = env.sim.physics_manager.get_model()
    friction = wp.to_torch(model.shape_material_mu)
    restitution = wp.to_torch(model.shape_material_restitution)
    shape_bodies = wp.to_torch(model.shape_body).tolist()
    selected = torch.tensor(
        [
            [
                index
                for index, body in enumerate(shape_bodies)
                if model.body_label[body].endswith(f"/env_{row}/Robot/right_tip")
            ]
            for row in range(env.num_envs)
        ],
        device=env.device,
    )
    friction[selected] = 0.4
    restitution[selected] = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.2], device=env.device)
    untouched = torch.ones(len(friction), dtype=torch.bool, device=env.device)
    untouched[selected] = False
    original_friction = friction.clone()
    original_restitution = restitution.clone()
    params = {
        "asset_cfg": SceneEntityCfg("robot", body_ids=[0]),
        "static_friction_range": (0.5, 1.0),
        "dynamic_friction_range": (0.0, 0.0),
        "restitution_range": (0.3, 0.9),
        "num_buckets": 1,
    }
    term = randomize_rigid_body_material(EventTermCfg(func=randomize_rigid_body_material, params=params), env)
    for _ in range(2):
        term(env, torch.tensor([1], device=env.device), **params)
        for values, original in ((friction, original_friction), (restitution, original_restitution)):
            torch.testing.assert_close(values[untouched], original[untouched])
            groups = values[selected]
            torch.testing.assert_close(groups, groups[0].expand_as(groups))
            torch.testing.assert_close(groups[:, 0], groups[:, 1])
            assert not torch.equal(groups[:, 0], groups[:, 2])
        # Equal sampled values must not merge the original material groups.
        friction[selected] = 0.7
        restitution[selected] = 0.6
