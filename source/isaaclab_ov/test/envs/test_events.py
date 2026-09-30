# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared MDP events against OVPhysX assets."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from pxr import Usd, UsdGeom, UsdPhysics

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov import tensor_types as TT
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, VisualMaterialCfg
from isaaclab.envs.mdp import (
    randomize_rigid_body_collider_offsets,
    randomize_rigid_body_material,
    randomize_visual_material,
)
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices


@pytest.fixture(scope="module", params=test_devices(DeviceScope.CUDA))
def event_env(request, tmp_path_factory):
    device = request.param
    source = Usd.Stage.Open(str(Path(__file__).parents[1] / "assets/data/articulation_ordering_branching.usda"))
    asset_path = tmp_path_factory.mktemp("event_assets") / "robot.usda"
    source.GetRootLayer().Export(str(asset_path))
    stage = Usd.Stage.Open(str(asset_path))
    bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    for index, body in enumerate(bodies):
        collider = UsdGeom.Cube.Define(stage, body.GetPath().AppendChild("Collider"))
        collider.CreateSizeAttr(0.1)
        UsdGeom.Xformable(collider).AddTranslateOp().Set((0.0, index * 0.5, 0.0))
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    stage.GetRootLayer().Save()
    scene_cfg = InteractiveSceneCfg(num_envs=2, env_spacing=10.0)
    scene_cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(asset_path)),
        actuators={},
        body_ordering=[body.GetName() for body in reversed(bodies)],
    )
    scene_cfg.cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    for name in ("body", "legs"):
        setattr(
            scene_cfg,
            name,
            VisualMaterialCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{name}",
                spawn=sim_utils.PreviewSurfaceCfg(),
            ),
        )
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device, gravity=(0.0, 0.0, 0.0))
    with build_simulation_context(device=device, sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        yield SimpleNamespace(sim=sim, scene=scene, num_envs=scene.num_envs, device=sim.device)


def test_material_body_and_environment_selection(event_env):
    env = event_env
    robot = env.scene["robot"]
    body_names = robot.body_names
    paths = [f"{path}/Robot/{name}" for path in env.scene.env_prim_paths for name in body_names]
    view = OvPhysxView(robot._ovphysx, prim_paths=paths, device=env.device)
    binding = view.binding_for(TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION)
    selected_path = f"{env.scene.env_prim_paths[-1]}/Robot/{body_names[0]}"
    selected_row = binding.prim_paths.index(selected_path)
    before = wp.to_torch(view.get_attribute(TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION)).clone()
    assert before.shape[1] == 1
    params = {
        "static_friction_range": (0.4, 0.4),
        "dynamic_friction_range": (0.8, 0.8),
        "restitution_range": (0.3, 0.3),
        "num_buckets": 1,
        "make_consistent": True,
        "asset_cfg": SceneEntityCfg("robot", body_ids=[0]),
    }
    term = randomize_rigid_body_material(EventTermCfg(func=randomize_rigid_body_material, params=params), env)
    term(env, torch.tensor([1]), **params)
    expected = before.clone()
    expected[selected_row] = torch.tensor([0.4, 0.4, 0.3])
    torch.testing.assert_close(wp.to_torch(view.get_attribute(TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION)), expected)

    params["asset_cfg"] = SceneEntityCfg("robot", body_ids=[])
    empty_term = randomize_rigid_body_material(EventTermCfg(func=randomize_rigid_body_material, params=params), env)
    empty_term(env, None, **params)
    torch.testing.assert_close(wp.to_torch(view.get_attribute(TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION)), expected)

    params["asset_cfg"] = SceneEntityCfg("robot", body_ids=list(reversed(range(robot.num_bodies))))
    full_term = randomize_rigid_body_material(EventTermCfg(func=randomize_rigid_body_material, params=params), env)
    full_term(env, slice(None), **params)
    torch.testing.assert_close(
        wp.to_torch(view.get_attribute(TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION)),
        torch.tensor([0.4, 0.4, 0.3]).expand_as(before),
    )


def test_visual_material_environment_selection(event_env):
    env = event_env
    materials = [env.scene[name] for name in ("body", "legs")]
    env.sim.render_context.finalize_consumers([])
    visual_params = {
        "materials": [SceneEntityCfg(name) for name in ("body", "legs")],
        "channels": {"color": ((0.25, 0.5, 0.75), (0.25, 0.5, 0.75))},
    }
    visual_term = randomize_visual_material(EventTermCfg(func=randomize_visual_material, params=visual_params), env)
    for selection in (slice(1, None, 2), slice(0, 0), slice(None)):
        expected_colors = [material.data["color"].clone() for material in materials]
        visual_term(env, selection, **visual_params)
        for material, expected_color in zip(materials, expected_colors, strict=True):
            expected_color[selection] = torch.tensor(visual_params["channels"]["color"][0], device=env.device)
            torch.testing.assert_close(material.data["color"], expected_color)


def test_collider_offsets_preserve_unselected_state(event_env):
    env = event_env
    for name, rest_type, contact_type in (
        ("robot", TT.REST_OFFSET, TT.CONTACT_OFFSET),
        ("cube", TT.RIGID_BODY_REST_OFFSET, TT.RIGID_BODY_CONTACT_OFFSET),
    ):
        view = env.scene[name].root_view
        before_rest = wp.to_torch(view.get_attribute(rest_type)).clone()
        before_contact = wp.to_torch(view.get_attribute(contact_type)).clone()
        asset_cfg = SceneEntityCfg(name)
        term = randomize_rigid_body_collider_offsets(
            EventTermCfg(func=randomize_rigid_body_collider_offsets, params={"asset_cfg": asset_cfg}), env
        )
        term(env, torch.tensor([1]), asset_cfg, (-0.01, -0.01), (0.04, 0.04))
        expected_rest = before_rest.clone()
        expected_contact = before_contact.clone()
        expected_rest[1] = -0.01
        expected_contact[1] = 0.04
        torch.testing.assert_close(wp.to_torch(view.get_attribute(rest_type)), expected_rest)
        torch.testing.assert_close(wp.to_torch(view.get_attribute(contact_type)), expected_contact)

        term(env, slice(1, 2), asset_cfg, contact_offset_distribution_params=(0.06, 0.06))
        expected_contact[1] = 0.06
        term(env, None, asset_cfg)
        torch.testing.assert_close(wp.to_torch(view.get_attribute(rest_type)), expected_rest)
        torch.testing.assert_close(wp.to_torch(view.get_attribute(contact_type)), expected_contact)
