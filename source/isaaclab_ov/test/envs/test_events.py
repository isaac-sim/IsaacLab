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

from pxr import UsdGeom, UsdPhysics

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov import tensor_types as TT
from isaaclab_ov.assets import Articulation, RigidObject
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs.mdp import randomize_rigid_body_collider_offsets, randomize_rigid_body_material
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sim import SimulationCfg, build_simulation_context


class _Scene(dict):
    num_envs = 2


@pytest.fixture
def event_env():
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device="cpu", gravity=(0.0, 0.0, 0.0))
    with build_simulation_context(device="cpu", sim_cfg=sim_cfg) as sim:
        body_names = ["base", "left_upper", "left_tip", "right_upper", "right_tip"]
        for index in range(2):
            sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 10.0, 0.0, 0.0))
        robot = Articulation(
            ArticulationCfg(
                prim_path="/World/Env_[^/]*/Robot",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(Path(__file__).parents[1] / "assets/data/articulation_ordering_branching.usda")
                ),
                actuators={},
                body_ordering=list(reversed(body_names)),
            )
        )
        for index in range(2):
            for body_index, name in enumerate(body_names):
                cube = UsdGeom.Cube.Define(sim.stage, f"/World/Env_{index}/Robot/{name}/Collider")
                cube.CreateSizeAttr(0.1)
                UsdGeom.Xformable(cube).AddTranslateOp().Set((0.0, body_index * 0.5, 0.0))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        cube = RigidObject(
            RigidObjectCfg(
                prim_path="/World/Env_[^/]*/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(0.1, 0.1, 0.1),
                    rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                    collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                    mass_props=sim_utils.MassCfg(mass=1.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
            )
        )
        sim.reset()
        yield SimpleNamespace(sim=sim, scene=_Scene(robot=robot, cube=cube), num_envs=2, device="cpu")


def test_material_body_and_environment_selection(event_env):
    env = event_env
    robot = env.scene["robot"]
    body_names = robot.body_names
    paths = [f"/World/Env_{index}/Robot/{name}" for index in range(2) for name in body_names]
    view = OvPhysxView(robot._ovphysx, prim_paths=paths, device="cpu")
    binding = view.binding_for(TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION)
    selected_path = f"/World/Env_1/Robot/{body_names[0]}"
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
