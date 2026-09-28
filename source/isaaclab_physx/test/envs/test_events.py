# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared MDP events against native PhysX views."""

from isaaclab.app import AppLauncher
from isaaclab.test.utils import DeviceScope, resolve_test_sim_device, test_devices

simulation_app = AppLauncher(headless=True, device=resolve_test_sim_device()).app

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import warp as wp

from pxr import PhysxSchema, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, replace

pytestmark = pytest.mark.isaacsim_ci


@configclass
class ActionsCfg:
    """Minimal action configuration for a managed environment."""

    effort = mdp.JointEffortActionCfg(asset_name="identity", joint_names=[".*"], scale=1.0)


def configure_articulations(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
    """Add a fixed tendon and distinct collision-shape counts to the local USD fixture."""
    body_names = ("base", "left_upper", "left_tip", "right_upper", "right_tip")
    for env_id in range(env.num_envs):
        for asset_name in ("identity", "reordered"):
            joint = env.sim.stage.GetPrimAtPath(f"/World/envs/env_{env_id}/{asset_name}/left_shoulder")
            PhysxSchema.PhysxTendonAxisRootAPI.Apply(joint, "tendon")
            PhysxSchema.PhysxTendonAxisAPI.Apply(joint, "tendon").CreateGearingAttr([1.0])
            for count, body_name in enumerate(body_names, start=1):
                for shape_id in range(count):
                    path = f"/World/envs/env_{env_id}/{asset_name}/{body_name}/shape_{shape_id}"
                    shape = UsdGeom.Cube.Define(env.sim.stage, path)
                    shape.CreateSizeAttr(0.01)
                    shape.AddTranslateOp().Set((0.05 * shape_id, 0.05 * count, 0.0))
                    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())


@pytest.fixture(scope="module", params=test_devices(DeviceScope.CUDA))
def env(request: pytest.FixtureRequest) -> Iterator[ManagerBasedEnv]:
    scene = InteractiveSceneCfg(num_envs=2, env_spacing=5.0, replicate_physics=False)
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/identity",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).parents[1] / "assets/data/articulation_ordering_branching.usda")
        ),
        actuators={},
        init_state=ArticulationCfg.InitialStateCfg(pos=(-1.0, 0.0, 0.0)),
    )
    scene.identity = robot
    scene.reordered = replace(
        robot,
        prim_path="{ENV_REGEX_NS}/reordered",
        init_state=ArticulationCfg.InitialStateCfg(pos=(1.0, 0.0, 0.0)),
        body_ordering=("base", "right_tip", "right_upper", "left_tip", "left_upper"),
    )
    scene.cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 3.0)),
    )
    observations = ObservationGroupCfg()
    observations.joint_pos = ObservationTermCfg(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("identity")})
    cfg = ManagerBasedEnvCfg(
        scene=scene,
        sim=sim_utils.SimulationCfg(device=request.param),
        decimation=1,
        actions=ActionsCfg(),
        seed=0,
        observations={"policy": observations},
        events={"articulations": EventTermCfg(func=configure_articulations, mode="prestartup")},
    )
    environment = ManagerBasedEnv(cfg)
    try:
        yield environment
    finally:
        environment.close()


def test_material_selection(env: ManagerBasedEnv) -> None:
    """Public body and environment selections reach the matching collision shapes."""
    for asset_name, selection in (
        ("reordered", torch.tensor([0], dtype=torch.int32, device=env.device)),
        ("identity", torch.tensor([0], dtype=torch.int64, device=env.device)),
        ("reordered", slice(1, 2)),
        ("cube", torch.tensor([1], dtype=torch.int64, device=env.device)),
    ):
        asset = env.scene[asset_name]
        body_name = "right_tip" if asset_name != "cube" else "cube"
        asset_cfg = SceneEntityCfg(asset_name, body_names=[body_name])
        asset_cfg.resolve(env.scene)
        params = dict(
            asset_cfg=asset_cfg,
            static_friction_range=(0.4, 0.4),
            dynamic_friction_range=(0.8, 0.8),
            restitution_range=(0.1, 0.1),
            num_buckets=1,
            make_consistent=True,
        )
        # Read by USD body path, independently of the articulation's public ordering.
        views = {
            (i, name): env.sim.physics_sim_view.create_rigid_body_view(
                f"/World/envs/env_{i}/{asset_name}/{name}" if asset_name != "cube" else f"/World/envs/env_{i}/cube"
            )
            for i in range(env.num_envs)
            for name in asset.body_names
        }
        expected = {key: wp.to_torch(view.get_material_properties()).clone() for key, view in views.items()}
        term = mdp.randomize_rigid_body_material(
            EventTermCfg(func=mdp.randomize_rigid_body_material, params=params), env
        )
        term(env, selection, **params)
        selected = selection.start if isinstance(selection, slice) else selection.item()
        expected[selected, body_name][:] = expected[selected, body_name].new_tensor([0.4, 0.4, 0.1])
        for key, view in views.items():
            torch.testing.assert_close(wp.to_torch(view.get_material_properties()), expected[key])


def test_gravity_distribution(env: ManagerBasedEnv) -> None:
    """The configured Gaussian distribution changes acceleration in every environment."""
    gravity_params = dict(
        gravity_distribution_params=((0.0, 0.0, -3.0), (0.0, 0.0, 0.0)),
        distribution="gaussian",
        operation="abs",
    )
    gravity = mdp.randomize_physics_scene_gravity(
        EventTermCfg(func=mdp.randomize_physics_scene_gravity, params=gravity_params), env
    )
    gravity(env, torch.tensor([0], device=env.device), **gravity_params)
    cube = env.scene["cube"]
    for _ in range(2):
        env.sim.step()
        cube.update(env.sim.cfg.dt)
    expected = torch.zeros_like(cube.data.body_acc_w.torch)
    expected[..., 2] = -3.0
    torch.testing.assert_close(cube.data.body_acc_w.torch, expected, atol=1e-4, rtol=0.0)


def test_collider_offsets(env: ManagerBasedEnv) -> None:
    """Offset changes preserve other environments and unspecified properties."""
    asset_cfg = SceneEntityCfg("cube")
    view = env.scene["cube"].root_view
    original_rest = wp.to_torch(view.get_rest_offsets()).clone()
    original_contact = wp.to_torch(view.get_contact_offsets()).clone()
    offset_params = {
        "asset_cfg": asset_cfg,
        "rest_offset_distribution_params": (-0.01, -0.01),
        "contact_offset_distribution_params": (0.04, 0.04),
    }
    offsets = mdp.randomize_rigid_body_collider_offsets(
        EventTermCfg(func=mdp.randomize_rigid_body_collider_offsets, params=offset_params), env
    )
    offsets(env, slice(1, 2), **offset_params)
    expected_rest = original_rest.clone()
    expected_contact = original_contact.clone()
    expected_rest[1] = -0.01
    expected_contact[1] = 0.04
    torch.testing.assert_close(wp.to_torch(view.get_rest_offsets()), expected_rest)
    torch.testing.assert_close(wp.to_torch(view.get_contact_offsets()), expected_contact)

    offsets(env, torch.tensor([1], device=env.device), asset_cfg, contact_offset_distribution_params=(0.06, 0.06))
    expected_contact[1] = 0.06
    torch.testing.assert_close(wp.to_torch(view.get_rest_offsets()), expected_rest)
    torch.testing.assert_close(wp.to_torch(view.get_contact_offsets()), expected_contact)
    offsets(env, None, asset_cfg)
    torch.testing.assert_close(wp.to_torch(view.get_rest_offsets()), expected_rest)
    torch.testing.assert_close(wp.to_torch(view.get_contact_offsets()), expected_contact)


def test_fixed_tendon_parameters(env: ManagerBasedEnv) -> None:
    """Randomized tendon parameters reach the simulation for the selected environment."""
    asset = env.scene["identity"]
    assert asset.num_fixed_tendons == 1
    tendon_params = dict(
        asset_cfg=SceneEntityCfg("identity"),
        limit_stiffness_distribution_params=(2.0, 2.0),
        rest_length_distribution_params=(0.5, 0.5),
        operation="abs",
    )
    tendon = mdp.randomize_fixed_tendon_parameters(
        EventTermCfg(func=mdp.randomize_fixed_tendon_parameters, params=tendon_params), env
    )
    expected_stiffness = wp.to_torch(asset.root_view.get_fixed_tendon_limit_stiffnesses()).clone()
    expected_length = wp.to_torch(asset.root_view.get_fixed_tendon_rest_lengths()).clone()
    tendon(env, torch.tensor([1], device=env.device), **tendon_params)
    expected_stiffness[1] = 2.0
    expected_length[1] = 0.5
    torch.testing.assert_close(wp.to_torch(asset.root_view.get_fixed_tendon_limit_stiffnesses()), expected_stiffness)
    torch.testing.assert_close(wp.to_torch(asset.root_view.get_fixed_tendon_rest_lengths()), expected_length)
