# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless real-Newton coverage of a rigid object spawned with :class:`~isaaclab.sim.SwitchableAssetsCfg`.

Each of two environments holds one object whose candidates are a 1 kg cube (0.2 m) and a 2 kg sphere
(0.15 m radius), dropped onto a shared ground plane.
"""

from isaaclab_newton.physics import NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg()))

from types import SimpleNamespace

import newton
import pytest
import torch
import warp as wp
from isaaclab_newton.assets import RigidObject
from isaaclab_newton.physics import NewtonCollisionPipelineCfg
from newton_test_utils import env_origins, newton_sim_cfg

from pxr import Gf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.cloner import CloneCfg, clone_plan_from_env_0, replicate
from isaaclab.sim import build_simulation_context

pytestmark = [pytest.mark.integration, pytest.mark.kitless]

_DT = 1.0 / 120.0


def _object_cfg(prim_path: str = "/World/Env_[^/]*/Object") -> RigidObjectCfg:
    """Return a switchable object with a cube and a sphere candidate."""
    collision = sim_utils.UsdPhysicsCollisionCfg()
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.SwitchableAssetsCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(0.2, 0.2, 0.2), mass_props=sim_utils.MassCfg(mass=1.0), collision_props=collision
                ),
                sim_utils.SphereCfg(radius=0.15, mass_props=sim_utils.MassCfg(mass=2.0), collision_props=collision),
            ],
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )


def _settle(sim, obj: RigidObject, env_ids: list[int]) -> torch.Tensor:
    """Drop the object from its default pose in ``env_ids`` and return the resting heights of all environments."""
    pose = obj.data.default_root_pose.torch[env_ids].clone()
    pose[:, :3] += env_origins(sim.device)[env_ids]
    obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(env_ids), 6, device=sim.device), env_ids=env_ids)
    for _ in range(240):
        sim.step(render=False)
        obj.update(_DT)
    return obj.data.root_pos_w.torch[:, 2]


def test_switchable_object_follows_selected_candidate():
    """Mass, collision shapes and published prim poses follow the selected candidate of each environment, and
    viewers of the model are told to redraw once per switch."""
    collision_cfg = NewtonCollisionPipelineCfg(broad_phase="sap")
    sim_cfg = newton_sim_cfg(dt=_DT, gravity=(0.0, 0.0, -9.81), newton_contacts=True, collision_cfg=collision_cfg)
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/Ground", ground_cfg)
        sim_utils.create_prim("/World/Env_0", "Xform")
        cfg = _object_cfg()
        plan = clone_plan_from_env_0(
            CloneCfg(clone_template="/World/Env_{}"),
            (cfg, AssetBaseCfg(prim_path="/World/Ground")),
            2,
            40.0,
            positions=env_origins("cpu").numpy(),
        )
        obj = RigidObject(cfg)
        replicate(plan)
        sim.reset()

        backend = sim.backend_manager.backend
        model = backend.model
        assert obj.num_instances == 2 and obj.num_bodies == 1
        assert model.body_variant_labels(0) == model.body_variant_labels(1) == ["asset_0", "asset_1"]
        assert wp.to_torch(model.body_variant).tolist().count(0) == 2
        # Each candidate is one collider, so each body holds its candidates in one variant shape row.
        rows = [model.body_shapes[body] for body in range(2)]
        assert [len(shapes) for shapes in rows] == [1, 1]
        rows = [shapes[0] for shapes in rows]
        selected = [0, 0]
        changes = []
        viewer = SimpleNamespace(notify_model_changed=changes.append)
        version = backend.notify_viewer(viewer, backend.variants_version)

        def select(variants: list[int], env_ids: list[int] | None = None) -> None:
            """Select candidates and notify the viewer twice, which redraws once."""
            nonlocal version
            variant_ids = torch.tensor(variants, dtype=torch.int32, device=sim.device)
            obj.replace_mesh_to_sim(variant_ids=variant_ids, env_ids=env_ids)
            for env, variant in zip(range(2) if env_ids is None else env_ids, variants):
                selected[env] = variant
            # The row of each environment holds the selected candidate's visible shape.
            shapes = [newton.GeoType.SPHERE if variant else newton.GeoType.BOX for variant in selected]
            assert model.shape_type.numpy()[rows].tolist() == shapes
            assert (model.shape_flags.numpy()[rows] & int(newton.ShapeFlags.VISIBLE)).all()
            version = backend.notify_viewer(viewer, backend.notify_viewer(viewer, version))

        select([0, 1])
        heights = _settle(sim, obj, [0, 1])
        torch.testing.assert_close(obj.data.body_mass.torch[:, 0], torch.tensor([1.0, 2.0], device=sim.device))
        torch.testing.assert_close(heights, torch.tensor([0.1, 0.15], device=sim.device), atol=5e-3, rtol=0.0)

        # Renderers see the selected candidate at the body pose and the other one out of view.
        scene_data = sim.backend_manager._scene_data_backend
        paths = scene_data.transform_paths
        transforms = wp.to_torch(scene_data.transforms.transforms)
        assert scene_data.transform_count == len(paths) == transforms.shape[0]
        body = paths.index("/World/Env_1/Object")
        torch.testing.assert_close(transforms[paths.index("/World/Env_1/Object/asset_1")], transforms[body])
        assert transforms[paths.index("/World/Env_1/Object/asset_0"), 2] < -1000.0

        select([1], [0])
        heights = _settle(sim, obj, [0])
        torch.testing.assert_close(obj.data.body_mass.torch[:, 0], torch.tensor([2.0, 2.0], device=sim.device))
        torch.testing.assert_close(heights, torch.tensor([0.15, 0.15], device=sim.device), atol=5e-3, rtol=0.0)
        assert changes == [newton.ModelFlags.SHAPE_TYPE_AND_SOURCE] * 2


def test_switchable_object_under_a_parent_source():
    """A switchable object declared beneath another asset is one body of the parent's import, not one per candidate."""
    sim_cfg = newton_sim_cfg(dt=_DT, gravity=(0.0, 0.0, -9.81), newton_contacts=True)
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim_utils.create_prim("/World/Env_0", "Xform")
        table = AssetBaseCfg(
            prim_path="/World/Env_[^/]*/Table",
            spawn=sim_utils.CuboidCfg(size=(1.0, 1.0, 0.1), collision_props=sim_utils.UsdPhysicsCollisionCfg()),
        )
        table.spawn.func("/World/Env_0/Table", table.spawn)
        cfg = _object_cfg("/World/Env_[^/]*/Table/Object")
        plan = clone_plan_from_env_0(
            CloneCfg(clone_template="/World/Env_{}"), (table, cfg), 2, 40.0, positions=env_origins("cpu").numpy()
        )
        obj = RigidObject(cfg)
        replicate(plan)
        sim.reset()

        model = sim.backend_manager.backend.model
        assert obj.num_instances == 2 and obj.num_bodies == 1
        assert model.body_variant_labels(0) == model.body_variant_labels(1) == ["asset_0", "asset_1"]
        obj.replace_mesh_to_sim(variant_ids=torch.tensor([0, 1], dtype=torch.int32, device=sim.device))
        sim.step(render=False)
        obj.update(_DT)
        torch.testing.assert_close(obj.data.body_mass.torch[:, 0], torch.tensor([1.0, 2.0], device=sim.device))


def test_switching_refreshes_com_and_render_data():
    """Switching candidates within a step updates the cached center-of-mass pose and republishes the transforms."""
    sim_cfg = newton_sim_cfg(dt=_DT, gravity=(0.0, 0.0, -9.81), newton_contacts=True)
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim_utils.create_prim("/World/Env_0", "Xform")
        cfg = _object_cfg()
        plan = clone_plan_from_env_0(
            CloneCfg(clone_template="/World/Env_{}"), (cfg,), 2, 40.0, positions=env_origins("cpu").numpy()
        )
        obj = RigidObject(cfg)
        # The cube's center of mass sits 5 cm above its origin; the sphere's at its origin.
        cube = sim_utils.find_first_matching_prim("/World/Env_0/Object/asset_0")
        UsdPhysics.MassAPI(cube).CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.05))
        replicate(plan)
        sim.reset()
        obj.replace_mesh_to_sim(variant_ids=torch.tensor([0, 0], dtype=torch.int32, device=sim.device))
        sim.step(render=False)
        obj.update(_DT)

        def com_height() -> torch.Tensor:
            return obj.data.root_com_pose_w.torch[:, 2] - obj.data.root_link_pose_w.torch[:, 2]

        torch.testing.assert_close(com_height(), torch.tensor([0.05, 0.05], device=sim.device), atol=1e-4, rtol=0.0)
        backend = sim.backend_manager._scene_data_backend
        timestamp = backend.transforms_timestamp
        obj.replace_mesh_to_sim(variant_ids=torch.tensor([1], dtype=torch.int32, device=sim.device), env_ids=[1])
        torch.testing.assert_close(com_height(), torch.tensor([0.05, 0.0], device=sim.device), atol=1e-4, rtol=0.0)
        assert backend.transforms_timestamp > timestamp


def test_spawner_applies_visibility_and_semantic_tags():
    """The spawned prim takes the configuration's visibility, which its candidates inherit, and semantic tags."""
    with build_simulation_context(sim_cfg=newton_sim_cfg(dt=_DT)):
        sim_utils.create_prim("/World/Env_0", "Xform")
        cfg = _object_cfg().spawn.replace(visible=False, semantic_tags=[("class", "toy")])
        prim = cfg.func("/World/Env_0/Object", cfg)
        for candidate in (prim, prim.GetChild("asset_0"), prim.GetChild("asset_1")):
            assert UsdGeom.Imageable(candidate).ComputeVisibility() == UsdGeom.Tokens.invisible
        assert sim_utils.get_labels(prim) == {"class": ["toy"]}
