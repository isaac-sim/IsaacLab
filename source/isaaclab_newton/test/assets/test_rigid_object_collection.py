# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless real-solver integration tests for Newton rigid-object collections."""

import pytest
import torch
import warp as wp
from isaaclab_newton.assets import RigidObjectCollection
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager
from newton import ModelFlags

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sim import SimulationCfg, build_simulation_context

pytestmark = pytest.mark.integration


def _newton_sim_context(device: str):
    """Create a fresh kitless Newton simulation context."""
    return build_simulation_context(
        sim_cfg=SimulationCfg(
            device=device,
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, 0.0),
            physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()),
        )
    )


def _local_cube_cfg(prim_path: str, y: float) -> RigidObjectCfg:
    """Create one local collection-body configuration."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyBaseCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionBaseCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, y, 1.0)),
    )


def _spawn_collection() -> RigidObjectCollection:
    """Author a two-by-two local collection plus an unrelated sibling body."""
    for env_index in range(2):
        sim_utils.create_prim(f"/World/Env_{env_index}", "Xform", translation=(3.0 * env_index, 0.0, 0.0))
    sibling_cfg = sim_utils.CuboidCfg(
        size=(0.2, 0.2, 0.2),
        rigid_props=sim_utils.RigidBodyBaseCfg(disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionBaseCfg(),
    )
    sibling_cfg.func("/World/Env_[^/]*/UnrelatedObject", sibling_cfg, translation=(0.0, -2.0, 1.0))
    return RigidObjectCollection(
        RigidObjectCollectionCfg(
            rigid_objects={
                "cube_0": _local_cube_cfg("/World/Env_[^/]*/Object_0", 0.0),
                "cube_1": _local_cube_cfg("/World/Env_[^/]*/Object_1", 1.0),
            }
        )
    )


def test_rigid_object_collection_real_newton_seams(monkeypatch) -> None:
    """Exercise exact model selection, partial state/property writes, and live gravity."""
    device = "cpu"
    with _newton_sim_context(device) as sim:
        collection = _spawn_collection()
        sim.reset()

        assert collection.is_initialized
        assert collection.num_instances == 2
        assert collection.body_names == ["cube_0", "cube_1"]
        assert collection.root_view.count == 4
        assert collection.data.body_mass.shape == (2, 2)
        assert collection.data.body_com_pos_b.shape == (2, 2)
        assert collection.data.body_inertia.shape == (2, 2, 9)
        torch.testing.assert_close(collection.data.body_link_pose_w.torch, collection.data.body_com_pose_w.torch)
        torch.testing.assert_close(
            collection.data.body_link_vel_w.torch[..., 3:],
            collection.data.body_com_vel_w.torch[..., 3:],
        )

        model_changes = []
        add_model_change = SimulationManager.add_model_change

        def record_model_change(change: ModelFlags) -> None:
            model_changes.append(change)
            add_model_change(change)

        monkeypatch.setattr(SimulationManager, "add_model_change", staticmethod(record_model_change))

        env_ids = torch.tensor([1], dtype=torch.int32, device=device)
        body_ids = torch.tensor([1], dtype=torch.int32, device=device)
        initial_pose = collection.data.body_link_pose_w.torch.clone()
        target_pose = initial_pose[env_ids][:, body_ids].clone()
        target_pose[..., :3] += torch.tensor([0.25, 0.5, 0.75], device=device)
        collection.write_body_link_pose_to_sim_index(
            body_poses=target_pose,
            env_ids=env_ids,
            body_ids=body_ids,
        )

        torch.testing.assert_close(collection.data.body_link_pose_w.torch[env_ids][:, body_ids], target_pose)
        torch.testing.assert_close(collection.data.body_link_pose_w.torch[0], initial_pose[0])
        torch.testing.assert_close(collection.data.body_link_pose_w.torch[1, :1], initial_pose[1, :1])

        initial_mass = collection.data.body_mass.torch.clone()
        initial_inv_mass = wp.to_torch(collection.data._sim_bind_body_inv_mass).clone()
        masses = torch.tensor([[5.0]], device=device)
        collection.set_masses_index(masses=masses, env_ids=env_ids, body_ids=body_ids)
        torch.testing.assert_close(collection.data.body_mass.torch[env_ids][:, body_ids], masses)
        torch.testing.assert_close(
            wp.to_torch(collection.data._sim_bind_body_inv_mass)[env_ids][:, body_ids], masses.reciprocal()
        )
        torch.testing.assert_close(collection.data.body_mass.torch[0], initial_mass[0])
        torch.testing.assert_close(collection.data.body_mass.torch[1, :1], initial_mass[1, :1])
        torch.testing.assert_close(wp.to_torch(collection.data._sim_bind_body_inv_mass)[0], initial_inv_mass[0])
        torch.testing.assert_close(wp.to_torch(collection.data._sim_bind_body_inv_mass)[1, :1], initial_inv_mass[1, :1])
        assert model_changes == [ModelFlags.BODY_INERTIAL_PROPERTIES]

        model_changes.clear()
        model = SimulationManager.get_model()
        material_mu = collection.root_view.get_attribute("shape_material_mu", model)
        initial_material_mu = wp.to_torch(material_mu).clone()
        material_values = wp.full(material_mu.shape, value=0.65, dtype=wp.float32, device=device)
        material_mask = wp.array([[False, False], [False, True]], dtype=wp.bool, device=device)
        collection.root_view.set_attribute("shape_material_mu", model, material_values, material_mask)
        SimulationManager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
        expected_material_mu = initial_material_mu.clone()
        expected_material_mu[1, 1] = 0.65
        torch.testing.assert_close(wp.to_torch(material_mu), expected_material_mu)
        assert model_changes == [ModelFlags.SHAPE_PROPERTIES]

        model_changes.clear()
        model_gravity = model.gravity[: model.world_count]
        new_gravity = torch.tensor([[0.0, 0.0, -2.0], [0.0, -3.0, -4.0]], device=device)
        wp.to_torch(model_gravity).copy_(new_gravity)
        SimulationManager.add_model_change(ModelFlags.MODEL_PROPERTIES)
        collection.update(0.0)

        torch.testing.assert_close(collection.data.GRAVITY_VEC_W.torch, new_gravity)
        expected_projected = torch.nn.functional.normalize(new_gravity, dim=-1).unsqueeze(1).expand(-1, 2, -1)
        torch.testing.assert_close(collection.data.projected_gravity_b.torch, expected_projected, atol=1e-6, rtol=1e-6)
        assert model_changes == [ModelFlags.MODEL_PROPERTIES]
