# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal real-PhysX integration coverage for rigid-object collections."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, device="cpu").app

import pytest
import torch
import warp as wp
from isaaclab_physx.assets import RigidObjectCollection

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sim import build_simulation_context

pytestmark = pytest.mark.integration


def _cube_cfg(prim_path: str, y: float) -> RigidObjectCfg:
    """Create one local collection-body configuration."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, y, 1.0)),
    )


def _spawn_collection() -> RigidObjectCollection:
    """Author a two-instance, two-body local collection."""
    for env_index in range(2):
        sim_utils.create_prim(f"/World/Env_{env_index}", "Xform", translation=(3.0 * env_index, 0.0, 0.0))
    return RigidObjectCollection(
        RigidObjectCollectionCfg(
            rigid_objects={
                "left": _cube_cfg("/World/Env_[^/]*/Object_0", 0.0),
                "right": _cube_cfg("/World/Env_[^/]*/Object_1", 1.0),
            }
        )
    )


def test_rigid_object_collection_real_physx_seams() -> None:
    """Prove body-major view remapping through nontrivial partial state and inertial writes."""
    with build_simulation_context(device="cpu", gravity_enabled=False) as sim:
        collection = _spawn_collection()
        sim.reset()

        assert collection.is_initialized
        assert collection.num_instances == 2
        assert collection.body_names == ["left", "right"]
        assert collection.data.body_mass.torch.shape == (2, 2)
        assert collection.data.body_com_pose_b.torch.shape == (2, 2, 7)
        assert collection.data.body_inertia.torch.shape == (2, 2, 9)

        env_ids = torch.tensor([1, 0], dtype=torch.int32)
        body_ids = torch.tensor([1], dtype=torch.int32)
        initial_pose = collection.data.body_link_pose_w.torch.clone()
        target_pose = initial_pose[env_ids][:, body_ids].clone()
        target_pose[0, 0, :3] += torch.tensor([0.2, 0.3, 0.4])
        target_pose[1, 0, :3] += torch.tensor([-0.1, -0.2, 0.1])
        collection.write_body_link_pose_to_sim_index(body_poses=target_pose, env_ids=env_ids, body_ids=body_ids)
        torch.testing.assert_close(collection.data.body_link_pose_w.torch[env_ids][:, body_ids], target_pose)
        torch.testing.assert_close(collection.data.body_link_pose_w.torch[:, :1], initial_pose[:, :1])

        initial_masses = collection.data.body_mass.torch.clone()
        initial_coms = collection.data.body_com_pose_b.torch.clone()
        initial_inertias = collection.data.body_inertia.torch.clone()
        masses = torch.tensor([[5.0], [7.0]])
        collection.set_masses_index(masses=masses, env_ids=env_ids, body_ids=body_ids)
        torch.testing.assert_close(collection.data.body_mass.torch[env_ids][:, body_ids], masses)
        raw_mass = wp.to_torch(collection.root_view.get_masses()).reshape(2, 2).T
        torch.testing.assert_close(raw_mass[env_ids][:, body_ids], masses)
        torch.testing.assert_close(collection.data.body_mass.torch[:, :1], initial_masses[:, :1])
        torch.testing.assert_close(raw_mass[:, :1], initial_masses[:, :1])

        coms = collection.data.body_com_pose_b.torch[env_ids][:, body_ids].clone()
        coms[0, 0, :3] = torch.tensor([0.02, 0.03, 0.04])
        coms[1, 0, :3] = torch.tensor([-0.01, 0.01, 0.02])
        collection.set_coms_index(coms=coms, env_ids=env_ids, body_ids=body_ids)
        torch.testing.assert_close(collection.data.body_com_pose_b.torch[env_ids][:, body_ids], coms)
        raw_coms = wp.to_torch(collection.root_view.get_coms().view(wp.float32)).reshape(2, 2, 7).transpose(0, 1)
        torch.testing.assert_close(raw_coms[env_ids][:, body_ids], coms)
        torch.testing.assert_close(collection.data.body_com_pose_b.torch[:, :1], initial_coms[:, :1])
        torch.testing.assert_close(raw_coms[:, :1], initial_coms[:, :1])

        inertias = collection.data.body_inertia.torch[env_ids][:, body_ids].clone()
        inertias[0, 0, 0] *= 1.2
        inertias[1, 0, 4] *= 1.3
        collection.set_inertias_index(inertias=inertias, env_ids=env_ids, body_ids=body_ids)
        torch.testing.assert_close(collection.data.body_inertia.torch[env_ids][:, body_ids], inertias)
        raw_inertias = wp.to_torch(collection.root_view.get_inertias()).reshape(2, 2, 9).transpose(0, 1)
        torch.testing.assert_close(raw_inertias[env_ids][:, body_ids], inertias)
        torch.testing.assert_close(collection.data.body_inertia.torch[:, :1], initial_inertias[:, :1])
        torch.testing.assert_close(raw_inertias[:, :1], initial_inertias[:, :1])

        materials = torch.tensor(
            [
                [[0.9, 0.4, 0.1], [0.8, 0.3, 0.2]],
                [[0.7, 0.2, 0.3], [0.6, 0.1, 0.4]],
            ]
        )
        view_materials = collection.reshape_data_to_view_3d(wp.from_torch(materials, dtype=wp.float32), 3, device="cpu")
        view_ids = wp.array([0, 1, 2, 3], dtype=wp.int32, device="cpu")
        collection.root_view.set_material_properties(view_materials, view_ids)
        raw_materials = collection.reshape_view_to_data_3d(
            collection.root_view.get_material_properties(), 3, device="cpu"
        )
        torch.testing.assert_close(wp.to_torch(raw_materials), materials)
