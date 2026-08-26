# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal real-OVPhysX integration coverage for rigid-object collections."""

from __future__ import annotations

import pytest
import torch
import warp as wp

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov import tensor_types as TT  # noqa: E402
from isaaclab_ov.assets import RigidObjectCollection  # noqa: E402
from isaaclab_ov.physics import OvPhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402

pytestmark = pytest.mark.integration


def _sim_context():
    """Build a local CPU OVPhysX context from an in-memory USD stage."""
    return build_simulation_context(
        sim_cfg=SimulationCfg(physics=OvPhysxCfg(), device="cpu", gravity=(0.0, 0.0, 0.0)),
        auto_add_lighting=False,
    )


def _cube_cfg(prim_path: str, y: float) -> RigidObjectCfg:
    """Create one local collection body."""
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
    """Author the canonical N=2, B=3 local collection."""
    for env_index in range(2):
        sim_utils.create_prim(f"/World/Env_{env_index}", "Xform", translation=(3.0 * env_index, 0.0, 0.0))
    return RigidObjectCollection(
        RigidObjectCollectionCfg(
            rigid_objects={
                "left": _cube_cfg("/World/Env_[^/]*/Object_0", 0.0),
                "middle": _cube_cfg("/World/Env_[^/]*/Object_1", 1.0),
                "right": _cube_cfg("/World/Env_[^/]*/Object_2", 2.0),
            }
        )
    )


def test_rigid_object_collection_real_ovphysx_seams() -> None:
    """Prove fused remapping through partial state, inertial, and material writes."""
    with _sim_context() as sim:
        collection = _spawn_collection()
        sim.reset()

        assert collection.is_initialized
        assert collection.num_instances == 2
        assert collection.body_names == ["left", "middle", "right"]
        assert collection.data.body_mass.torch.shape == (2, 3)

        env_ids = torch.tensor([1, 0], dtype=torch.int32)
        body_ids = torch.tensor([2, 0], dtype=torch.int32)
        initial_pose = collection.data.body_link_pose_w.torch.clone()
        target_pose = initial_pose[env_ids][:, body_ids].clone()
        target_pose[0, 0, :3] += torch.tensor([0.2, 0.3, 0.4])
        target_pose[1, 1, :3] += torch.tensor([-0.1, -0.2, 0.1])
        collection.write_body_link_pose_to_sim_index(body_poses=target_pose, env_ids=env_ids, body_ids=body_ids)
        torch.testing.assert_close(collection.data.body_link_pose_w.torch[env_ids][:, body_ids], target_pose)
        torch.testing.assert_close(collection.data.body_link_pose_w.torch[:, 1], initial_pose[:, 1])

        initial_mass = collection.data.body_mass.torch.clone()
        raw_mass_before = wp.to_torch(collection.root_view.get_attribute(TT.BODY_MASS)).reshape(3, 2).T.clone()
        masses = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        collection.set_masses_index(masses=masses, env_ids=env_ids, body_ids=body_ids)
        expected_raw_mass = raw_mass_before.clone()
        expected_raw_mass[env_ids[:, None], body_ids[None, :]] = masses
        raw_mass = wp.to_torch(collection.root_view.get_attribute(TT.BODY_MASS)).reshape(3, 2).T
        torch.testing.assert_close(raw_mass, expected_raw_mass)
        torch.testing.assert_close(collection.data.body_mass.torch[env_ids][:, body_ids], masses)
        torch.testing.assert_close(collection.data.body_mass.torch[:, 1], initial_mass[:, 1])

        raw_com_before = (
            wp.to_torch(collection.root_view.get_attribute(TT.BODY_COM_POSE)).reshape(3, 2, 7).transpose(0, 1).clone()
        )
        coms = collection.data.body_com_pose_b.torch[env_ids][:, body_ids].clone()
        coms[..., :3] = torch.tensor(
            [[[0.01, 0.02, 0.03], [-0.01, 0.03, 0.02]], [[0.02, -0.01, 0.01], [0.03, 0.01, -0.02]]]
        )
        collection.set_coms_index(coms=coms, env_ids=env_ids, body_ids=body_ids)
        expected_raw_com = raw_com_before.clone()
        expected_raw_com[env_ids[:, None], body_ids[None, :]] = coms
        raw_com = wp.to_torch(collection.root_view.get_attribute(TT.BODY_COM_POSE)).reshape(3, 2, 7).transpose(0, 1)
        torch.testing.assert_close(raw_com, expected_raw_com)
        torch.testing.assert_close(collection.data.body_com_pose_b.torch[env_ids][:, body_ids], coms)

        raw_inertia_before = (
            wp.to_torch(collection.root_view.get_attribute(TT.BODY_INERTIA)).reshape(3, 2, 9).transpose(0, 1).clone()
        )
        inertias = collection.data.body_inertia.torch[env_ids][:, body_ids].clone()
        inertias[..., 0] *= 1.2
        inertias[..., 4] *= 1.3
        collection.set_inertias_index(inertias=inertias, env_ids=env_ids, body_ids=body_ids)
        expected_raw_inertia = raw_inertia_before.clone()
        expected_raw_inertia[env_ids[:, None], body_ids[None, :]] = inertias
        raw_inertia = wp.to_torch(collection.root_view.get_attribute(TT.BODY_INERTIA)).reshape(3, 2, 9).transpose(0, 1)
        torch.testing.assert_close(raw_inertia, expected_raw_inertia)
        torch.testing.assert_close(collection.data.body_inertia.torch[env_ids][:, body_ids], inertias)

        materials = torch.tensor(
            [
                [[0.9, 0.4, 0.1], [0.8, 0.3, 0.2], [0.7, 0.2, 0.3]],
                [[0.6, 0.1, 0.4], [0.5, 0.2, 0.1], [0.4, 0.3, 0.2]],
            ]
        )
        fused_materials = collection.reshape_data_to_view_3d(
            wp.from_torch(materials, dtype=wp.float32), 3, device="cpu"
        )
        collection.root_view.set_attribute(TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION, fused_materials)
        raw_materials = wp.to_torch(collection.root_view.get_attribute(TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION))
        torch.testing.assert_close(raw_materials.reshape(3, 2, 3).transpose(0, 1), materials)
