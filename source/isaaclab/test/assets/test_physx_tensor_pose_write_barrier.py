# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the PhysX tensor-pose write barrier."""

from unittest.mock import patch

import pytest
import warp as wp
from _articulation_iface_test_utils import BACKENDS as ARTICULATION_BACKENDS
from _articulation_iface_test_utils import get_articulation
from _rigid_object_collection_iface_test_utils import BACKENDS as COLLECTION_BACKENDS
from _rigid_object_collection_iface_test_utils import get_rigid_object_collection
from _rigid_object_iface_test_utils import BACKENDS as RIGID_OBJECT_BACKENDS
from _rigid_object_iface_test_utils import get_rigid_object
from isaaclab_physx.assets.articulation import articulation as articulation_module
from isaaclab_physx.assets.rigid_object import rigid_object as rigid_object_module
from isaaclab_physx.assets.rigid_object_collection import rigid_object_collection as collection_module

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not all(
            "physx" in backends for backends in (RIGID_OBJECT_BACKENDS, COLLECTION_BACKENDS, ARTICULATION_BACKENDS)
        ),
        reason="PhysX backend is unavailable",
    ),
]


@pytest.mark.parametrize(
    ("asset_kind", "method_name"),
    [
        ("rigid_object", "write_root_link_pose_to_sim_index"),
        ("rigid_object", "write_root_com_pose_to_sim_index"),
        ("collection", "write_body_link_pose_to_sim_index"),
        ("collection", "write_body_com_pose_to_sim_index"),
        ("articulation", "write_root_link_pose_to_sim_index"),
        ("articulation", "write_root_com_pose_to_sim_index"),
    ],
)
def test_pose_writer_notifies_tensor_pose_write(asset_kind: str, method_name: str):
    if asset_kind == "rigid_object":
        asset, _ = get_rigid_object("physx", num_instances=2, device="cpu")
        module = rigid_object_module
        argument_name = "root_pose"
        pose = wp.zeros((asset.num_instances,), dtype=wp.transformf, device="cpu")
    elif asset_kind == "collection":
        asset, _ = get_rigid_object_collection("physx", num_instances=2, num_bodies=3, device="cpu")
        module = collection_module
        argument_name = "body_poses"
        pose = wp.zeros((asset.num_instances, asset.num_bodies), dtype=wp.transformf, device="cpu")
    else:
        asset, _ = get_articulation("physx", num_instances=2, num_joints=3, num_bodies=4, device="cpu")
        module = articulation_module
        argument_name = "root_pose"
        pose = wp.zeros((asset.num_instances,), dtype=wp.transformf, device="cpu")

    with patch.object(module.SimulationManager, "notify_tensor_pose_write") as notify:
        getattr(asset, method_name)(**{argument_name: pose})

    notify.assert_called_once_with()
