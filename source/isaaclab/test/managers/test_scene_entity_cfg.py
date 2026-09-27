# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene selections resolve on the host, then finalize to device indices used without synchronization."""

import copy
import json
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from isaaclab.envs.mdp.events import apply_external_force_torque
from isaaclab.envs.mdp.observations import joint_pos
from isaaclab.managers import SceneEntityCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import index_fill_, replace_slices_with_strings, torch_index
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.unit


class Scene(dict):
    """Only scene lookup and the device are required to resolve static selections."""

    device = "cpu"


@pytest.fixture
def scene():
    names = ["part_0", "part_1", "part_2", "part_3"]
    find = partial(resolve_matching_names, list_of_strings=names)
    return Scene(
        robot=SimpleNamespace(
            joint_names=names,
            body_names=names,
            fixed_tendon_names=names,
            object_names=names,
            num_joints=4,
            num_bodies=4,
            num_fixed_tendons=4,
            find_joints=find,
            find_bodies=find,
            find_fixed_tendons=find,
            find_objects=find,
        )
    )


@pytest.mark.parametrize("device", test_devices())
def test_finalize_moves_resolved_selections_to_device(scene, device):
    """Ordering, slices, and empty selections survive finalization, copies, and serialization."""
    cfg = SceneEntityCfg(
        "robot",
        joint_names=["part_2", "part_0"],
        preserve_order=True,
        body_ids=[3, 1],
        object_collection_ids=[],
    )
    cfg.resolve(scene)
    # class-based terms read host selections before finalization
    assert cfg.joint_ids == [2, 0]
    assert cfg.body_ids == [3, 1]

    cfg.finalize(device)
    joint_ids = cfg.joint_ids
    cfg.finalize(device)
    assert cfg.joint_ids is joint_ids
    assert isinstance(cfg.body_ids, ProxyArray)
    assert cfg.body_ids.torch.dtype == torch.long
    assert cfg.body_ids.torch.device == torch.device(device)
    assert cfg.fixed_tendon_ids == slice(None)
    values = torch.arange(4, device=device)
    assert values[torch_index(cfg.joint_ids)].tolist() == [2, 0]
    assert values[torch_index(cfg.body_ids)].tolist() == [3, 1]
    assert values[torch_index(cfg.fixed_tendon_ids)].tolist() == [0, 1, 2, 3]
    assert values[torch_index(cfg.object_collection_ids)].numel() == 0

    snapshot = cfg.to_dict()
    assert snapshot["joint_ids"] == [2, 0]
    assert snapshot["body_ids"] == [3, 1]
    json.dumps(replace_slices_with_strings(snapshot))
    for copied in (cfg.copy(), copy.deepcopy(cfg)):
        assert isinstance(copied.body_ids, ProxyArray)
        assert values[torch_index(copied.body_ids)].tolist() == [3, 1]


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_finalized_indices_do_not_upload_or_read_back(scene, device):
    """Exercise a real MDP consumer, reads, and writes with CUDA synchronization checks."""
    scene.device = device
    cfg = SceneEntityCfg("robot", joint_ids=[2, 0])
    cfg.resolve(scene)
    cfg.finalize(device)
    values = torch.arange(16, dtype=torch.float, device=device).reshape(4, 4)
    destination = torch.zeros_like(values)
    replacements = torch.full((4, 2), 7.0, device=device)
    scene["robot"].data = SimpleNamespace(joint_pos=SimpleNamespace(torch=values))
    env = SimpleNamespace(scene=scene)
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.synchronize(device)
    torch.cuda.set_sync_debug_mode("error")
    try:
        observed = joint_pos(env, cfg)
        indices = torch_index(cfg.joint_ids)
        selected = torch.index_select(values, dim=1, index=indices)
        first = values[:, indices][:, 0]
        destination[:, indices] = replacements
        index_fill_(destination, indices, 9.0, dim=1)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    expected = torch.tensor([[2.0, 0.0], [6.0, 4.0], [10.0, 8.0], [14.0, 12.0]], device=device)
    torch.testing.assert_close(observed, expected)
    torch.testing.assert_close(selected, expected)
    torch.testing.assert_close(first, expected[:, 0])
    torch.testing.assert_close(destination, torch.tensor([[9.0, 0.0, 9.0, 0.0]], device=device).repeat(4, 1))


def test_external_force_uses_selected_body_count(scene):
    """A finalized subset must size forces for that subset, rather than every body."""
    cfg = SceneEntityCfg("robot", body_ids=[3, 1])
    cfg.resolve(scene)
    cfg.finalize(scene.device)
    asset = scene["robot"]
    asset.device = "cpu"
    asset.permanent_wrench_composer = SimpleNamespace(set_forces_and_torques_index=Mock())
    env = SimpleNamespace(scene=scene, num_envs=2)
    apply_external_force_torque(env, None, (2.0, 2.0), (3.0, 3.0), cfg)
    kwargs = asset.permanent_wrench_composer.set_forces_and_torques_index.call_args.kwargs
    torch.testing.assert_close(kwargs["forces"], torch.full((2, 2, 3), 2.0))
    torch.testing.assert_close(kwargs["torques"], torch.full((2, 2, 3), 3.0))
    assert kwargs["body_ids"].tolist() == [3, 1]
