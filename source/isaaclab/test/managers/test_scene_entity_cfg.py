# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native scene selectors avoid index transfers during stepping and serialize as host lists."""

import copy
import json
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import warp as wp

from isaaclab.envs.mdp.events import apply_external_force_torque
from isaaclab.envs.mdp.observations import joint_pos
from isaaclab.envs.mdp.rewards import position_command_error
from isaaclab.managers import SceneEntityCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import class_to_dict, index_fill_, replace_slices_with_strings
from isaaclab.utils.string import resolve_matching_names

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


def device_ids(cfg, kind):
    indices = getattr(cfg, f"{kind}_ids")
    return indices


def test_resolved_selectors_survive_copy_serialization_and_rebinding(scene):
    """Ordering, duplicates, negative IDs, slices, and empty selections retain their meaning."""
    cfg = SceneEntityCfg(
        "robot",
        joint_names=["part_2", "part_0"],
        preserve_order=True,
        body_ids=[3, 1, 3, -1],
        fixed_tendon_ids=slice(1, None, 2),
        object_collection_ids=[],
    )
    cfg.resolve(scene)
    assert cfg.joint_ids.tolist() == [2, 0]
    assert cfg.body_ids[0] == 3
    assert cfg.body_ids.tolist() == [3, 1, 3, -1]
    assert cfg.fixed_tendon_ids == slice(1, None, 2)
    assert cfg.object_collection_ids.tolist() == []
    values = torch.arange(4)
    assert values[device_ids(cfg, "body")].tolist() == [3, 1, 3, 3]
    assert values[device_ids(cfg, "fixed_tendon")].tolist() == [1, 3]
    assert values[device_ids(cfg, "object_collection")].numel() == 0

    from isaaclab.envs.utils.io_descriptors import record_body_names, record_joint_names

    descriptor = SimpleNamespace()
    env = SimpleNamespace(scene=scene)
    record_joint_names(None, descriptor, env=env, asset_cfg=cfg)
    record_body_names(None, descriptor, env=env, asset_cfg=cfg)
    assert descriptor.joint_names == ["part_2", "part_0"]
    assert descriptor.body_names == ["part_3", "part_1", "part_3", "part_3"]
    snapshot = cfg.to_dict()
    assert snapshot["joint_ids"] == [2, 0]
    assert snapshot["body_ids"] == [3, 1, 3, -1]
    json.dumps(replace_slices_with_strings(snapshot))
    assert class_to_dict({"params": {"asset_cfg": cfg}})["params"]["asset_cfg"] == snapshot
    for copied in (cfg.copy(), copy.deepcopy(cfg), SceneEntityCfg(**snapshot)):
        copied.resolve(scene)
        assert copied.to_dict() == snapshot
        assert values[device_ids(copied, "joint")].tolist() == [2, 0]

    changed = cfg.replace(joint_names=None, joint_ids=[1])
    changed.resolve(scene)
    assert values[device_ids(changed, "joint")].tolist() == [1]
    assert values[device_ids(cfg, "joint")].tolist() == [2, 0]


def test_regex_consistency_and_repeated_resolution(scene):
    cfg = SceneEntityCfg("robot", joint_names="part_[02]")
    cfg.resolve(scene)
    cfg.resolve(scene)
    assert cfg.joint_ids.tolist() == [0, 2]
    cfg.joint_ids = [1]
    with pytest.raises(ValueError, match="not consistent"):
        cfg.resolve(scene)

    all_parts = SceneEntityCfg("robot", body_names=scene["robot"].body_names)
    all_parts.resolve(scene)
    assert all_parts.body_ids == slice(None)
    assert device_ids(all_parts, "body") == slice(None)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_resolved_indices_do_not_upload_or_read_back_during_torch_operations(scene, device):
    """Exercise MDP reads, single-body rewards, and indexed writes without CUDA readback."""
    scene.device = device
    cfg = SceneEntityCfg("robot", joint_ids=[2, 0], body_ids=[2, 0])
    cfg.resolve(scene)
    indices = device_ids(cfg, "joint")
    values = torch.arange(16, dtype=torch.float, device=device).reshape(4, 4)
    destination = torch.zeros_like(values)
    replacements = torch.full((4, 2), 7.0, device=device)
    root_pos = torch.zeros((4, 3), device=device)
    root_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).repeat(4, 1)
    body_pos = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]], device=device
    ).repeat(4, 1, 1)
    scene["robot"].data = SimpleNamespace(
        joint_pos=SimpleNamespace(torch=values),
        root_pos_w=SimpleNamespace(torch=root_pos),
        root_quat_w=SimpleNamespace(torch=root_quat),
        body_pos_w=SimpleNamespace(torch=body_pos),
    )
    env = SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_command=lambda _: root_pos))
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.synchronize(device)
    torch.cuda.set_sync_debug_mode("error")
    try:
        observed = joint_pos(env, cfg)
        selected = torch.index_select(values, dim=1, index=indices)
        destination[:, indices] = replacements
        index_fill_(destination, indices, 9.0, dim=1)
        position_error = position_command_error(env, "pose", cfg)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.testing.assert_close(position_error, torch.full((4,), 5.0, device=device))
    # Host consumers explicitly read back only outside the step loop.
    assert cfg.joint_ids.tolist() == [2, 0]
    json.dumps(replace_slices_with_strings(cfg.to_dict()))
    expected = torch.tensor([[2.0, 0.0], [6.0, 4.0], [10.0, 8.0], [14.0, 12.0]], device=device)
    torch.testing.assert_close(observed, expected)
    torch.testing.assert_close(selected, expected)
    torch.testing.assert_close(destination, torch.tensor([[9.0, 0.0, 9.0, 0.0]], device=device).repeat(4, 1))


def test_external_force_uses_selected_body_count(scene):
    """A resolved subset must size forces for that subset, rather than every body."""
    cfg = SceneEntityCfg("robot", body_ids=[3, 1])
    cfg.resolve(scene)
    asset = scene["robot"]
    asset.device = "cpu"
    asset.permanent_wrench_composer = SimpleNamespace(set_forces_and_torques_index=Mock())
    env = SimpleNamespace(scene=scene, num_envs=2)
    apply_external_force_torque(env, None, (2.0, 2.0), (3.0, 3.0), cfg)
    kwargs = asset.permanent_wrench_composer.set_forces_and_torques_index.call_args.kwargs
    torch.testing.assert_close(kwargs["forces"], torch.full((2, 2, 3), 2.0))
    torch.testing.assert_close(kwargs["torques"], torch.full((2, 2, 3), 3.0))


def test_tensor_input_validation(scene):
    """Reject ambiguous masks or non-vector tensors before using them as integer selectors."""
    for indices in (torch.tensor([True, False]), torch.tensor([[1, 2]])):
        with pytest.raises(ValueError, match="one-dimensional integer tensors"):
            SceneEntityCfg("robot", joint_ids=indices).resolve(scene)


def test_experimental_selectors_feed_warp_rewards(scene):
    """Resolved tensors feed Warp masks and first-body reward kernels with the same selection."""
    from importlib import import_module

    from isaaclab_experimental.managers import SceneEntityCfg as WarpSceneEntityCfg

    from isaaclab.assets import ArticulationCfg

    rewards = import_module("isaaclab_tasks_experimental.core.reach.mdp.rewards")

    wp.init()
    scene["robot"].cfg = ArticulationCfg()
    cfg = WarpSceneEntityCfg("robot", joint_ids=[2, 0], body_ids=[-1, 1])
    cfg.resolve(scene)
    assert wp.to_torch(cfg.joint_mask).tolist() == [True, False, True, False]
    assert wp.to_torch(cfg.joint_ids_wp).tolist() == [2, 0]
    assert wp.to_torch(cfg.body_ids_wp).tolist() == [3, 1]
    root_pos = torch.zeros((1, 3))
    root_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    body_pos = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]])
    body_quat = root_quat.repeat(4, 1).reshape(1, 4, 4)
    command = torch.cat((root_pos, root_quat), dim=1)
    scene["robot"].data = SimpleNamespace(
        root_pos_w=SimpleNamespace(warp=wp.from_torch(root_pos, dtype=wp.vec3f)),
        root_quat_w=SimpleNamespace(warp=wp.from_torch(root_quat, dtype=wp.quatf)),
        body_pos_w=SimpleNamespace(warp=wp.from_torch(body_pos, dtype=wp.vec3f)),
        body_quat_w=SimpleNamespace(warp=wp.from_torch(body_quat, dtype=wp.quatf)),
    )
    env = SimpleNamespace(
        scene=scene, device="cpu", num_envs=1, command_manager=SimpleNamespace(get_command=lambda _: command)
    )
    out = wp.zeros(1, dtype=wp.float32, device="cpu")
    for name, kwargs, expected in (
        ("position_command_error", {}, 5.0),
        ("position_command_error_tanh", {"std": 5.0}, 1.0 - 0.7615941559557649),
        ("orientation_command_error", {}, 0.0),
    ):
        fn = getattr(rewards, name)
        try:
            fn(env, out, command_name="pose", asset_cfg=cfg, **kwargs)
            torch.testing.assert_close(wp.to_torch(out), torch.tensor([expected]))
        finally:
            for attr in ("_cmd_wp", "_cmd_name"):
                if hasattr(fn, attr):
                    delattr(fn, attr)
