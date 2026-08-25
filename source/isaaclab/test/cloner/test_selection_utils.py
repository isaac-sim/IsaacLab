# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for selection-aware scene entity resolution."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab import cloner
from isaaclab.cloner.selection_utils import SceneEntitySelectionCfg
from isaaclab.managers import SceneEntityCfg


class _View:
    """Physics-view stand-in exposing concrete prim paths in view order."""

    def __init__(self, prim_paths: list[str]):
        self.prim_paths = prim_paths


class _Asset:
    """Scene-entity stand-in exposing the selection resolution contract."""

    joint_names = ["joint_0", "joint_1"]
    num_joints = 2

    def __init__(self, prim_paths: list[str], num_bodies: int = 1, device: str = "cpu"):
        self.device = device
        self.root_view = _View(prim_paths)
        self.num_instances = len(prim_paths) // num_bodies

    def find_joints(self, joint_names: list[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Resolve exact joint names for inherited ``SceneEntityCfg`` coverage."""
        joint_ids = [self.joint_names.index(name) for name in joint_names]
        if not preserve_order:
            joint_ids.sort()
        return joint_ids, [self.joint_names[joint_id] for joint_id in joint_ids]


class _Scene(dict):
    """Interactive-scene stand-in used by ``SceneEntityCfg.resolve``."""

    def __init__(self, asset: _Asset, num_envs: int, env_template: str = "/World/envs/env_{}"):
        super().__init__(robot=asset)
        self.num_envs = num_envs
        self.cloner_cfg = SimpleNamespace(clone_template=env_template)


def _slot_paths(slot: int, env_ids: list[int], env_prefix: str = "/World/envs/env_") -> list[str]:
    """Return concrete clone paths for one asset slot."""
    return [f"{env_prefix}{env_id}/Groceries/Grocery_{slot:02d}" for env_id in env_ids]


def test_selection_cfg_extends_scene_entity_cfg() -> None:
    """The heterogeneous entity should follow the existing manager resolution contract."""
    cfg = SceneEntitySelectionCfg("robot")

    assert isinstance(cfg, SceneEntityCfg)
    assert cloner.selection_utils.SceneEntitySelectionCfg is SceneEntitySelectionCfg


def test_resolve_retains_scene_entity_member_resolution() -> None:
    """Joint-name resolution should run before physics-view selection resolution."""
    cfg = SceneEntitySelectionCfg("robot", joint_names="joint_1")

    cfg.resolve(_Scene(_Asset(_slot_paths(0, [0])), num_envs=1))

    assert cfg.joint_names == ["joint_1"]
    assert cfg.joint_ids == [1]


def test_resolve_maps_both_directions_in_view_order() -> None:
    """Resolution should preserve view order and mark absent environments with ``-1``."""
    asset = _Asset(_slot_paths(0, [3, 0, 2]))
    cfg = SceneEntitySelectionCfg("robot")

    cfg.resolve(_Scene(asset, num_envs=5))

    assert cfg.env_ids.tolist() == [3, 0, 2]
    assert cfg.instance_ids.tolist() == [1, -1, 2, 0, -1]


def test_resolve_reads_first_body_block() -> None:
    """A multi-body view should report its environment order once per instance."""
    env_ids = [0, 2, 3]
    asset = _Asset(_slot_paths(0, env_ids) + _slot_paths(8, env_ids), num_bodies=2)
    cfg = SceneEntitySelectionCfg("robot")

    cfg.resolve(_Scene(asset, num_envs=5))

    assert cfg.env_ids.tolist() == env_ids


def test_resolve_honors_scene_clone_template() -> None:
    """Resolution should use the active scene's clone template."""
    asset = _Asset(_slot_paths(0, [1, 4], env_prefix="/World/scenes/scene_"))
    cfg = SceneEntitySelectionCfg("robot")

    cfg.resolve(_Scene(asset, num_envs=5, env_template="/World/scenes/scene_{}"))

    assert cfg.env_ids.tolist() == [1, 4]


def test_resolve_rejects_non_environment_paths() -> None:
    """Every physics-view row should belong to a concrete replicated environment."""
    cfg = SceneEntitySelectionCfg("robot")

    with pytest.raises(ValueError, match="not under the environment template"):
        cfg.resolve(_Scene(_Asset(["/World/Table/Object"]), num_envs=1))


def test_select_maps_rows_and_reports_selected_env_ids() -> None:
    """Selection should return aligned rows and environment IDs after dropping absent environments."""
    cfg = SceneEntitySelectionCfg("robot")
    cfg.resolve(_Scene(_Asset(_slot_paths(0, [0, 2, 3])), num_envs=5))

    rows, selected_env_ids = cfg.select(cfg.instance_ids.new_tensor([0, 1, 3]))

    assert rows.tolist() == [0, 2]
    assert selected_env_ids.tolist() == [0, 3]
    with pytest.raises(ValueError, match=r"Environments \[1\] contain no 'robot'"):
        cfg.select(cfg.instance_ids.new_tensor([0, 1]), strict=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_select_accepts_cpu_env_ids_for_cuda_entity() -> None:
    """CPU environment IDs should be selected on the resolved entity's CUDA device."""
    cfg = SceneEntitySelectionCfg("robot")
    cfg.resolve(_Scene(_Asset(_slot_paths(0, [0, 2, 3]), device="cuda"), num_envs=5))

    rows, selected_env_ids = cfg.select(torch.tensor([0, 1, 3]))

    assert rows.device.type == "cuda"
    assert selected_env_ids.device.type == "cuda"
    assert rows.tolist() == [0, 2]
    assert selected_env_ids.tolist() == [0, 3]
    with pytest.raises(ValueError, match=r"Environments \[1\] contain no 'robot'"):
        cfg.select(torch.tensor([0, 1]), strict=True)


def test_scatter_to_envs_restores_global_order() -> None:
    """Physics-view values should be scattered to global order with the requested fill value."""
    cfg = SceneEntitySelectionCfg("robot")
    cfg.resolve(_Scene(_Asset(_slot_paths(0, [3, 0, 2])), num_envs=5))
    values = cfg.instance_ids.new_tensor([[30, 31], [0, 1], [20, 21]])

    result = cfg.scatter_to_envs(values, fill_value=-1)
    mask = cfg.scatter_to_envs(cfg.instance_ids.new_tensor([True, False, True]).bool())

    assert result.tolist() == [[0, 1], [-1, -1], [20, 21], [30, 31], [-1, -1]]
    assert mask.tolist() == [False, False, True, True, False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_scatter_to_envs_accepts_cpu_values_for_cuda_entity() -> None:
    """Scattering should preserve the values device when selection indices are on CUDA."""
    cfg = SceneEntitySelectionCfg("robot")
    cfg.resolve(_Scene(_Asset(_slot_paths(0, [3, 0, 2]), device="cuda"), num_envs=5))
    values = torch.tensor([[30, 31], [0, 1], [20, 21]])

    result = cfg.scatter_to_envs(values, fill_value=-1)

    assert result.device.type == "cpu"
    assert result.tolist() == [[0, 1], [-1, -1], [20, 21], [30, 31], [-1, -1]]


def test_scatter_to_envs_rejects_misaligned_values() -> None:
    """Scatter should reject values without exactly one row per physics-view instance."""
    cfg = SceneEntitySelectionCfg("robot")
    cfg.resolve(_Scene(_Asset(_slot_paths(0, [0, 2, 3])), num_envs=5))

    with pytest.raises(ValueError, match="instance dimension"):
        cfg.scatter_to_envs(cfg.instance_ids.new_tensor(1))
    with pytest.raises(ValueError, match=r"Expected 3 values for 'robot', got 2"):
        cfg.scatter_to_envs(cfg.instance_ids.new_tensor([1, 2]))
