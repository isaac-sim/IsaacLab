# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`RobotInfo`, :class:`EnvLayout`, and :func:`partition_env_ids`.

No Isaac Sim / USD / PhysX dependency required.
"""

import pytest
import torch

from isaaclab.scene.env_layout import EnvLayout, RobotInfo, partition_env_ids

# ===========================================================================
# RobotInfo
# ===========================================================================


class TestRobotInfo:
    def test_init(self):
        r = RobotInfo("robot_a")
        assert r.asset_name == "robot_a"
        assert r.ee_body is None
        assert r.command_name is None
        assert r.joint_patterns == []
        assert r.meta == {}

    def test_update_merges_metadata(self):
        r = RobotInfo("robot_a")
        r.update(ee_body="ee_link", joint_patterns=["joint_.*"])
        assert r.ee_body == "ee_link"
        assert r.joint_patterns == ["joint_.*"]

    def test_update_ignores_none_values(self):
        r = RobotInfo("robot_a")
        r.update(ee_body="ee_link")
        r.update(ee_body=None, command_name="reach")
        assert r.ee_body == "ee_link"
        assert r.command_name == "reach"

    def test_update_overwrites_existing(self):
        r = RobotInfo("robot_a")
        r.update(ee_body="old")
        r.update(ee_body="new")
        assert r.ee_body == "new"

    def test_meta_exposes_all_keys(self):
        r = RobotInfo("robot_a")
        r.update(ee_body="link", custom_key=42)
        assert r.meta == {"ee_body": "link", "custom_key": 42}

    def test_update_invalidates_resolved_cfg(self):
        r = RobotInfo("robot_a")
        r._resolved_cfg = "cached"
        r.update(ee_body="link")
        assert r._resolved_cfg is None

    def test_repr(self):
        r = RobotInfo("robot_a")
        r.update(ee_body="link")
        s = repr(r)
        assert "robot_a" in s
        assert "ee_body" in s


# ===========================================================================
# EnvLayout: registration & basic queries
# ===========================================================================


class TestRegistration:
    def test_empty_layout_is_homogeneous(self):
        layout = EnvLayout(24, "cpu")
        assert not layout.is_heterogeneous
        assert layout.num_envs == 24
        assert layout.group_names == []

    def test_register_creates_group(self):
        layout = EnvLayout(24, "cpu")
        layout.register("lift", [0, 1, 2, 3])
        assert layout.is_heterogeneous
        assert "lift" in layout.group_names
        assert layout.num_envs_for("lift") == 4
        assert layout.env_ids("lift") == (0, 1, 2, 3)

    def test_unregistered_key_returns_all(self):
        layout = EnvLayout(6, "cpu")
        assert layout.num_envs_for("missing") == 6
        assert layout.env_ids("missing") == tuple(range(6))

    def test_none_key_returns_all(self):
        layout = EnvLayout(6, "cpu")
        assert layout.num_envs_for(None) == 6
        assert layout.env_ids(None) == tuple(range(6))

    def test_register_out_of_range_raises(self):
        layout = EnvLayout(10, "cpu")
        with pytest.raises(ValueError, match="out of range"):
            layout.register("bad", [0, 1, 100])

    def test_register_negative_raises(self):
        layout = EnvLayout(10, "cpu")
        with pytest.raises(ValueError, match="out of range"):
            layout.register("bad", [-1, 0, 1])

    def test_register_duplicates_raises(self):
        layout = EnvLayout(10, "cpu")
        with pytest.raises(ValueError, match="duplicates"):
            layout.register("bad", [0, 1, 1, 2])

    def test_re_register_overwrites(self):
        layout = EnvLayout(10, "cpu")
        layout.register("g", [0, 1])
        layout.register("g", [3, 4, 5])
        assert layout.env_ids("g") == (3, 4, 5)
        assert layout.num_envs_for("g") == 3


# ===========================================================================
# EnvLayout: env_ids_t & asset_env_ids_t
# ===========================================================================


class TestEnvIdsTensor:
    def test_env_ids_t_returns_long_tensor(self):
        layout = EnvLayout(8, "cpu")
        layout.register("g", [2, 5])
        t = layout.env_ids_t("g")
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.long
        assert t.tolist() == [2, 5]

    def test_env_ids_t_cached(self):
        layout = EnvLayout(8, "cpu")
        layout.register("g", [2, 5])
        t1 = layout.env_ids_t("g")
        t2 = layout.env_ids_t("g")
        assert t1.data_ptr() == t2.data_ptr()

    def test_env_ids_t_none_returns_all(self):
        layout = EnvLayout(4, "cpu")
        t = layout.env_ids_t(None)
        assert t.tolist() == [0, 1, 2, 3]

    def test_asset_env_ids_t_registered(self):
        layout = EnvLayout(12, "cpu")
        layout.register("lift", [0, 1, 2])
        layout.register_asset("robot_a", "lift")
        t = layout.asset_env_ids_t("robot_a")
        assert t is not None
        assert t.tolist() == [0, 1, 2]

    def test_asset_env_ids_t_unregistered(self):
        layout = EnvLayout(12, "cpu")
        assert layout.asset_env_ids_t("unknown") is None


# ===========================================================================
# EnvLayout: env_slice
# ===========================================================================


class TestEnvSlice:
    def test_unregistered_returns_slice_none(self):
        layout = EnvLayout(8, "cpu")
        assert layout.env_slice("missing") == slice(None)

    def test_contiguous_returns_slice(self):
        layout = EnvLayout(24, "cpu")
        layout.register("stack", [8, 9, 10, 11, 12, 13, 14, 15])
        s = layout.env_slice("stack")
        assert isinstance(s, slice)
        assert s == slice(8, 16)

    def test_sparse_returns_tensor(self):
        layout = EnvLayout(24, "cpu")
        layout.register("sparse", [0, 2, 5, 10])
        s = layout.env_slice("sparse")
        assert isinstance(s, torch.Tensor)
        assert s.tolist() == [0, 2, 5, 10]

    def test_single_element(self):
        layout = EnvLayout(8, "cpu")
        layout.register("one", [3])
        s = layout.env_slice("one")
        assert isinstance(s, slice)
        assert s == slice(3, 4)


# ===========================================================================
# EnvLayout: global_to_local
# ===========================================================================


class TestGlobalToLocal:
    def test_homogeneous_passthrough(self):
        layout = EnvLayout(8, "cpu")
        ids = torch.tensor([2, 5], dtype=torch.long)
        result = layout.global_to_local("any", ids)
        assert result.tolist() == [2, 5]

    def test_exact_match(self):
        layout = EnvLayout(24, "cpu")
        layout.register("lift", [4, 7, 10])
        result = layout.global_to_local("lift", torch.tensor([4, 7, 10]))
        assert result.tolist() == [0, 1, 2]

    def test_drops_non_matching(self):
        layout = EnvLayout(24, "cpu")
        layout.register("lift", [4, 7, 10])
        result = layout.global_to_local("lift", torch.tensor([0, 4, 5, 7, 100]))
        assert result.tolist() == [0, 1]

    def test_empty_when_no_match(self):
        layout = EnvLayout(24, "cpu")
        layout.register("lift", [4, 7, 10])
        result = layout.global_to_local("lift", torch.tensor([0, 1, 2, 3]))
        assert result.numel() == 0

    def test_preserves_order(self):
        layout = EnvLayout(24, "cpu")
        layout.register("g", [10, 20])
        result = layout.global_to_local("g", torch.tensor([20, 10]))
        assert result.tolist() == [1, 0]


# ===========================================================================
# EnvLayout: local_to_global
# ===========================================================================


class TestLocalToGlobal:
    def test_homogeneous_passthrough(self):
        layout = EnvLayout(8, "cpu")
        ids = torch.tensor([2, 5], dtype=torch.long)
        result = layout.local_to_global("any", ids)
        assert result.tolist() == [2, 5]

    def test_maps_back_correctly(self):
        layout = EnvLayout(24, "cpu")
        layout.register("g", [10, 15, 20])
        result = layout.local_to_global("g", torch.tensor([0, 1, 2]))
        assert result.tolist() == [10, 15, 20]

    def test_roundtrip_with_global_to_local(self):
        layout = EnvLayout(24, "cpu")
        layout.register("g", [4, 8, 12])
        global_ids = torch.tensor([4, 12])
        local = layout.global_to_local("g", global_ids)
        recovered = layout.local_to_global("g", local)
        assert recovered.tolist() == [4, 12]


# ===========================================================================
# EnvLayout: filter_and_split
# ===========================================================================


class TestFilterAndSplit:
    def test_homogeneous(self):
        layout = EnvLayout(8, "cpu")
        ids = torch.tensor([2, 5])
        local, glob = layout.filter_and_split("any", ids)
        assert local.tolist() == [2, 5]
        assert glob.tolist() == [2, 5]

    def test_heterogeneous(self):
        layout = EnvLayout(24, "cpu")
        layout.register("reach", [16, 17, 18, 19, 20, 21, 22, 23])
        ids = torch.tensor([2, 5, 18, 20])
        local, glob = layout.filter_and_split("reach", ids)
        assert local.tolist() == [2, 4]
        assert glob.tolist() == [18, 20]

    def test_no_match_returns_empty(self):
        layout = EnvLayout(24, "cpu")
        layout.register("g", [10, 11, 12])
        local, glob = layout.filter_and_split("g", torch.tensor([0, 1, 2]))
        assert local.numel() == 0
        assert glob.numel() == 0


# ===========================================================================
# EnvLayout: scatter
# ===========================================================================


class TestScatter:
    def test_homogeneous_passthrough(self):
        layout = EnvLayout(4, "cpu")
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = layout.scatter("any", data)
        assert torch.equal(result, data)

    def test_partial_1d(self):
        layout = EnvLayout(6, "cpu")
        layout.register("g", [1, 3, 5])
        data = torch.tensor([10.0, 30.0, 50.0])
        result = layout.scatter("g", data, fill=0.0)
        assert result.tolist() == [0.0, 10.0, 0.0, 30.0, 0.0, 50.0]

    def test_partial_2d(self):
        layout = EnvLayout(4, "cpu")
        layout.register("g", [0, 2])
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = layout.scatter("g", data, fill=-1.0)
        assert result.shape == (4, 2)
        assert result[0].tolist() == [1.0, 2.0]
        assert result[1].tolist() == [-1.0, -1.0]
        assert result[2].tolist() == [3.0, 4.0]
        assert result[3].tolist() == [-1.0, -1.0]


# ===========================================================================
# EnvLayout: cross_slice
# ===========================================================================


class TestCrossSlice:
    def test_both_homogeneous(self):
        layout = EnvLayout(8, "cpu")
        assert layout.cross_slice("term", "asset") == slice(None)

    def test_term_partial_asset_full(self):
        layout = EnvLayout(24, "cpu")
        layout.register("cmd", [16, 17, 18, 19, 20, 21, 22, 23])
        s = layout.cross_slice("cmd", "robot")
        assert isinstance(s, slice)
        assert s == slice(16, 24)

    def test_both_partial_same_ids(self):
        layout = EnvLayout(24, "cpu")
        layout.register("term", [0, 1, 2])
        layout.register("asset", [0, 1, 2])
        assert layout.cross_slice("term", "asset") == slice(None)


# ===========================================================================
# EnvLayout: entity registry
# ===========================================================================


class TestEntityRegistry:
    def test_register_asset_and_lookup(self):
        layout = EnvLayout(12, "cpu")
        layout.register("lift", [0, 1, 2])
        layout.register_asset("robot_a", "lift")
        assert layout.group_for_asset("robot_a") == "lift"
        assert layout.group_for_asset("unknown") is None

    def test_register_term_and_lookup(self):
        layout = EnvLayout(12, "cpu")
        layout.register("lift", [0, 1, 2])
        layout.register_term("joint_pos", "lift")
        assert layout.group_for_term("joint_pos") == "lift"
        assert layout.group_for_term("unknown") is None

    def test_resolve_group_key_task_group_priority(self):
        layout = EnvLayout(12, "cpu")
        layout.register("lift", [0, 1, 2])
        layout.register("stack", [3, 4, 5])
        layout.register_asset("robot_a", "stack")
        key = layout.resolve_group_key(task_group="lift", asset_name="robot_a")
        assert key == "lift"

    def test_resolve_group_key_falls_back_to_asset(self):
        layout = EnvLayout(12, "cpu")
        layout.register("stack", [3, 4, 5])
        layout.register_asset("robot_a", "stack")
        key = layout.resolve_group_key(asset_name="robot_a")
        assert key == "stack"

    def test_resolve_group_key_none_when_homogeneous(self):
        layout = EnvLayout(12, "cpu")
        assert layout.resolve_group_key() is None
        assert layout.resolve_group_key(task_group="nonexistent") is None


# ===========================================================================
# EnvLayout: dispatch helpers
# ===========================================================================


class TestDispatchHelpers:
    def test_resolve_env_ids_homogeneous_passthrough(self):
        layout = EnvLayout(8, "cpu")
        result = layout.resolve_env_ids(None, [0, 1, 2])
        assert result == [0, 1, 2]

    def test_resolve_env_ids_none_passthrough(self):
        layout = EnvLayout(8, "cpu")
        layout.register("g", [0, 1, 2])
        assert layout.resolve_env_ids("g", None) is None

    def test_resolve_env_ids_slice_passthrough(self):
        layout = EnvLayout(8, "cpu")
        layout.register("g", [0, 1, 2])
        result = layout.resolve_env_ids("g", slice(None))
        assert result == slice(None)

    def test_resolve_env_ids_maps_to_local(self):
        layout = EnvLayout(12, "cpu")
        layout.register("g", [4, 7, 10])
        result = layout.resolve_env_ids("g", [4, 7])
        assert isinstance(result, torch.Tensor)
        assert result.tolist() == [0, 1]

    def test_resolve_env_ids_returns_none_on_no_match(self):
        layout = EnvLayout(12, "cpu")
        layout.register("g", [4, 7, 10])
        result = layout.resolve_env_ids("g", [0, 1, 2])
        assert result is None

    def test_resolve_term_env_ids(self):
        layout = EnvLayout(12, "cpu")
        layout.register("g", [4, 7, 10])
        layout.register_term("my_term", "g")
        result = layout.resolve_term_env_ids("my_term", [4, 10])
        assert isinstance(result, torch.Tensor)
        assert result.tolist() == [0, 2]

    def test_resolve_asset_env_ids(self):
        layout = EnvLayout(12, "cpu")
        layout.register("g", [4, 7, 10])
        layout.register_asset("robot_a", "g")
        result = layout.resolve_asset_env_ids("robot_a", [7, 10])
        assert isinstance(result, torch.Tensor)
        assert result.tolist() == [1, 2]

    def test_term_env_slice_registered(self):
        layout = EnvLayout(12, "cpu")
        layout.register("g", [4, 5, 6])
        layout.register_term("my_term", "g")
        s = layout.term_env_slice("my_term")
        assert isinstance(s, slice)
        assert s == slice(4, 7)

    def test_term_env_slice_unregistered(self):
        layout = EnvLayout(12, "cpu")
        assert layout.term_env_slice("unknown") == slice(None)


# ===========================================================================
# EnvLayout: robot metadata
# ===========================================================================


class TestRobotMetadata:
    def test_register_robot_meta_creates_info(self):
        layout = EnvLayout(12, "cpu")
        layout.register_robot_meta("robot_a", ee_body="link", joint_patterns=["j.*"])
        infos = layout.robot_infos
        assert len(infos) == 1
        assert infos[0].asset_name == "robot_a"
        assert infos[0].ee_body == "link"

    def test_register_robot_meta_merges(self):
        layout = EnvLayout(12, "cpu")
        layout.register_robot_meta("robot_a", ee_body="link")
        layout.register_robot_meta("robot_a", command_name="reach")
        infos = layout.robot_infos
        assert len(infos) == 1
        assert infos[0].ee_body == "link"
        assert infos[0].command_name == "reach"

    def test_multiple_robots_ordered(self):
        layout = EnvLayout(12, "cpu")
        layout.register_robot_meta("robot_b", ee_body="b_link")
        layout.register_robot_meta("robot_a", ee_body="a_link")
        infos = layout.robot_infos
        assert len(infos) == 2
        assert infos[0].asset_name == "robot_b"
        assert infos[1].asset_name == "robot_a"

    def test_robot_infos_empty_by_default(self):
        layout = EnvLayout(12, "cpu")
        assert layout.robot_infos == []


# ===========================================================================
# EnvLayout: multi_task_onehot
# ===========================================================================


class TestMultiTaskOnehot:
    def test_onehot_shape_and_values(self):
        layout = EnvLayout(6, "cpu")
        layout.apply_task_groups({"a": 1, "b": 1})
        oh = layout.multi_task_onehot()
        assert oh.shape == (6, 2)
        assert oh[:3, 0].sum().item() == 3.0
        assert oh[:3, 1].sum().item() == 0.0
        assert oh[3:, 0].sum().item() == 0.0
        assert oh[3:, 1].sum().item() == 3.0

    def test_each_row_sums_to_one(self):
        layout = EnvLayout(12, "cpu")
        layout.apply_task_groups({"a": 1, "b": 1, "c": 1})
        oh = layout.multi_task_onehot()
        assert torch.allclose(oh.sum(dim=1), torch.ones(12))


# ===========================================================================
# EnvLayout: apply_task_groups & resolve_task_group
# ===========================================================================


class TestApplyTaskGroups:
    def test_apply_registers_all_groups(self):
        layout = EnvLayout(24, "cpu")
        layout.apply_task_groups({"lift": 1, "stack": 1, "reach": 1})
        assert layout.is_heterogeneous
        assert set(layout.group_names) == {"lift", "stack", "reach"}

    def test_apply_with_int(self):
        layout = EnvLayout(12, "cpu")
        layout.apply_task_groups(3)
        assert len(layout.group_names) == 3
        total = sum(layout.num_envs_for(g) for g in layout.group_names)
        assert total == 12

    def test_apply_twice_raises(self):
        layout = EnvLayout(12, "cpu")
        layout.apply_task_groups(2)
        with pytest.raises(RuntimeError):
            layout.apply_task_groups(2)

    def test_resolve_task_group_success(self):
        layout = EnvLayout(12, "cpu")
        layout.apply_task_groups({"x": 1, "y": 1})
        ids = layout.resolve_task_group("my_asset", "x")
        assert len(ids) == 6
        assert ids == list(range(6))

    def test_resolve_unknown_group_raises(self):
        layout = EnvLayout(12, "cpu")
        layout.apply_task_groups({"x": 1, "y": 1})
        with pytest.raises(ValueError):
            layout.resolve_task_group("my_asset", "z")

    def test_resolve_without_apply_raises(self):
        layout = EnvLayout(12, "cpu")
        with pytest.raises(ValueError):
            layout.resolve_task_group("my_asset", "x")

    def test_weighted_partition(self):
        layout = EnvLayout(12, "cpu")
        layout.apply_task_groups({"small": 1, "large": 3})
        assert layout.num_envs_for("small") == 3
        assert layout.num_envs_for("large") == 9


# ===========================================================================
# EnvLayout: repr
# ===========================================================================


class TestRepr:
    def test_homogeneous_repr(self):
        layout = EnvLayout(8, "cpu")
        r = repr(layout)
        assert "homogeneous" in r
        assert "8" in r

    def test_heterogeneous_repr(self):
        layout = EnvLayout(24, "cpu")
        layout.register("lift", [0, 1, 2])
        r = repr(layout)
        assert "lift" in r
        assert "3 envs" in r


# ===========================================================================
# partition_env_ids (standalone function)
# ===========================================================================


class TestPartitionEnvIds:
    def test_equal_split_with_dict(self):
        result = partition_env_ids(12, {"a": 4, "b": 4, "c": 4})
        assert result == {"a": [0, 1, 2, 3], "b": [4, 5, 6, 7], "c": [8, 9, 10, 11]}

    def test_weighted_split(self):
        result = partition_env_ids(12, {"big": 2, "small": 1})
        assert len(result["big"]) == 8
        assert len(result["small"]) == 4
        all_ids = result["big"] + result["small"]
        assert sorted(all_ids) == list(range(12))

    def test_int_shorthand(self):
        result = partition_env_ids(9, 3)
        assert len(result) == 3
        all_ids = []
        for ids in result.values():
            all_ids.extend(ids)
        assert sorted(all_ids) == list(range(9))

    def test_full_coverage_parametric(self):
        for n in (7, 13, 24, 100):
            for g in (1, 2, 3, 5):
                result = partition_env_ids(n, g)
                all_ids = []
                for ids in result.values():
                    all_ids.extend(ids)
                assert sorted(all_ids) == list(range(n)), f"failed for n={n}, g={g}"

    def test_single_group(self):
        result = partition_env_ids(5, 1)
        assert len(result) == 1
        assert list(result.values())[0] == [0, 1, 2, 3, 4]

    def test_indivisible_last_group_absorbs(self):
        result = partition_env_ids(10, {"A": 1, "B": 1, "C": 1})
        total = sum(len(v) for v in result.values())
        assert total == 10
        assert len(result["C"]) >= len(result["A"])
