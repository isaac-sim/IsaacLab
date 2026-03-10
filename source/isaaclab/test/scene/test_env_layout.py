# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`EnvLayout` and :func:`partition_env_ids`.

No Isaac Sim / USD / PhysX dependency required.
"""

import pytest
import torch

from isaaclab.scene.env_layout import EnvLayout, partition_env_ids

# ===========================================================================
# EnvLayout: registration & basic queries
# ===========================================================================


class TestEnvLayoutRegistration:
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
        assert layout.env_ids("missing") == (0, 1, 2, 3, 4, 5)

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

    def test_shared_cache_for_same_ids(self):
        layout = EnvLayout(24, "cpu")
        layout.register("a", [0, 1, 2])
        layout.register("b", [0, 1, 2])
        sa = layout.env_slice("a")
        sb = layout.env_slice("b")
        if isinstance(sa, torch.Tensor):
            assert sa.data_ptr() == sb.data_ptr()
        else:
            assert sa == sb


# ===========================================================================
# EnvLayout: mask
# ===========================================================================


class TestMask:
    def test_unregistered_returns_all_true(self):
        layout = EnvLayout(4, "cpu")
        m = layout.mask("missing")
        assert m.all()
        assert m.shape == (4,)

    def test_partial_mask(self):
        layout = EnvLayout(8, "cpu")
        layout.register("lift", [2, 5, 7])
        m = layout.mask("lift")
        expected = [False, False, True, False, False, True, False, True]
        assert m.tolist() == expected


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


# ===========================================================================
# EnvLayout: scatter / gather
# ===========================================================================


class TestScatterGather:
    def test_scatter_homogeneous_passthrough(self):
        layout = EnvLayout(4, "cpu")
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = layout.scatter("any", data)
        assert torch.equal(result, data)

    def test_scatter_partial(self):
        layout = EnvLayout(6, "cpu")
        layout.register("g", [1, 3, 5])
        data = torch.tensor([10.0, 30.0, 50.0])
        result = layout.scatter("g", data, fill=0.0)
        assert result.tolist() == [0.0, 10.0, 0.0, 30.0, 0.0, 50.0]

    def test_scatter_2d(self):
        layout = EnvLayout(4, "cpu")
        layout.register("g", [0, 2])
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = layout.scatter("g", data, fill=-1.0)
        assert result.shape == (4, 2)
        assert result[0].tolist() == [1.0, 2.0]
        assert result[1].tolist() == [-1.0, -1.0]
        assert result[2].tolist() == [3.0, 4.0]
        assert result[3].tolist() == [-1.0, -1.0]

    def test_gather_homogeneous_passthrough(self):
        layout = EnvLayout(4, "cpu")
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = layout.gather("any", data)
        assert torch.equal(result, data)

    def test_gather_partial(self):
        layout = EnvLayout(6, "cpu")
        layout.register("g", [1, 3, 5])
        data = torch.tensor([0.0, 10.0, 0.0, 30.0, 0.0, 50.0])
        result = layout.gather("g", data)
        assert result.tolist() == [10.0, 30.0, 50.0]

    def test_scatter_gather_roundtrip(self):
        layout = EnvLayout(8, "cpu")
        layout.register("g", [2, 4, 6])
        local_data = torch.tensor([100.0, 200.0, 300.0])
        full = layout.scatter("g", local_data)
        recovered = layout.gather("g", full)
        assert torch.equal(recovered, local_data)


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
# EnvLayout: per_group_mean
# ===========================================================================


class TestPerGroupMean:
    def test_per_group_mean(self):
        layout = EnvLayout(6, "cpu")
        layout.register("a", [0, 1, 2])
        layout.register("b", [3, 4, 5])
        values = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        result = layout.per_group_mean(values)
        assert abs(result["a"] - 2.0) < 1e-6
        assert abs(result["b"] - 20.0) < 1e-6


# ===========================================================================
# EnvLayout: validation
# ===========================================================================


class TestValidation:
    def test_no_overlap_passes(self):
        layout = EnvLayout(12, "cpu")
        layout.register("a", [0, 1, 2, 3])
        layout.register("b", [4, 5, 6, 7])
        layout.validate_no_overlap(["a", "b"])

    def test_overlap_raises(self):
        layout = EnvLayout(12, "cpu")
        layout.register("a", [0, 1, 2, 3])
        layout.register("b", [3, 4, 5, 6])
        with pytest.raises(ValueError, match="Overlapping"):
            layout.validate_no_overlap(["a", "b"])

    def test_full_coverage_passes(self):
        layout = EnvLayout(6, "cpu")
        layout.register("a", [0, 1, 2])
        layout.register("b", [3, 4, 5])
        layout.validate_full_coverage(["a", "b"])

    def test_missing_coverage_raises(self):
        layout = EnvLayout(6, "cpu")
        layout.register("a", [0, 1, 2])
        with pytest.raises(ValueError, match="not covered"):
            layout.validate_full_coverage(["a"])


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
# partition_env_ids
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

    def test_full_coverage(self):
        for n in (7, 13, 24, 100):
            for g in (1, 2, 3, 5):
                result = partition_env_ids(n, g)
                all_ids = []
                for ids in result.values():
                    all_ids.extend(ids)
                assert sorted(all_ids) == list(range(n))


class TestApplyTaskGroups:
    """Tests for EnvLayout.apply_task_groups and resolve_task_group."""

    def test_apply_registers_all_groups(self):
        layout = EnvLayout(24, "cpu")
        layout.apply_task_groups({"lift": 1, "stack": 1, "reach": 1})
        assert layout.is_heterogeneous
        assert "lift" in layout.group_names
        assert "stack" in layout.group_names
        assert "reach" in layout.group_names

    def test_apply_stores_partition(self):
        layout = EnvLayout(24, "cpu")
        layout.apply_task_groups({"a": 1, "b": 1, "c": 1})
        p = layout.task_group_partition
        assert p is not None
        all_ids = []
        for ids in p.values():
            all_ids.extend(ids)
        assert sorted(all_ids) == list(range(24))

    def test_apply_twice_raises(self):
        layout = EnvLayout(12, "cpu")
        layout.apply_task_groups(2)
        with pytest.raises(RuntimeError):
            layout.apply_task_groups(2)

    def test_partition_none_before_apply(self):
        layout = EnvLayout(12, "cpu")
        assert layout.task_group_partition is None

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
        p = layout.task_group_partition
        assert p is not None
        assert len(p["small"]) == 3
        assert len(p["large"]) == 9
