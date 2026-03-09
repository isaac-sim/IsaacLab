# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Pure-logic tests for heterogeneous multi-task environment utilities.

These tests exercise the global-to-local env-id mapping algorithm (used by
both :class:`AssetBase._filter_env_ids` and
:class:`ManagerTermBase._filter_env_ids`), the selective-cloning mask function
(:func:`_apply_asset_env_masks`), and the environment partitioning helper.

No Isaac Sim / USD / PhysX dependency is required.
"""

import torch

# ---------------------------------------------------------------------------
# Reusable implementations of the algorithms under test.
#
# These mirror the production code exactly so that the logic is validated
# without pulling in the heavy Isaac Sim import chain.
# ---------------------------------------------------------------------------


def _build_lookup(assigned_envs: tuple[int, ...], device: str = "cpu") -> torch.Tensor:
    """Build a global→local env-id lookup table (mirrors AssetBase._get_env_id_lookup)."""
    assigned = torch.tensor(assigned_envs, device=device, dtype=torch.long)
    lookup = torch.full((int(assigned.max().item()) + 1,), -1, dtype=torch.long, device=device)
    lookup[assigned] = torch.arange(len(assigned), device=device)
    return lookup


def _filter_env_ids(
    assigned_envs: tuple[int, ...],
    env_ids: list[int] | torch.Tensor | None,
    device: str = "cpu",
) -> torch.Tensor:
    """Map global env ids to local indices (mirrors AssetBase/ManagerTermBase._filter_env_ids)."""
    if env_ids is None:
        return torch.arange(len(assigned_envs), dtype=torch.long, device=device)
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=device)
    lookup = _build_lookup(assigned_envs, device)
    max_id = lookup.shape[0] - 1
    clamped = env_ids.clamp(max=max_id)
    valid = (env_ids <= max_id) & (lookup[clamped] >= 0)
    return lookup[env_ids[valid]]


def _apply_asset_env_masks(
    src_paths: list[str],
    clone_masking: torch.Tensor,
    asset_env_masks: dict[str, list[int]],
    num_clones: int,
) -> None:
    """Zero-out clone_masking for excluded envs (mirrors cloner_utils._apply_asset_env_masks)."""
    for row_idx, src_path in enumerate(src_paths):
        proto_root = "/".join(src_path.split("/")[:-1])
        if proto_root in asset_env_masks:
            allowed = asset_env_masks[proto_root]
            env_filter = torch.zeros(num_clones, dtype=torch.bool, device=clone_masking.device)
            env_filter[torch.tensor(allowed, dtype=torch.long, device=clone_masking.device)] = True
            clone_masking[row_idx] &= env_filter


def _partition_env_ids(num_envs: int, num_groups: int) -> list[list[int]]:
    """Split env indices evenly across groups (mirrors demo config helper)."""
    base, remainder = divmod(num_envs, num_groups)
    groups: list[list[int]] = []
    start = 0
    for g in range(num_groups):
        size = base + (1 if g < remainder else 0)
        groups.append(list(range(start, start + size)))
        start += size
    return groups


# ===========================================================================
# Tests: _filter_env_ids  (global → local mapping)
# ===========================================================================


class TestFilterEnvIds:
    """Tests for the global-to-local env-id mapping algorithm."""

    def test_none_returns_all_local(self):
        assigned = (2, 5, 8)
        result = _filter_env_ids(assigned, None)
        assert result.tolist() == [0, 1, 2]

    def test_exact_match(self):
        assigned = (4, 7, 10)
        result = _filter_env_ids(assigned, [4, 7, 10])
        assert result.tolist() == [0, 1, 2]

    def test_partial_match(self):
        assigned = (4, 7, 10)
        result = _filter_env_ids(assigned, [4, 10])
        assert result.tolist() == [0, 2]

    def test_no_match(self):
        assigned = (4, 7, 10)
        result = _filter_env_ids(assigned, [0, 1, 2, 3])
        assert result.numel() == 0

    def test_mixed_valid_and_invalid(self):
        assigned = (2, 5, 8)
        result = _filter_env_ids(assigned, [0, 2, 3, 5, 6, 8, 9, 100])
        assert result.tolist() == [0, 1, 2]

    def test_ids_beyond_lookup_range(self):
        assigned = (0, 1, 2)
        result = _filter_env_ids(assigned, [0, 1, 2, 50, 100])
        assert result.tolist() == [0, 1, 2]

    def test_tensor_input(self):
        assigned = (3, 6, 9)
        env_ids = torch.tensor([3, 6, 9], dtype=torch.long)
        result = _filter_env_ids(assigned, env_ids)
        assert result.tolist() == [0, 1, 2]

    def test_preserves_order(self):
        assigned = (10, 20, 30)
        result = _filter_env_ids(assigned, [30, 10])
        assert result.tolist() == [2, 0]

    def test_single_env(self):
        assigned = (5,)
        result = _filter_env_ids(assigned, [5])
        assert result.tolist() == [0]

    def test_single_env_miss(self):
        assigned = (5,)
        result = _filter_env_ids(assigned, [4])
        assert result.numel() == 0

    def test_contiguous_block(self):
        assigned = (8, 9, 10, 11, 12, 13, 14, 15)
        result = _filter_env_ids(assigned, [8, 10, 12, 14])
        assert result.tolist() == [0, 2, 4, 6]

    def test_duplicate_input_ids(self):
        assigned = (2, 5, 8)
        result = _filter_env_ids(assigned, [5, 5, 8, 8])
        assert result.tolist() == [1, 1, 2, 2]


# ===========================================================================
# Tests: _build_lookup
# ===========================================================================


class TestBuildLookup:
    """Tests for the global→local lookup table construction."""

    def test_contiguous(self):
        lookup = _build_lookup((0, 1, 2, 3))
        assert lookup.tolist() == [0, 1, 2, 3]

    def test_sparse(self):
        lookup = _build_lookup((2, 5))
        expected = [-1, -1, 0, -1, -1, 1]
        assert lookup.tolist() == expected

    def test_non_sorted_input(self):
        lookup = _build_lookup((10, 3, 7))
        assert lookup[3].item() == 1
        assert lookup[7].item() == 2
        assert lookup[10].item() == 0
        assert lookup[0].item() == -1

    def test_single_element(self):
        lookup = _build_lookup((0,))
        assert lookup.tolist() == [0]


# ===========================================================================
# Tests: _apply_asset_env_masks
# ===========================================================================


class TestApplyAssetEnvMasks:
    """Tests for selective cloning mask application."""

    def test_no_mask_entries_unchanged(self):
        masking = torch.ones(3, 6, dtype=torch.bool)
        _apply_asset_env_masks(
            src_paths=["/World/env_0/Robot/body", "/World/env_0/Table/mesh", "/World/env_0/Cube/geom"],
            clone_masking=masking,
            asset_env_masks={},
            num_clones=6,
        )
        assert masking.all()

    def test_single_asset_filtered(self):
        masking = torch.ones(2, 8, dtype=torch.bool)
        src_paths = ["/World/env_0/Robot/body", "/World/env_0/Cube/geom"]
        _apply_asset_env_masks(
            src_paths=src_paths,
            clone_masking=masking,
            asset_env_masks={"/World/env_0/Cube": [0, 1, 2, 3]},
            num_clones=8,
        )
        # Robot row unchanged
        assert masking[0].all()
        # Cube row: only envs 0-3
        assert masking[1, :4].all()
        assert not masking[1, 4:].any()

    def test_multiple_assets_filtered(self):
        masking = torch.ones(3, 12, dtype=torch.bool)
        src_paths = [
            "/World/env_0/RobotA/body",
            "/World/env_0/RobotB/body",
            "/World/env_0/SharedTable/mesh",
        ]
        _apply_asset_env_masks(
            src_paths=src_paths,
            clone_masking=masking,
            asset_env_masks={
                "/World/env_0/RobotA": [0, 1, 2, 3],
                "/World/env_0/RobotB": [4, 5, 6, 7],
            },
            num_clones=12,
        )
        # RobotA: only envs 0-3
        assert masking[0, :4].all()
        assert not masking[0, 4:].any()
        # RobotB: only envs 4-7
        assert not masking[1, :4].any()
        assert masking[1, 4:8].all()
        assert not masking[1, 8:].any()
        # SharedTable: unchanged
        assert masking[2].all()

    def test_respects_existing_masking(self):
        masking = torch.zeros(1, 8, dtype=torch.bool)
        masking[0, [0, 2, 4, 6]] = True
        _apply_asset_env_masks(
            src_paths=["/World/env_0/Cube/geom"],
            clone_masking=masking,
            asset_env_masks={"/World/env_0/Cube": [0, 1, 2, 3]},
            num_clones=8,
        )
        # Intersection of existing mask and allowed: only 0, 2
        assert masking[0].tolist() == [True, False, True, False, False, False, False, False]


# ===========================================================================
# Tests: _partition_env_ids
# ===========================================================================


class TestPartitionEnvIds:
    """Tests for the environment partitioning helper."""

    def test_even_split(self):
        groups = _partition_env_ids(12, 3)
        assert groups == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

    def test_uneven_split(self):
        groups = _partition_env_ids(10, 3)
        assert groups == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def test_single_group(self):
        groups = _partition_env_ids(5, 1)
        assert groups == [[0, 1, 2, 3, 4]]

    def test_more_groups_than_envs(self):
        groups = _partition_env_ids(2, 5)
        assert groups == [[0], [1], [], [], []]

    def test_one_env_per_group(self):
        groups = _partition_env_ids(3, 3)
        assert groups == [[0], [1], [2]]

    def test_all_ids_covered(self):
        for num_envs in (7, 13, 24, 100):
            for num_groups in (1, 2, 3, 5, 8):
                groups = _partition_env_ids(num_envs, num_groups)
                flat = [i for g in groups for i in g]
                assert sorted(flat) == list(range(num_envs))
                assert len(groups) == num_groups
