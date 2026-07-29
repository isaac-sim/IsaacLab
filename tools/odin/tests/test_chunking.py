# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Chunking of planned rows into OSMO workflows."""

import pytest

from tools.odin.plan import PlannedRow, chunk_rows


def _row(key: str, timeout_s: int | None) -> PlannedRow:
    return PlannedRow(
        row_key=key,
        task_id=key,
        rl_library="rsl_rl",
        physics="physx",
        renderer=None,
        presets=(),
        play=False,
        video_length=200,
        seed=42,
        num_envs=None,
        max_iterations=None,
        timeout_s=timeout_s,
        uv_extras=("rsl-rl", "isaacsim"),
        profile="isaacsim",
        osmo_task_name=key.lower(),
    )


def test_empty_input_yields_no_chunks() -> None:
    assert chunk_rows([], 25) == []


def test_rows_are_split_at_chunk_size() -> None:
    rows = [_row(f"t{i}", 100) for i in range(5)]
    assert [len(chunk) for _, chunk in chunk_rows(rows, 2)] == [2, 2, 1]


def test_chunk_indices_ascend_from_zero() -> None:
    rows = [_row(f"t{i}", 100) for i in range(5)]
    assert [index for index, _ in chunk_rows(rows, 2)] == [0, 1, 2]


def test_rows_are_ordered_by_ascending_timeout() -> None:
    rows = [_row("slow", 9000), _row("fast", 60), _row("mid", 600)]
    flat = [row.row_key for _, chunk in chunk_rows(rows, 10) for row in chunk]
    assert flat == ["fast", "mid", "slow"]


def test_unbudgeted_rows_sort_last() -> None:
    # A row with no harvested budget must not share a chunk with a known-short
    # task, or the chunk's ceiling would be sized by the wrong row.
    rows = [_row("unknown", None), _row("fast", 60)]
    flat = [row.row_key for _, chunk in chunk_rows(rows, 10) for row in chunk]
    assert flat == ["fast", "unknown"]


def test_chunking_is_deterministic_under_input_reordering() -> None:
    first = [_row("x", 100), _row("y", 100), _row("z", 100)]
    second = [_row("z", 100), _row("x", 100), _row("y", 100)]
    assert [r.row_key for _, c in chunk_rows(first, 2) for r in c] == [
        r.row_key for _, c in chunk_rows(second, 2) for r in c
    ]


def test_chunk_size_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_rows([_row("t", 100)], 0)
