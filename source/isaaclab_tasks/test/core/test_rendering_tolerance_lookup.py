# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the per-environment pixel tolerance lookup."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_UTILS_PATH = Path(__file__).resolve().parent.parent / "rendering_test_utils.py"
_SOURCE = _UTILS_PATH.read_text()

# Call sites must resolve tolerances through the helper; a bare subscript yields the whole
# ``[sync, async]`` list instead of a float.
_RAW_SUBSCRIPT_RE = re.compile(r"MAX_DIFFERENT_PIXELS_PERCENTAGE_BY_ENV_NAME\[")

# The helper is the only sanctioned reader of the table.
_HELPER_UNPACK = "synchronous, asynchronous = MAX_DIFFERENT_PIXELS_PERCENTAGE_BY_ENV_NAME"


def test_no_call_site_subscripts_the_tolerance_table_directly():
    offenders = [
        f"{_UTILS_PATH.name}:{index}: {line.strip()}"
        for index, line in enumerate(_SOURCE.splitlines(), start=1)
        if _RAW_SUBSCRIPT_RE.search(line) and _HELPER_UNPACK not in line
    ]

    assert not offenders, "resolve tolerances via max_different_pixels_percentage_for():\n" + "\n".join(offenders)


def test_every_environment_declares_a_sync_and_async_tolerance():
    """Every entry must be a two-element ``[sync, async]`` list."""
    tree = ast.parse(_SOURCE)
    table = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "MAX_DIFFERENT_PIXELS_PERCENTAGE_BY_ENV_NAME"
            for target in node.targets
        )
    )

    assert isinstance(table, ast.Dict)
    for key, value in zip(table.keys, table.values):
        assert isinstance(value, ast.List), f"{ast.literal_eval(key)} must be [sync, async]"
        assert len(value.elts) == 2, f"{ast.literal_eval(key)} must declare exactly two tolerances"


@pytest.mark.parametrize("env_name", ["cartpole", "shadow_hand", "franka_cloth"])
def test_helper_returns_a_float_for_both_lanes(env_name, monkeypatch):
    import sys

    sys.path.insert(0, str(_UTILS_PATH.parent))
    from rendering_test_utils import max_different_pixels_percentage_for

    monkeypatch.delenv("ISAAC_LAB_ASYNC_RENDERING", raising=False)
    sync = max_different_pixels_percentage_for(env_name)
    monkeypatch.setenv("ISAAC_LAB_ASYNC_RENDERING", "1")
    async_ = max_different_pixels_percentage_for(env_name)

    assert isinstance(sync, float) and isinstance(async_, float)
    assert async_ >= sync, "the async tolerance must not be tighter than the synchronous one"
