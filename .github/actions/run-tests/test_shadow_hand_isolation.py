# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for isolated Shadow Hand rendering CI processes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHADOW_TEST_DIR = _REPO_ROOT / "source/isaaclab_tasks/test/core"
_SHADOW_TEST_FILES = {
    "test_rendering_shadow_hand_newton_isaacsim_rtx.py",
    "test_rendering_shadow_hand_physx_isaacsim_rtx.py",
    "test_rendering_shadow_hand_physx_newton_warp.py",
}

sys.path.insert(0, str(_REPO_ROOT / "source/isaaclab"))
sys.path.insert(0, str(_REPO_ROOT / "tools"))


def _load_module(name: str, path: Path):
    """Load a repository helper that is intentionally not a Python package."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rendering_utils = _load_module(
    "rendering_test_utils_for_shadow_isolation",
    _REPO_ROOT / "source/isaaclab_tasks/test/rendering_test_utils.py",
)
runner = _load_module("isaaclab_tools_conftest_for_shadow_isolation", _REPO_ROOT / "tools/conftest.py")


def _parameter_ids(params: list[pytest.ParameterSet]) -> set[str]:
    """Return explicit pytest IDs from rendering parameter sets."""
    return {param.id for param in params}


def test_standard_rendering_slices_are_disjoint_and_exhaustive():
    slices = (
        rendering_utils.PHYSX_ISAACSIM_RTX_AOV_COMBINATIONS,
        rendering_utils.NEWTON_ISAACSIM_RTX_AOV_COMBINATIONS,
        rendering_utils.PHYSX_NEWTON_WARP_AOV_COMBINATIONS,
    )
    ids = [_parameter_ids(params) for params in slices]

    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    assert ids[0] | ids[1] | ids[2] == _parameter_ids(rendering_utils.PHYSICS_RENDERER_AOV_COMBINATIONS)


def test_timeout_watchdog_schedules_one_thread_dump(monkeypatch):
    calls = []
    monkeypatch.setattr(
        rendering_utils.faulthandler,
        "dump_traceback_later",
        lambda delay, repeat: calls.append((delay, repeat)),
    )

    rendering_utils.arm_timeout_traceback_watchdog()

    assert calls == [(1200, False)]


def test_shadow_hand_workflow_runs_three_isolated_modules():
    workflow = (_REPO_ROOT / ".github/workflows/build.yaml").read_text(encoding="utf-8")

    assert "test_rendering_shadow_hand.py" not in workflow
    assert all(file_name in workflow for file_name in _SHADOW_TEST_FILES)
    assert not (_SHADOW_TEST_DIR / "test_rendering_shadow_hand.py").exists()
    assert all((_SHADOW_TEST_DIR / file_name).is_file() for file_name in _SHADOW_TEST_FILES)


@pytest.mark.parametrize("file_name", sorted(_SHADOW_TEST_FILES))
def test_shadow_hand_files_receive_one_timeout_retry(file_name: str):
    assert runner._timeout_retries_for_file(file_name) == 1


def test_unrelated_file_receives_no_timeout_retry():
    assert runner._timeout_retries_for_file("test_rendering_franka_cloth.py") == 0
