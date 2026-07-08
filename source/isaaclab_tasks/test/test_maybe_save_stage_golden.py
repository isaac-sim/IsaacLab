# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for golden USD stage comparison in maybe_save_stage."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest
from rendering_test_utils import (
    _GOLDEN_STAGES_DIRECTORY,
    GOLDEN_STAGE_RENDERING_TESTS,
    golden_stage_pytest_node_ids,
    maybe_save_stage,
)


@pytest.fixture
def golden_stage_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect golden stage output to a temporary directory."""
    golden_root = tmp_path / "golden_stages"
    monkeypatch.setattr("rendering_test_utils._GOLDEN_STAGES_DIRECTORY", str(golden_root))
    return golden_root


def test_maybe_save_stage_writes_and_matches_golden(golden_stage_dir: Path):
    """maybe_save_stage bootstraps a golden USDA file and passes on the next identical export."""
    stage_text = (
        '#usda 1.0\n(\n    doc = """Generated from Composed Stage of root layer \n"""\n)\ndef Xform "World"\n{\n}\n'
    )

    with mock.patch("isaaclab.sim.save_stage", return_value=True) as save_stage_mock:

        def _write_stage(path: str, save_and_reload_in_place: bool = True) -> bool:
            Path(path).write_text(stage_text, encoding="utf-8")
            return True

        save_stage_mock.side_effect = _write_stage

        with pytest.raises(pytest.fail.Exception, match="Golden stage not found"):
            maybe_save_stage(
                "cartpole",
                "physx",
                "isaacsim_rtx_renderer",
                "rgb",
                compare_golden=True,
            )

        golden_path = golden_stage_dir / "cartpole" / "physx-isaacsim_rtx_renderer-rgb.usda"
        assert golden_path.exists()

        maybe_save_stage(
            "cartpole",
            "physx",
            "isaacsim_rtx_renderer",
            "rgb",
            compare_golden=True,
        )


def test_maybe_save_stage_fails_on_stage_mismatch(golden_stage_dir: Path):
    """maybe_save_stage reports a diff when the exported stage changes."""
    golden_path = golden_stage_dir / "cartpole" / "physx-isaacsim_rtx_renderer-rgb.usda"
    golden_path.parent.mkdir(parents=True)
    golden_path.write_text('def Xform "World"\n{\n    double radius = 1\n}\n', encoding="utf-8")

    result_text = 'def Xform "World"\n{\n    double radius = 2\n}\n'

    with mock.patch("isaaclab.sim.save_stage", return_value=True) as save_stage_mock:

        def _write_stage(path: str, save_and_reload_in_place: bool = True) -> bool:
            Path(path).write_text(result_text, encoding="utf-8")
            return True

        save_stage_mock.side_effect = _write_stage

        with pytest.raises(pytest.fail.Exception, match="USD stage mismatch"):
            maybe_save_stage(
                "cartpole",
                "physx",
                "isaacsim_rtx_renderer",
                "rgb",
                compare_golden=True,
            )


def test_maybe_save_stage_normalizes_platform_and_asset_version(golden_stage_dir: Path):
    """A Windows-generated golden matches a Linux export despite path separators and asset version.

    The baseline is bootstrapped from a Windows-style export (backslash separators, asset
    version ``6.0``); a later export using Linux-style forward slashes and a bumped ``6.1``
    version must still compare equal after canonicalization.
    """
    windows_stage = (
        'def Xform "Robot" (\n'
        "    prepend references = @Assets\\Isaac\\6.0\\Isaac\\IsaacLab\\Robots\\Cartpole\\cartpole.usd@\n"
        ")\n{\n}\n"
    )
    linux_stage = (
        'def Xform "Robot" (\n'
        "    prepend references = @Assets/Isaac/6.1/Isaac/IsaacLab/Robots/Cartpole/cartpole.usd@\n"
        ")\n{\n}\n"
    )

    with mock.patch("isaaclab.sim.save_stage", return_value=True) as save_stage_mock:
        exported = {"text": windows_stage}

        def _write_stage(path: str, save_and_reload_in_place: bool = True) -> bool:
            Path(path).write_text(exported["text"], encoding="utf-8")
            return True

        save_stage_mock.side_effect = _write_stage

        # Pass 1: bootstrap the golden from the Windows-style export.
        with pytest.raises(pytest.fail.Exception, match="Golden stage not found"):
            maybe_save_stage("cartpole", "physx", "isaacsim_rtx_renderer", "rgb", compare_golden=True)

        # Pass 2: a Linux-style export with a bumped asset version must still match.
        exported["text"] = linux_stage
        maybe_save_stage("cartpole", "physx", "isaacsim_rtx_renderer", "rgb", compare_golden=True)


def test_maybe_save_stage_noop_without_dump_or_compare():
    """maybe_save_stage remains a no-op unless compare_golden or ISAAC_LAB_SAVE_STAGES is set."""
    with mock.patch("isaaclab.sim.save_stage") as save_stage_mock:
        maybe_save_stage("cartpole", "physx", "isaacsim_rtx_renderer", "rgb")
        save_stage_mock.assert_not_called()


def test_golden_stage_pytest_node_ids_are_derived_from_rendering_cases():
    """Golden-stage generation discovers tests from shared rendering helpers."""
    node_ids = golden_stage_pytest_node_ids()

    assert len(node_ids) == len(GOLDEN_STAGE_RENDERING_TESTS)
    for rendering_test, node_id in zip(GOLDEN_STAGE_RENDERING_TESTS, node_ids, strict=True):
        assert node_id == rendering_test.pytest_node_ids()[0]


def test_golden_stages_directory_exists_in_repo():
    """The checked-in golden stage directory is present for LFS baselines."""
    assert os.path.isdir(_GOLDEN_STAGES_DIRECTORY)
