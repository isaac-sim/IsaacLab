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
    compare_golden_stage,
    maybe_save_stage,
)


def _usda_with_robot(translate: tuple[float, float, float] = (0.0, 0.0, 0.0), *, extra_prim: bool = False) -> str:
    """Build a self-contained USDA stage with a translated ``/World/Robot`` Xform (no asset refs)."""
    robot = (
        '    def Xform "Robot"\n'
        "    {\n"
        f"        double3 xformOp:translate = ({translate[0]}, {translate[1]}, {translate[2]})\n"
        '        uniform token[] xformOpOrder = ["xformOp:translate"]\n'
        "    }\n"
    )
    extra = '    def Xform "Extra"\n    {\n    }\n' if extra_prim else ""
    return f'#usda 1.0\ndef Xform "World"\n{{\n{robot}{extra}}}\n'


def _mock_export(stage_text: str | dict[str, str]):
    """Return a ``save_stage`` side effect that writes ``stage_text`` (or ``stage_text['text']``)."""

    def _write_stage(path: str, save_and_reload_in_place: bool = True) -> bool:
        text = stage_text["text"] if isinstance(stage_text, dict) else stage_text
        Path(path).write_text(text, encoding="utf-8")
        return True

    return _write_stage


@pytest.fixture
def golden_stage_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect golden stage output to a temporary directory."""
    golden_root = tmp_path / "golden_stages"
    monkeypatch.setattr("rendering_test_utils._GOLDEN_STAGES_DIRECTORY", str(golden_root))
    return golden_root


def test_maybe_save_stage_writes_and_matches_golden(golden_stage_dir: Path):
    """maybe_save_stage bootstraps a golden USDA file and passes on the next identical export."""
    with mock.patch("isaaclab.sim.save_stage", return_value=True) as save_stage_mock:
        save_stage_mock.side_effect = _mock_export(_usda_with_robot((1.0, 2.0, 3.0)))

        with pytest.raises(pytest.fail.Exception, match="Golden stage not found"):
            maybe_save_stage("cartpole", "physx", "isaacsim_rtx_renderer", "rgb", compare_golden=True)

        golden_path = golden_stage_dir / "cartpole" / "physx-isaacsim_rtx_renderer-rgb.usda"
        assert golden_path.exists()

        maybe_save_stage("cartpole", "physx", "isaacsim_rtx_renderer", "rgb", compare_golden=True)


def test_maybe_save_stage_fails_on_transform_mismatch(golden_stage_dir: Path):
    """maybe_save_stage fails when a prim's transform moves beyond tolerance."""
    exported = {"text": _usda_with_robot((1.0, 2.0, 3.0))}
    with mock.patch("isaaclab.sim.save_stage", return_value=True) as save_stage_mock:
        save_stage_mock.side_effect = _mock_export(exported)

        with pytest.raises(pytest.fail.Exception, match="Golden stage not found"):
            maybe_save_stage("cartpole", "physx", "isaacsim_rtx_renderer", "rgb", compare_golden=True)

        # A meaningfully moved prim must fail the comparison.
        exported["text"] = _usda_with_robot((1.0, 2.0, 9.0))
        with pytest.raises(pytest.fail.Exception, match="USD stage mismatch"):
            maybe_save_stage("cartpole", "physx", "isaacsim_rtx_renderer", "rgb", compare_golden=True)


def test_maybe_save_stage_tolerates_small_transform_noise(golden_stage_dir: Path):
    """A sub-tolerance transform difference (per-platform float noise) still compares equal."""
    exported = {"text": _usda_with_robot((1.0, 2.0, 3.0))}
    with mock.patch("isaaclab.sim.save_stage", return_value=True) as save_stage_mock:
        save_stage_mock.side_effect = _mock_export(exported)

        with pytest.raises(pytest.fail.Exception, match="Golden stage not found"):
            maybe_save_stage("cartpole", "physx", "isaacsim_rtx_renderer", "rgb", compare_golden=True)

        # 1e-7 is well below the 1e-5 absolute tolerance and must not trip the comparison.
        exported["text"] = _usda_with_robot((1.0, 2.0, 3.0 + 1e-7))
        maybe_save_stage("cartpole", "physx", "isaacsim_rtx_renderer", "rgb", compare_golden=True)


def test_maybe_save_stage_fails_on_added_prim(golden_stage_dir: Path):
    """maybe_save_stage fails when the exported stage adds a prim (structure change)."""
    exported = {"text": _usda_with_robot((1.0, 2.0, 3.0))}
    with mock.patch("isaaclab.sim.save_stage", return_value=True) as save_stage_mock:
        save_stage_mock.side_effect = _mock_export(exported)

        with pytest.raises(pytest.fail.Exception, match="Golden stage not found"):
            maybe_save_stage("cartpole", "physx", "isaacsim_rtx_renderer", "rgb", compare_golden=True)

        exported["text"] = _usda_with_robot((1.0, 2.0, 3.0), extra_prim=True)
        with pytest.raises(pytest.fail.Exception, match="USD stage mismatch"):
            maybe_save_stage("cartpole", "physx", "isaacsim_rtx_renderer", "rgb", compare_golden=True)


def test_compare_golden_stage_reports_structure_and_transform_diffs(tmp_path: Path):
    """compare_golden_stage returns no problems when equal and flags added prims and moved transforms."""
    golden = tmp_path / "golden.usda"
    result = tmp_path / "result.usda"

    golden.write_text(_usda_with_robot((1.0, 2.0, 3.0)), encoding="utf-8")
    result.write_text(_usda_with_robot((1.0, 2.0, 3.0)), encoding="utf-8")
    assert compare_golden_stage(str(golden), str(result)) == []

    result.write_text(_usda_with_robot((1.0, 2.0, 3.0), extra_prim=True), encoding="utf-8")
    problems = compare_golden_stage(str(golden), str(result))
    assert any("added prim" in problem and "/World/Extra" in problem for problem in problems)

    result.write_text(_usda_with_robot((1.0, 2.0, 9.0)), encoding="utf-8")
    problems = compare_golden_stage(str(golden), str(result))
    assert any("transform" in problem and "/World/Robot" in problem for problem in problems)


def test_maybe_save_stage_matches_golden_with_sublayer_result(golden_stage_dir: Path, tmp_path: Path):
    """Comparison still passes when save_stage writes a root layer with a sublayer reference.

    The bootstrap flattens the result stage to produce the golden.  On the second call the
    result also has sublayer references, but both sides are flattened before comparison so
    the composed prim structure and transforms match the golden exactly.
    """
    sublayer = tmp_path / "sub.usda"
    sublayer.write_text(_usda_with_robot((1.0, 2.0, 3.0)), encoding="utf-8")

    # Root layer that pulls all prims in via a sublayer — no inline definitions.
    root_usda = f"#usda 1.0\n(\n    subLayers = [\n        @{sublayer}@\n    ]\n)\n"

    with mock.patch("isaaclab.sim.save_stage", return_value=True) as save_stage_mock:
        save_stage_mock.side_effect = _mock_export(root_usda)

        with pytest.raises(pytest.fail.Exception, match="Golden stage not found"):
            maybe_save_stage("cartpole", "physx", "isaacsim_rtx_renderer", "rgb", compare_golden=True)

        # Same sublayered stage must compare equal to the flattened golden.
        maybe_save_stage("cartpole", "physx", "isaacsim_rtx_renderer", "rgb", compare_golden=True)


def test_maybe_save_stage_noop_without_dump_or_compare(monkeypatch: pytest.MonkeyPatch):
    """maybe_save_stage remains a no-op unless compare_golden or ISAAC_LAB_SAVE_STAGES is set."""
    monkeypatch.delenv("ISAAC_LAB_SAVE_STAGES", raising=False)
    with mock.patch("isaaclab.sim.save_stage") as save_stage_mock:
        maybe_save_stage("cartpole", "physx", "isaacsim_rtx_renderer", "rgb")
        save_stage_mock.assert_not_called()


def test_golden_stages_directory_exists_in_repo():
    """The checked-in golden stage directory is present for LFS baselines."""
    assert os.path.isdir(_GOLDEN_STAGES_DIRECTORY)
