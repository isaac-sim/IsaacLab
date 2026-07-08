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
    GoldenStageRenderingTest,
    _parametrize_case_id_and_values,
    compare_golden_stage,
    golden_stage_pytest_node_ids,
    maybe_save_stage,
    should_compare_golden_stage,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


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


def test_maybe_save_stage_noop_without_dump_or_compare():
    """maybe_save_stage remains a no-op unless compare_golden or ISAAC_LAB_SAVE_STAGES is set."""
    with mock.patch("isaaclab.sim.save_stage") as save_stage_mock:
        maybe_save_stage("cartpole", "physx", "isaacsim_rtx_renderer", "rgb")
        save_stage_mock.assert_not_called()


def test_golden_stage_node_ids_match_selected_cases():
    """Generation node IDs are exactly the cases ``should_compare`` selects, with no duplicates."""
    node_ids = golden_stage_pytest_node_ids()

    expected = [node_id for rt in GOLDEN_STAGE_RENDERING_TESTS for node_id in rt.pytest_node_ids()]
    assert list(node_ids) == expected
    assert len(set(node_ids)) == len(node_ids), "duplicate node IDs would double-generate a golden"
    # The representative subset selects one AOV per cartpole backend/renderer and the default variant
    # per registered task; guard the expected count so an accidental predicate change is caught.
    assert len(node_ids) == 5


def test_runtime_selection_matches_generation_selection():
    """The call-site dispatcher selects exactly the cases golden generation bootstraps.

    If these diverged, a normally-run test could compare against a golden the generator never
    created, failing with "Golden stage not found" for a baseline nobody committed.
    """
    for rendering_test in GOLDEN_STAGE_RENDERING_TESTS:
        generated = set(rendering_test.pytest_node_ids())
        for entry in rendering_test.combinations:
            case_id, values = _parametrize_case_id_and_values(entry)
            node_id = (
                f"{rendering_test.test_root}/{rendering_test.test_module}::{rendering_test.test_function}[{case_id}]"
            )
            assert should_compare_golden_stage(rendering_test.test_function, *values) == (node_id in generated)
    assert not should_compare_golden_stage("test_not_registered", "x")


def test_golden_stages_directory_exists_in_repo():
    """The checked-in golden stage directory is present for LFS baselines."""
    assert os.path.isdir(_GOLDEN_STAGES_DIRECTORY)


def test_representative_subset_covers_every_backend_and_task():
    """The selected cartpole cases span every backend/renderer, one AOV each; registered spans tasks."""
    cartpole = next(rt for rt in GOLDEN_STAGE_RENDERING_TESTS if rt.test_function == "test_rendering_cartpole")
    selected = [
        _parametrize_case_id_and_values(e)[1]
        for e in cartpole.combinations
        if should_compare_golden_stage("test_rendering_cartpole", *_parametrize_case_id_and_values(e)[1])
    ]
    # One selected case per (physics_backend, renderer), all rgb.
    backend_renderers = {(values[0], values[1]) for values in selected}
    assert backend_renderers == {
        ("physx", "isaacsim_rtx_renderer"),
        ("newton", "isaacsim_rtx_renderer"),
        ("physx", "newton_renderer"),
    }
    assert all(values[2] == "rgb" for values in selected)


def test_registry_test_functions_are_unique():
    """``should_compare_golden_stage`` keys on ``test_function``, so each must be distinct."""
    functions = [rendering_test.test_function for rendering_test in GOLDEN_STAGE_RENDERING_TESTS]
    assert len(functions) == len(set(functions))


def test_registry_targets_exist_on_disk():
    """Guard against a module rename/move silently orphaning a golden-stage registry entry."""
    for rendering_test in GOLDEN_STAGE_RENDERING_TESTS:
        module_path = _REPO_ROOT / rendering_test.test_root / rendering_test.test_module
        assert module_path.is_file(), f"missing test module: {module_path}"
        source = module_path.read_text(encoding="utf-8")
        assert f"def {rendering_test.test_function}(" in source, (
            f"{rendering_test.test_module} no longer defines {rendering_test.test_function}"
        )


def test_pytest_param_without_explicit_id_is_parsed():
    """A ``pytest.param`` without ``id=`` yields its values (ParameterSet is itself a tuple)."""
    param = pytest.param("Isaac-Cartpole-Camera-Direct", None, "cartpole", marks=pytest.mark.flaky)
    case_id, values = _parametrize_case_id_and_values(param)

    assert values == ("Isaac-Cartpole-Camera-Direct", None, "cartpole")
    assert case_id == "Isaac-Cartpole-Camera-Direct-None-cartpole"


def test_pytest_node_id_derived_from_id_less_param():
    """An id-less ``pytest.param`` still yields pytest's default node ID (ParameterSet is a tuple)."""
    rendering_test = GoldenStageRenderingTest(
        test_module="test_rendering_registered_tasks.py",
        test_function="test_rendering_registered_tasks",
        combinations=[pytest.param("Isaac-Cartpole-Camera-Direct", None, "cartpole", marks=pytest.mark.flaky)],
        should_compare=lambda *_: True,
    )
    assert rendering_test.pytest_node_ids() == (
        "source/isaaclab_tasks/test/core/test_rendering_registered_tasks.py"
        "::test_rendering_registered_tasks[Isaac-Cartpole-Camera-Direct-None-cartpole]",
    )
