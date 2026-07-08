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
    canonicalize_stage_text,
    compares_golden_stage,
    golden_stage_pytest_node_ids,
    maybe_save_stage,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


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


def test_golden_stage_pytest_node_ids_cover_every_backend():
    """Golden-stage generation covers every parametrized case (all backends), not a subset."""
    node_ids = golden_stage_pytest_node_ids()

    expected_count = sum(len(rendering_test.combinations) for rendering_test in GOLDEN_STAGE_RENDERING_TESTS)
    assert len(node_ids) == expected_count
    assert len(set(node_ids)) == len(node_ids), "duplicate node IDs would double-generate a golden"


def test_golden_stages_directory_exists_in_repo():
    """The checked-in golden stage directory is present for LFS baselines."""
    assert os.path.isdir(_GOLDEN_STAGES_DIRECTORY)


def test_registered_test_functions_compare_golden():
    """Every test in the registry is reported as a golden-stage comparer (single source of truth)."""
    for rendering_test in GOLDEN_STAGE_RENDERING_TESTS:
        assert compares_golden_stage(rendering_test.test_function)
    assert not compares_golden_stage("test_not_registered")


def test_registry_test_functions_are_unique():
    """``compares_golden_stage`` keys on ``test_function``, so each must be distinct."""
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
    )
    assert rendering_test.pytest_node_ids() == (
        "source/isaaclab_tasks/test/core/test_rendering_registered_tasks.py"
        "::test_rendering_registered_tasks[Isaac-Cartpole-Camera-Direct-None-cartpole]",
    )


def test_canonicalize_stage_text_masks_volatile_fields():
    """Canonicalization strips provenance, anon layers, and portability-masks asset references."""
    raw = (
        '#usda 1.0\n(\n    doc = """Generated at 12:00\nfrom root layer\n"""\n)\n'
        'def "World" (\n'
        "    prepend references = @Assets\\Isaac\\6.0\\Isaac\\IsaacLab\\Robots\\Cartpole\\cartpole.usd@\n"
        ")\n{\n"
        "    add references = @anon:0x1234abcd:World.usd@\n"
        "}\n"
    )
    canonical = canonicalize_stage_text(raw)

    assert 'doc = """"""' in canonical
    assert "12:00" not in canonical
    assert "6.0" not in canonical and "<VERSION>" in canonical
    assert "\\" not in canonical
    assert "0x1234abcd" not in canonical and "<ANON_LAYER>" in canonical
