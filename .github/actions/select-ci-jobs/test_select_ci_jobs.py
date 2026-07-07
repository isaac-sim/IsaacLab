# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the path-based CI job selector."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from select_ci_jobs import _BASE_JOBS, select_ci_jobs

_TASKS_FAMILY = {"tasks", "rendering", "rendering_kitless", "environments_training", "curobo", "skillgen"}


def _true_jobs(paths: list[str]) -> set[str]:
    """Return the set of base job groups selected to run for ``paths``."""
    flags = select_ci_jobs(paths)
    return {job for job in _BASE_JOBS if flags[job]}


def test_empty_change_set_runs_everything_as_failsafe():
    assert _true_jobs([]) == set(_BASE_JOBS)
    assert select_ci_jobs([])["any"] is True


def test_core_change_runs_everything():
    assert _true_jobs(["source/isaaclab/isaaclab/assets/articulation.py"]) == set(_BASE_JOBS)


@pytest.mark.parametrize(
    "path",
    [
        "source/isaaclab_assets/x.py",
        "source/isaaclab_tasks/x.py",
        "source/isaaclab_newton/x.py",
        "source/isaaclab_physx/x.py",
        "source/isaaclab_experimental/x.py",
        "source/isaaclab_ppisp/x.py",
        "source/isaaclab_tasks_experimental/x.py",
    ],
)
def test_hub_and_shared_packages_run_everything(path):
    assert _true_jobs([path]) == set(_BASE_JOBS)


@pytest.mark.parametrize(
    "path",
    [
        "docker/Dockerfile.base",
        "tools/conftest.py",
        "apps/foo.kit",
        "scripts/train.py",
        ".github/actions/run-tests/action.yml",
        ".github/workflows/build.yaml",
        ".github/workflows/config.yaml",
        ".github/test-subsets/postmerge-rendering.toml",
        "pyproject.toml",
        "isaaclab.sh",
        ".gitmodules",
    ],
)
def test_tooling_and_ci_changes_run_everything(path):
    assert _true_jobs([path]) == set(_BASE_JOBS)


def test_mimic_only_runs_mimic():
    assert _true_jobs(["source/isaaclab_mimic/isaaclab_mimic/foo.py"]) == {"mimic"}


def test_rl_only_runs_rl_and_tasks_family():
    assert _true_jobs(["source/isaaclab_rl/isaaclab_rl/foo.py"]) == {"rl"} | _TASKS_FAMILY


def test_teleop_only_runs_teleop_tasks_family_and_core():
    assert _true_jobs(["source/isaaclab_teleop/foo.py"]) == {"teleop", "core"} | _TASKS_FAMILY


def test_visualizers_only_runs_visualizers_and_core():
    assert _true_jobs(["source/isaaclab_visualizers/foo.py"]) == {"visualizers", "core"}


@pytest.mark.parametrize("pkg", ["isaaclab_ov", "isaaclab_ovphysx"])
def test_ov_packages_run_ov_tasks_family_and_core(pkg):
    assert _true_jobs([f"source/{pkg}/foo.py"]) == {"ov", "core"} | _TASKS_FAMILY


def test_contrib_only_runs_contrib_and_dependents():
    assert _true_jobs(["source/isaaclab_contrib/foo.py"]) == (
        {"contrib", "core", "newton", "physx", "assets"} | _TASKS_FAMILY
    )


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "docs/source/setup/installation/pip_installation.rst",
        "source/isaaclab_mimic/docs/CHANGELOG.rst",
        "source/isaaclab/changelog.d/foo.skip",
        "source/isaaclab_tasks/docs/index.md",
    ],
)
def test_docs_and_changelog_only_run_nothing(path):
    assert _true_jobs([path]) == set()
    assert select_ci_jobs([path])["any"] is False


def test_unrelated_files_run_nothing():
    # A change touching only files that cannot affect the docker tests (e.g. an unrelated
    # workflow or license metadata) selects no jobs, matching the previous all-or-nothing gate.
    assert _true_jobs([".github/workflows/labeler.yml", "LICENSE"]) == set()


def test_leaf_plus_hub_falls_back_to_everything():
    assert _true_jobs(["source/isaaclab_mimic/foo.py", "source/isaaclab/isaaclab/bar.py"]) == set(_BASE_JOBS)


def test_multiple_leaf_packages_union_their_jobs():
    assert _true_jobs(["source/isaaclab_mimic/foo.py", "source/isaaclab_visualizers/bar.py"]) == {
        "mimic",
        "visualizers",
        "core",
    }


def test_unknown_source_package_falls_back_to_everything():
    assert _true_jobs(["source/isaaclab_brandnew/foo.py"]) == set(_BASE_JOBS)


def test_derived_flags():
    # curobo_image tracks the curobo image jobs; any tracks build gating.
    docs = select_ci_jobs(["README.md"])
    assert docs["curobo_image"] is False and docs["any"] is False

    rl = select_ci_jobs(["source/isaaclab_rl/foo.py"])
    assert rl["curobo_image"] is True  # curobo + skillgen are in the tasks family

    mimic = select_ci_jobs(["source/isaaclab_mimic/foo.py"])
    assert mimic["curobo_image"] is False and mimic["any"] is True
