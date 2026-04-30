# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for required CI checks that must always report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_WORKFLOW_DIR = Path(__file__).resolve().parent


def _load_workflow(name: str) -> dict[str, Any]:
    with (_WORKFLOW_DIR / name).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _on_config(workflow: dict[str, Any]) -> dict[str, Any]:
    # PyYAML follows YAML 1.1, where the key "on" is parsed as True.
    return workflow.get("on", workflow.get(True, {}))


def _as_list(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return value
    return [value]


def _assert_job_if_is_exactly(job: dict[str, Any], expected: str) -> None:
    assert job["if"] == expected


def test_required_docker_test_workflow_reports_for_docs_only_prs():
    workflow = _load_workflow("build.yaml")

    pull_request = _on_config(workflow)["pull_request"]
    assert "paths" not in pull_request

    jobs = workflow["jobs"]
    assert jobs["changes"]["outputs"]["run_docker_tests"] == "${{ steps.detect.outputs.run_docker_tests }}"

    for job_name in ("build", "build-curobo"):
        job = jobs[job_name]
        assert "changes" in _as_list(job["needs"])
        _assert_job_if_is_exactly(job, "needs.changes.outputs.run_docker_tests == 'true'")

    gate = jobs["docker-tests-gate"]
    assert gate["name"] == "Docker Tests Gate"
    assert gate["if"] == "always()"
    assert gate["needs"] == [
        "changes",
        "build",
        "build-curobo",
        "test-isaaclab-tasks",
        "test-isaaclab-tasks-2",
        "test-isaaclab-tasks-3",
        "test-isaaclab-core",
        "test-isaaclab-core-2",
        "test-isaaclab-core-3",
        "test-isaaclab-rl",
        "test-isaaclab-mimic",
        "test-isaaclab-assets",
        "test-isaaclab-contrib",
        "test-isaaclab-teleop",
        "test-isaaclab-visualizers",
        "test-isaaclab-newton",
        "test-isaaclab-physx",
        "test-isaaclab-ov",
        "test-curobo",
        "test-skillgen",
        "test-environments-training",
    ]


def test_required_installation_workflow_reports_for_docs_only_prs():
    workflow = _load_workflow("install-ci.yml")

    pull_request = _on_config(workflow)["pull_request"]
    assert "paths" not in pull_request

    jobs = workflow["jobs"]
    assert jobs["changes"]["outputs"]["run_install_tests"] == "${{ steps.detect.outputs.run_install_tests }}"

    install_tests = jobs["install-tests"]
    assert "changes" in _as_list(install_tests["needs"])
    _assert_job_if_is_exactly(install_tests, "needs.changes.outputs.run_install_tests == 'true'")

    gate = jobs["installation-tests-gate"]
    assert gate["name"] == "Installation Tests Gate"
    assert gate["if"] == "always()"
    assert gate["needs"] == ["changes", "install-tests"]
