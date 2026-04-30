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


def test_required_docker_test_workflow_reports_for_docs_only_prs():
    workflow = _load_workflow("build.yaml")

    pull_request = _on_config(workflow)["pull_request"]
    assert "paths" not in pull_request

    jobs = workflow["jobs"]
    assert jobs["changes"]["outputs"]["run_docker_tests"] == "${{ steps.detect.outputs.run_docker_tests }}"

    for job_name in ("build", "build-curobo"):
        job = jobs[job_name]
        assert "changes" in _as_list(job["needs"])
        assert "needs.changes.outputs.run_docker_tests == 'true'" in job["if"]

    gate = jobs["docker-required-tests-gate"]
    assert gate["name"] == "Docker Required Tests Gate"
    assert gate["if"] == "always()"
    assert "changes" in gate["needs"]
    assert "build" in gate["needs"]
    assert "test-isaaclab-core" in gate["needs"]
    assert "test-isaaclab-core-2" in gate["needs"]
    assert "test-isaaclab-core-3" in gate["needs"]
    assert "test-isaaclab-assets" in gate["needs"]
    assert "test-isaaclab-contrib" in gate["needs"]
    assert "test-isaaclab-newton" in gate["needs"]


def test_required_installation_workflow_reports_for_docs_only_prs():
    workflow = _load_workflow("install-ci.yml")

    pull_request = _on_config(workflow)["pull_request"]
    assert "paths" not in pull_request

    jobs = workflow["jobs"]
    assert jobs["changes"]["outputs"]["run_install_tests"] == "${{ steps.detect.outputs.run_install_tests }}"

    install_tests = jobs["install-tests"]
    assert "changes" in _as_list(install_tests["needs"])
    assert "needs.changes.outputs.run_install_tests == 'true'" in install_tests["if"]

    gate = jobs["installation-tests-gate"]
    assert gate["name"] == "Installation Tests Gate"
    assert gate["if"] == "always()"
    assert gate["needs"] == ["changes", "install-tests"]
