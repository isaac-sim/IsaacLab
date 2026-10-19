# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``tools/ci_marker_jobs.py``."""

from __future__ import annotations

import json
import os

import pytest
import yaml
from ci_marker_jobs import RUN_PACKAGE_TESTS, WORKSPACE_ROOT, main, resolve

BUILD_WORKFLOW = os.path.join(WORKSPACE_ROOT, ".github", "workflows", "build.yaml")


def _marker_jobs(tmp_path, monkeypatch, capsys, workflow: dict, marker: str = "kitless") -> dict[str, bool]:
    workflow_json = tmp_path / "build.json"
    workflow_json.write_text(json.dumps(workflow))
    monkeypatch.setattr(
        "sys.argv",
        [
            "ci_marker_jobs.py",
            "--workflow",
            str(workflow_json),
            "--marker",
            marker,
            "--event-name",
            "workflow_dispatch",
        ],
    )
    assert main() == 0
    output = capsys.readouterr().out.strip()
    assert output.startswith("marker_jobs=")
    return json.loads(output.removeprefix("marker_jobs="))


def _job(**inputs) -> dict:
    return {"steps": [{"uses": RUN_PACKAGE_TESTS, "with": {"ci-marker": "${{ inputs.marker }}", **inputs}}]}


@pytest.mark.parametrize(
    ("value", "event_name", "expected"),
    [
        ("isaaclab_ov", "workflow_dispatch", "isaaclab_ov"),
        ("${{ github.event_name == 'push' && 'rendering' || '' }}", "push", "rendering"),
        ("${{ github.event_name == 'push' && 'rendering' || '' }}", "workflow_dispatch", ""),
        ("${{ needs.config.outputs.subset }}", "workflow_dispatch", None),
    ],
)
def test_resolve_evaluates_only_event_conditional_values(value, event_name, expected):
    assert resolve(value, event_name) == expected


def test_jobs_without_marked_files_are_skipped(tmp_path, monkeypatch, capsys):
    jobs = _marker_jobs(
        tmp_path,
        monkeypatch,
        capsys,
        {
            "jobs": {
                "ov": _job(**{"filter-pattern": "isaaclab_ov"}),
                "no-files": _job(**{"filter-pattern": "no_such_package"}),
                # The legacy "not <pattern>" filter excludes instead of including; core has kitless files.
                "core": _job(**{"filter-pattern": "not isaaclab_"}),
                "unknown-input": _job(**{"filter-pattern": "${{ needs.config.outputs.subset }}"}),
                "ignores-marker": {"steps": [{"uses": RUN_PACKAGE_TESTS, "with": {"filter-pattern": "isaaclab_ov"}}]},
            }
        },
    )

    assert jobs == {"ov": True, "no-files": False, "core": True, "unknown-input": True}


def test_unregistered_marker_fails(tmp_path, monkeypatch, capsys):
    workflow_json = tmp_path / "build.json"
    workflow_json.write_text(json.dumps({"jobs": {}}))
    monkeypatch.setattr(
        "sys.argv",
        ["ci_marker_jobs.py", "--workflow", str(workflow_json), "--marker", "kitles", "--event-name", "push"],
    )

    assert main() == 1
    assert "kitles" in capsys.readouterr().out


def test_every_job_forwarding_the_marker_is_gated_on_its_result():
    with open(BUILD_WORKFLOW) as f:
        jobs = yaml.safe_load(f)["jobs"]

    forwarding = {
        job_id
        for job_id, job in jobs.items()
        for step in job.get("steps", [])
        if "inputs.marker" in str((step.get("with") or {}).get("ci-marker", ""))
    }
    gated = {job_id for job_id, job in jobs.items() if f"marker_jobs)['{job_id}']" in str(job.get("if", ""))}

    assert forwarding
    assert forwarding == gated
