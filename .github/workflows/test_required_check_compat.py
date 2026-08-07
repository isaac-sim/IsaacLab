# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for CI compatibility status checks."""

from pathlib import Path
import re


_WORKFLOW = Path(__file__).with_name("build.yaml")
_COMPATIBILITY_JOBS = {
    "required-standalone-demos-kit": (
        "standalone demos (headless, Kit)",
        "test-standalone-demos",
    ),
    "required-standalone-demos-non-kit": (
        "standalone demos (headless, non-Kit)",
        "test-standalone-demos",
    ),
    "required-rendering-correctness-kitless-legacy": (
        "rendering-correctness-kitless (legacy)",
        "test-rendering-correctness-kitless",
    ),
}


def _job_block(workflow: str, job_id: str) -> str:
    """Return one top-level job block from the workflow text."""
    match = re.search(rf"^  {re.escape(job_id)}:\n(?P<body>(?:^(?!  \w).*(?:\n|$))*)", workflow, re.MULTILINE)
    assert match is not None, f"missing compatibility job: {job_id}"
    return match.group("body")


def test_required_compatibility_jobs_publish_stable_contexts() -> None:
    """Each legacy required check has a static job name and matching dependency."""
    workflow = _WORKFLOW.read_text(encoding="utf-8")

    for job_id, (job_name, dependency) in _COMPATIBILITY_JOBS.items():
        block = _job_block(workflow, job_id)
        assert f"name: {job_name}" in block
        assert "needs: [changes, " + dependency + "]" in block
        assert "if: always() && github.event_name == 'pull_request'" in block


def test_required_compatibility_jobs_accept_intentional_skips() -> None:
    """Docs-only changes pass without requiring a skipped expensive test job."""
    workflow = _WORKFLOW.read_text(encoding="utf-8")

    for job_id in _COMPATIBILITY_JOBS:
        block = _job_block(workflow, job_id)
        assert 'if [ "$SHOULD_RUN" != "true" ]; then' in block
        assert 'exit 0' in block
        assert 'if [ "$TEST_RESULT" != "success" ]; then' in block
