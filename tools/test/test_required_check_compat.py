# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for CI compatibility status checks."""

import re
from pathlib import Path

_WORKFLOW = Path(__file__).parents[2] / ".github/workflows/build.yaml"
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


def _job_name(job_block: str) -> str:
    """Return the configured display name from a job block."""
    match = re.search(r'^    name: ["\']?(?P<name>.+?)["\']?$', job_block, re.MULTILINE)
    assert match is not None, "job has no display name"
    return match.group("name")


def test_required_compatibility_jobs_publish_stable_contexts() -> None:
    """Each legacy required check has a static job name and matching dependency."""
    workflow = _WORKFLOW.read_text(encoding="utf-8")

    for job_id, (job_name, dependency) in _COMPATIBILITY_JOBS.items():
        block = _job_block(workflow, job_id)
        assert _job_name(block) == job_name
        assert "needs: [changes, " + dependency + "]" in block
        assert "if: always() && github.event_name == 'pull_request'" in block


def test_required_contexts_have_one_job_name_producer() -> None:
    """Matrix runners do not duplicate the stable required-check contexts."""
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    produced_names = [_job_name(_job_block(workflow, job_id)) for job_id in _COMPATIBILITY_JOBS]

    standalone_template = _job_name(_job_block(workflow, "test-standalone-demos"))
    produced_names.extend(
        standalone_template.replace("${{ matrix.runtime-label }}", runtime_label)
        for runtime_label in ("Kit", "non-Kit")
    )

    rendering_template = _job_name(_job_block(workflow, "test-rendering-correctness-kitless"))
    produced_names.extend(
        rendering_template.replace("${{ matrix.variant }}", variant) for variant in ("legacy", "ovstage")
    )

    for required_name, _ in _COMPATIBILITY_JOBS.values():
        assert produced_names.count(required_name) == 1, f"duplicate required check name: {required_name}"


def test_required_compatibility_jobs_accept_intentional_skips() -> None:
    """Docs-only changes pass without requiring a skipped expensive test job."""
    workflow = _WORKFLOW.read_text(encoding="utf-8")

    for job_id in _COMPATIBILITY_JOBS:
        block = _job_block(workflow, job_id)
        assert 'if [ "$SHOULD_RUN" != "true" ]; then' in block
        assert "exit 0" in block
        assert 'if [ "$TEST_RESULT" != "success" ]; then' in block
