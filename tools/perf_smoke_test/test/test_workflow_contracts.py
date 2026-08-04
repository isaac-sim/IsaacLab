# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checks on workflow wiring that the Python tests cannot reach.

Several gate defects have lived in the YAML rather than the code: a fork's
read-only token failing a reporting step, an input default that made the
documented empty value unreachable, a cleanup line naming a file that does not
exist yet. Each is invisible to a unit test of the modules those workflows call,
so the invariants are pinned here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_WORKFLOWS = Path(__file__).resolve().parents[3] / ".github" / "workflows"
_GATE = _WORKFLOWS / "perf-smoke-test.yaml"
_SEED = _WORKFLOWS / "perf-smoke-seed-baselines.yaml"

# Steps that call the GitHub API and would 403 under a fork's read-only token.
_REPORTING_STEPS = ("Report per-task status", "Report aggregate status", "Post verdict PR comment")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _steps(workflow: dict, job: str) -> list[dict]:
    return workflow["jobs"][job].get("steps") or []


def _step(workflow: dict, job: str, name: str) -> dict:
    for step in _steps(workflow, job):
        if step.get("name") == name:
            return step
    raise AssertionError(f"step {name!r} not found in job {job!r}")


@pytest.mark.parametrize("step_name", _REPORTING_STEPS)
def test_api_reporting_is_skipped_for_fork_pull_requests(step_name: str) -> None:
    """A fork PR must not fail the gate just because it cannot write statuses.

    ``pull_request`` from a fork gets a read-only token no matter what the
    permissions block requests, so an unguarded createCommitStatus or
    createComment turns the job red for every external contributor.
    """
    gate = _load(_GATE)
    job = "bench" if step_name == "Report per-task status" else "aggregate"

    condition = str(_step(gate, job, step_name).get("if"))

    assert "github.event.pull_request.head.repo.full_name == github.repository" in condition
    assert "always()" in condition, "the step must still run for failed runs on same-repo PRs"


def test_write_scopes_are_not_granted_workflow_wide() -> None:
    """Jobs that run PR-authored code must not hold a token that can forge reports."""
    gate = _load(_GATE)

    assert gate["permissions"] == {"contents": "read"}
    for job in ("config", "validate"):
        assert gate["jobs"][job].get("permissions") is None, f"{job} should inherit read-only"
    assert gate["jobs"]["bench"]["permissions"] == {"contents": "read", "statuses": "write"}


def test_image_tag_follows_the_target_branch_not_the_merge_ref() -> None:
    """A PR into main or release must benchmark that branch's image, not develop's.

    On ``pull_request`` the ref name is the synthetic ``<n>/merge``, so keying the
    tag off it silently selects latest-develop and produces samples whose
    runtime_contract_hash no baseline on the target branch can match.
    """
    gate = _load(_GATE)

    load_step = next(step for step in _steps(gate, "config") if step.get("id") == "load")
    target = load_step["env"]["TARGET_BRANCH"]

    assert "github.base_ref" in target
    assert "github.event.merge_group.base_ref" in target
    assert "GITHUB_REF_NAME" not in load_step["run"], "the case must switch on the resolved target branch"


def test_retry_clears_the_runtime_bundle_of_the_failed_attempt() -> None:
    """Otherwise a retry that dies early reports the first attempt's FPS as its own."""
    gate = _load(_GATE)

    retry = _step(gate, "bench", "Retry benchmark on failure")["run"]

    assert "benchmark_runtime_*.json" in retry
    assert "rm -f" in retry


@pytest.mark.parametrize("input_name", ["tasks", "branches"])
def test_empty_seed_inputs_reach_the_seeder(input_name: str) -> None:
    """``empty = all tasks`` and ``empty = use commits`` are documented, so they must work.

    ``${{ inputs.x || 'default' }}`` treats an explicitly empty value as unset and
    substitutes the default, which makes those documented modes unreachable.
    """
    seed_source = _SEED.read_text(encoding="utf-8")

    assert f"${{{{ inputs.{input_name} }}}}" in seed_source
    assert f"inputs.{input_name} ||" not in seed_source, "a || default defeats an explicitly empty input"


def test_seed_workflow_declares_the_documented_empty_behavior() -> None:
    """The input description and the expression must agree about what empty means."""
    seed = _load(_SEED)
    call_inputs = seed[True]["workflow_call"]["inputs"]

    assert "empty = all" in call_inputs["tasks"]["description"].lower()


def test_reseed_passes_only_the_credential_the_seeder_declares() -> None:
    """`secrets: inherit` would hand the seeding workflow every repository secret."""
    gate = _load(_GATE)
    seed = _load(_SEED)

    passed = gate["jobs"]["reseed"]["secrets"]

    assert set(passed) == set(seed[True]["workflow_call"]["secrets"])


# --- diagnostics must survive a nonzero aggregate exit ---------------------
#
# The aggregate step runs under `bash -e`. Before this was fixed, a nonzero
# aggregate.py exit aborted the step at the python call, so the job summary was
# never written -- on precisely the runs that needed explaining. On a fork PR,
# where the reporting steps are also skipped, that left a red check with no
# verdict anywhere: not in a comment, not in a status, not in the summary.


def _aggregate_run_block() -> str:
    return _step(_load(_GATE), "aggregate", "Run aggregate oracle")["run"]


def test_aggregate_step_does_not_abort_before_publishing_diagnostics() -> None:
    """`set +e` must wrap the aggregate call so the summary is still written."""
    run = _aggregate_run_block()
    call = run.index("aggregate.py")
    assert "set +e" in run[:call], "aggregate.py must be invoked with errexit disabled"
    assert "AGGREGATE_STATUS=$?" in run, "the aggregate exit code must be captured, not swallowed"


def test_aggregate_step_still_reports_its_exit_code() -> None:
    """Disabling errexit must not silently turn every aggregate run green."""
    run = _aggregate_run_block()
    assert 'exit "${AGGREGATE_STATUS}"' in run, "the captured aggregate exit code must be re-raised"
    assert run.index("AGGREGATE_STATUS=$?") < run.index('exit "${AGGREGATE_STATUS}"')


def test_summary_is_written_on_every_path() -> None:
    """Both branches (summary produced, or not) must append to the step summary."""
    run = _aggregate_run_block()
    assert run.count("GITHUB_STEP_SUMMARY") >= 2, (
        "the step must write to the job summary whether or not aggregate produced a verdict table"
    )
    status_write = run.index("GITHUB_STEP_SUMMARY")
    assert status_write < run.index('exit "${AGGREGATE_STATUS}"'), "diagnostics must be published before exiting"


def test_aggregate_status_reports_the_verdict_not_the_step_outcome() -> None:
    """The commit status must carry the verdict, so advisory mode still signals."""
    step = _step(_load(_GATE), "aggregate", "Report aggregate status")
    env = step.get("env") or {}
    assert "steps.aggregate.outputs.status_state" in env.get("STATUS_STATE", ""), (
        "the aggregate commit status must be driven by the emitted verdict"
    )
    script = step["with"]["script"]
    assert "STATUS_STATE" in script and "STATUS_DESCRIPTION" in script
    # A missing verdict must not be read as success.
    assert "did not produce a verdict" in script


def test_fork_pull_requests_are_told_where_the_verdict_is() -> None:
    """Fork PRs cannot get a comment or a status, so point them at the summary."""
    gate = _load(_GATE)
    step = _step(gate, "aggregate", "Explain skipped reporting (fork pull request)")
    assert "head.repo.full_name != github.repository" in step["if"]
    assert "summary" in step["run"].lower()


# --- protected-branch runs must not cancel each other ----------------------
#
# The push run is the only thing that appends to perf-baselines. develop lands
# ~10 commits a day (median gap ~45 min) against a perf run that takes over an
# hour, so a shared concurrency group cancelled the majority of baseline runs
# and the window could never reach MIN_BASELINE_SAMPLES.


def test_push_runs_are_not_cancelled_by_the_next_push() -> None:
    concurrency = _load(_GATE)["concurrency"]
    assert "github.sha" in concurrency["group"], (
        "protected-branch pushes must get a per-commit concurrency group, "
        "otherwise the next merge cancels the run that publishes baselines"
    )
    assert "github.ref" not in concurrency["group"]


def test_pull_request_runs_still_supersede_each_other() -> None:
    """Cancelling a stale PR run is the useful half of cancel-in-progress."""
    concurrency = _load(_GATE)["concurrency"]
    cancel = str(concurrency["cancel-in-progress"])
    assert "pull_request" in cancel, "cancel-in-progress must remain enabled for pull requests"
    assert cancel.strip() != "true", "cancel-in-progress must not apply unconditionally to pushes"
    assert "github.event.pull_request.number" in concurrency["group"]
