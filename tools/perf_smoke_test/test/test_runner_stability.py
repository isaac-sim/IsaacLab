# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free tests for runner-pool FPS stability qualification."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

import runner_stability  # noqa: E402

_BUCKET = {("task-a", "newton")}


def _records() -> list[dict]:
    records: list[dict] = []
    runner_offsets = (-1.0, -0.5, 0.0, 0.5, 1.0)
    sample_offsets = (-0.25, 0.0, 0.25)
    for allocation_index, runner_offset in enumerate(runner_offsets, start=1):
        for sample_index, sample_offset in enumerate(sample_offsets):
            records.append(
                {
                    "gpu_model": "l40s",
                    "task_id": "task-a",
                    "backend": "newton",
                    "target_branch": "develop",
                    "commit_sha": "commit-a",
                    "launch_config_hash": "launch-a",
                    "benchmark_contract_hash": "benchmark-a",
                    "runtime_contract_hash": "runtime-a",
                    "baseline_epoch": 1,
                    "ci_run_label": f"allocation-{allocation_index}",
                    "ci_runner_name": f"runner-{allocation_index}",
                    "sample_index": sample_index,
                    "sample_id": f"sample-{allocation_index}-{sample_index}",
                    "fps": 100.0 + runner_offset + sample_offset,
                }
            )
    return records


def test_complete_stable_pool_qualifies_for_gating() -> None:
    """Five homogeneous allocations with low FPS dispersion produce a pass."""
    overall, buckets = runner_stability.evaluate_stability(
        _records(),
        expected_buckets=_BUCKET,
    )

    assert overall == runner_stability.STABLE
    assert buckets[0].verdict == runner_stability.STABLE
    assert buckets[0].allocation_count == 5
    assert buckets[0].runner_count == 5
    assert buckets[0].sample_count == 15
    assert buckets[0].effective_block_regression_pct is not None
    assert buckets[0].effective_block_regression_pct <= runner_stability.MAX_EFFECTIVE_BLOCK_PCT


def test_runtime_heterogeneity_is_inconclusive() -> None:
    """Different runtime fingerprints cannot be pooled into stability evidence."""
    records = _records()
    for record in records[-3:]:
        record["runtime_contract_hash"] = "runtime-b"

    overall, buckets = runner_stability.evaluate_stability(records, expected_buckets=_BUCKET)

    assert overall == runner_stability.INCONCLUSIVE
    assert "expected one runtime fingerprint, observed 2" in buckets[0].reasons


def test_missing_runtime_fingerprint_fields_are_inconclusive() -> None:
    """Absent provenance cannot establish a homogeneous runtime environment."""
    records = _records()
    missing_fields = ("launch_config_hash", "benchmark_contract_hash", "runtime_contract_hash", "baseline_epoch")
    for record in records:
        for field in missing_fields:
            record.pop(field)

    overall, buckets = runner_stability.evaluate_stability(records, expected_buckets=_BUCKET)

    assert overall == runner_stability.INCONCLUSIVE
    assert (
        "missing runtime fingerprint fields: baseline_epoch, benchmark_contract_hash, "
        "launch_config_hash, runtime_contract_hash"
    ) in buckets[0].reasons


@pytest.mark.parametrize("fps", [0.0, -1.0])
def test_non_positive_fps_evidence_is_inconclusive(fps: float) -> None:
    """Degenerate FPS measurements cannot qualify regression sensitivity."""
    records = _records()
    for record in records:
        record["fps"] = fps

    overall, buckets = runner_stability.evaluate_stability(records, expected_buckets=_BUCKET)

    assert overall == runner_stability.INCONCLUSIVE
    assert buckets[0].sample_count == 0
    assert buckets[0].median_fps is None


def test_missing_runner_allocation_is_inconclusive() -> None:
    """Partial evidence cannot accidentally qualify the pool."""
    records = [record for record in _records() if record["ci_run_label"] != "allocation-5"]

    overall, buckets = runner_stability.evaluate_stability(records, expected_buckets=_BUCKET)

    assert overall == runner_stability.INCONCLUSIVE
    assert any("expected 5 allocations, observed 4" in reason for reason in buckets[0].reasons)
    assert any("expected 15 samples, observed 12" in reason for reason in buckets[0].reasons)


def test_duplicate_sample_identity_is_inconclusive() -> None:
    """Repeated evidence rows cannot satisfy the required sample count."""
    records = _records()
    records[1]["sample_id"] = records[0]["sample_id"]

    overall, buckets = runner_stability.evaluate_stability(records, expected_buckets=_BUCKET)

    assert overall == runner_stability.INCONCLUSIVE
    assert "observed 1 duplicate sample identities" in buckets[0].reasons


def test_duplicate_sample_index_is_inconclusive() -> None:
    """Each allocation must contain every requested repetition exactly once."""
    records = _records()
    records[1]["sample_index"] = records[0]["sample_index"]

    overall, buckets = runner_stability.evaluate_stability(records, expected_buckets=_BUCKET)

    assert overall == runner_stability.INCONCLUSIVE
    assert "allocations without sample indexes 0..2: allocation-1" in buckets[0].reasons


def test_reused_runners_can_cover_all_independent_allocations() -> None:
    """Queued allocations remain useful when the pool has at least three runners."""
    records = _records()
    allocation_runners = {
        "allocation-1": "runner-a",
        "allocation-2": "runner-a",
        "allocation-3": "runner-b",
        "allocation-4": "runner-b",
        "allocation-5": "runner-c",
    }
    for record in records:
        record["ci_runner_name"] = allocation_runners[record["ci_run_label"]]

    overall, buckets = runner_stability.evaluate_stability(
        records,
        expected_buckets=_BUCKET,
        minimum_distinct_runners=3,
    )

    assert overall == runner_stability.STABLE
    assert buckets[0].allocation_count == 5
    assert buckets[0].runner_count == 3
    assert set(buckets[0].runner_medians_fps) == {"runner-a", "runner-b", "runner-c"}


def test_insufficient_distinct_runner_coverage_is_inconclusive() -> None:
    """The report fails closed when serialization reveals too little pool coverage."""
    records = _records()
    for record in records:
        record["ci_runner_name"] = f"runner-{int(record['ci_run_label'][-1]) % 2}"

    overall, buckets = runner_stability.evaluate_stability(
        records,
        expected_buckets=_BUCKET,
        minimum_distinct_runners=3,
    )

    assert overall == runner_stability.INCONCLUSIVE
    assert "expected at least 3 distinct runner names, observed 2" in buckets[0].reasons


def test_runner_specific_fps_bias_fails_qualification() -> None:
    """A runner whose median is materially shifted makes the pool unsafe."""
    records = _records()
    for record in records:
        if record["ci_run_label"] == "allocation-5":
            record["fps"] += 12.0

    overall, buckets = runner_stability.evaluate_stability(records, expected_buckets=_BUCKET)

    assert overall == runner_stability.UNSTABLE
    assert any("runner median deviation" in reason for reason in buckets[0].reasons)


def test_single_fps_outlier_fails_qualification() -> None:
    """A noisy one-off sample is unsafe because each PR gate observes one run."""
    records = _records()
    records[0]["fps"] = 88.0

    overall, buckets = runner_stability.evaluate_stability(records, expected_buckets=_BUCKET)

    assert overall == runner_stability.UNSTABLE
    assert any("single-sample deviation" in reason for reason in buckets[0].reasons)


def test_configured_noise_floor_can_make_gate_too_insensitive() -> None:
    """Qualification accounts for the same noise floor used by the oracle."""
    overall, buckets = runner_stability.evaluate_stability(
        _records(),
        expected_buckets=_BUCKET,
        noise_floors={("task-a", "newton"): 3.0},
    )

    assert overall == runner_stability.UNSTABLE
    assert buckets[0].effective_block_regression_pct == 12.0
    assert any("effective BLOCK band" in reason for reason in buckets[0].reasons)


def test_expected_scope_guards_accept_matching_evidence() -> None:
    """The workflow's expected GPU, branch, and commit match valid records."""
    overall, buckets = runner_stability.evaluate_stability(
        _records(),
        expected_buckets=_BUCKET,
        expected_gpu_model="NVIDIA L40S",
        expected_target_branch="develop",
        expected_commit="commit-a",
    )

    assert overall == runner_stability.STABLE
    assert buckets[0].reasons == ()


def test_empty_configured_scope_is_inconclusive() -> None:
    """A missing task matrix cannot qualify vacuously."""
    report, markdown = runner_stability.build_report(_records(), expected_buckets=set())

    assert report["overall_verdict"] == runner_stability.INCONCLUSIVE
    assert report["buckets"] == []
    assert "No task/backend buckets were configured." in markdown


@pytest.mark.parametrize(
    ("scope_override", "expected_reason"),
    [
        ({"expected_gpu_model": "a100"}, "expected GPU a100"),
        ({"expected_target_branch": "main"}, "expected target branch main"),
        ({"expected_commit": "commit-b"}, "expected commit commit-b"),
    ],
)
def test_expected_scope_guards_reject_mismatched_evidence(scope_override: dict, expected_reason: str) -> None:
    """A qualification cannot silently pool evidence from a different scope."""
    overall, buckets = runner_stability.evaluate_stability(
        _records(),
        expected_buckets=_BUCKET,
        **scope_override,
    )

    assert overall == runner_stability.INCONCLUSIVE
    assert any(expected_reason in reason for reason in buckets[0].reasons)


def test_main_enforces_require_ready_exit_status(monkeypatch, tmp_path: Path) -> None:
    """The CLI fails closed only when requested and evidence is not stable."""
    records_path = tmp_path / "records.json"
    monkeypatch.setattr(runner_stability, "load_tasks", lambda _path: [])
    monkeypatch.setattr(runner_stability, "configured_scope", lambda _tasks, _gpu: (_BUCKET, {}))

    def run(records: list[dict], *, require_ready: bool) -> int:
        records_path.write_text(json.dumps(records), encoding="utf-8")
        argv = [
            "runner_stability.py",
            "--records",
            str(records_path),
            "--gpu_model",
            "l40s",
            "--expected_target_branch",
            "develop",
            "--expected_commit",
            "commit-a",
        ]
        if require_ready:
            argv.append("--require_ready")
        monkeypatch.setattr(sys, "argv", argv)
        return runner_stability.main()

    stable_records = _records()
    unstable_records = _records()
    unstable_records[0]["fps"] = 88.0

    assert run(stable_records, require_ready=False) == 0
    assert run(stable_records, require_ready=True) == 0
    assert run(unstable_records, require_ready=False) == 0
    assert run(unstable_records, require_ready=True) == 1


def test_report_states_scope_and_decision() -> None:
    """The reviewer-facing report explains what a passing result proves."""
    report, markdown = runner_stability.build_report(
        _records(),
        expected_buckets=_BUCKET,
    )

    assert report["overall_verdict"] == runner_stability.STABLE
    assert report["policy"]["minimum_distinct_runners"] == 3
    assert "Qualification policy fixed before measurement" in markdown
    assert "at least 3 distinct runner registrations" in markdown
    assert "greater than 10%" in markdown
    assert "runner-1" in markdown


def test_staging_workflow_fans_out_complete_independent_evidence() -> None:
    """One staging merge automatically gathers and qualifies five allocations."""
    repo_root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load(
        (repo_root / ".github/workflows/perf-smoke-runner-stability.yaml").read_text(encoding="utf-8")
    )
    seeder = yaml.safe_load(
        (repo_root / ".github/workflows/perf-smoke-seed-baselines.yaml").read_text(encoding="utf-8")
    )

    trigger = workflow.get("on", workflow.get(True))
    assert trigger["push"]["paths"] == [".github/workflows/perf-smoke-runner-stability.yaml"]
    assert "workflow_dispatch" not in trigger
    assert workflow["permissions"]["contents"] == "write"

    wait_job = workflow["jobs"]["wait_for_quiet_pool"]
    assert "needs" not in wait_job
    assert wait_job["permissions"]["contents"] == "read"
    wait_step = wait_job["steps"][0]
    assert "Performance Smoke Test" in wait_step["run"]
    assert "Perf Smoke — Publish CI Image" in wait_step["run"]
    assert "Perf Smoke — Auto Era Roll" in wait_step["run"]
    assert "QUIET_POLLS" in wait_step["run"]

    sample_job = workflow["jobs"]["stability_sample"]
    assert sample_job["needs"] == ["wait_for_quiet_pool"]
    assert sample_job["strategy"]["matrix"]["allocation"] == [1, 2, 3, 4, 5]
    assert sample_job["with"]["samples_per_commit"] == "3"
    assert sample_job["with"]["tasks"] == "__ALL_TASKS__"
    assert sample_job["with"]["dry_run"] is True
    assert "${{ matrix.allocation }}" in sample_job["with"]["artifact_suffix"]
    assert "${{ github.run_id }}" in sample_job["with"]["concurrency_group"]
    assert "${{ matrix.allocation }}" in sample_job["with"]["concurrency_group"]
    assert sample_job["secrets"] == {"NGC_API_KEY": "${{ secrets.NGC_API_KEY }}"}

    qualify_job = workflow["jobs"]["qualify"]
    qualify_steps = qualify_job["steps"]
    report_step = next(step for step in qualify_steps if step.get("name") == "Build qualification report")
    assert "--require_ready" in report_step["run"]
    assert "--minimum_distinct_runners 3" in report_step["run"]
    download_step = next(step for step in qualify_steps if step.get("name") == "Download runner evidence")
    assert download_step["continue-on-error"] is True
    assert any(step.get("name") == "Report artifact download failure" for step in qualify_steps)
    assert any(step.get("name") == "Fail on artifact download error" for step in qualify_steps)

    # Qualification only measures the pool. It must not publish baselines or chain
    # another gate run, so a bad verdict can never contaminate the rolling window.
    assert qualify_job["permissions"]["contents"] == "read"
    assert all("baseline" not in step.get("run", "").lower() for step in qualify_steps)
    assert not any(job.get("uses") == "./.github/workflows/perf-smoke-test.yaml" for job in workflow["jobs"].values())
    assert seeder["concurrency"]["group"] == "${{ inputs.concurrency_group || 'perf-smoke-seed' }}"
    seed_step = next(
        step for step in seeder["jobs"]["seed"]["steps"] if step.get("name") == "Seed baselines from commit history"
    )
    assert seed_step["env"]["PERF_SMOKE_RUNNER_NAME"] == "${{ runner.name }}"
    variance_step = next(
        step for step in seeder["jobs"]["seed"]["steps"] if step.get("name") == "Report run-to-run variance"
    )
    assert "inputs.run_label == ''" in variance_step["if"]
    assert (
        seed_step["env"]["SEED_DRY_RUN"]
        == "${{ inputs.dry_run == true && 'true' || github.event_name == 'push' && 'false' || "
        "inputs.dry_run == false && 'false' || 'true' }}"
    )
