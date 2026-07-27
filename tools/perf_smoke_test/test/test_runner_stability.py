# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free tests for runner-pool FPS stability qualification."""

from __future__ import annotations

import sys
from pathlib import Path

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


def test_missing_runner_allocation_is_inconclusive() -> None:
    """Partial evidence cannot accidentally qualify the pool."""
    records = [record for record in _records() if record["ci_run_label"] != "allocation-5"]

    overall, buckets = runner_stability.evaluate_stability(records, expected_buckets=_BUCKET)

    assert overall == runner_stability.INCONCLUSIVE
    assert any("expected 5 allocations, observed 4" in reason for reason in buckets[0].reasons)
    assert any("expected 15 samples, observed 12" in reason for reason in buckets[0].reasons)


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


def test_report_states_scope_and_decision() -> None:
    """The reviewer-facing report explains what a passing result proves."""
    report, markdown = runner_stability.build_report(
        _records(),
        expected_buckets=_BUCKET,
    )

    assert report["overall_verdict"] == runner_stability.STABLE
    assert "Qualification policy fixed before measurement" in markdown
    assert "10% or larger" in markdown
    assert "runner-1" in markdown


def test_staging_workflow_fans_out_complete_independent_evidence() -> None:
    """One staging merge automatically gathers and qualifies five allocations."""
    repo_root = Path(__file__).resolve().parents[3]
    workflow = (repo_root / ".github/workflows/perf-smoke-runner-stability.yaml").read_text()
    seeder = (repo_root / ".github/workflows/perf-smoke-seed-baselines.yaml").read_text()

    assert "allocation: [1, 2, 3, 4, 5]" in workflow
    assert 'samples_per_commit: "3"' in workflow
    assert 'tasks: "__ALL_TASKS__"' in workflow
    assert "dry_run: true" in workflow
    assert "--require_ready" in workflow
    assert "PERF_SMOKE_RUNNER_NAME: ${{ runner.name }}" in seeder
    assert "group: ${{ inputs.concurrency_group || 'perf-smoke-seed' }}" in seeder
