# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free tests for the baseline seeding variance report."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

import analyze_seed_variance  # noqa: E402


def _record(
    fps: float,
    sample_index: int,
    runtime_hash: str = "runtime-a",
    ci_run_id: str = "run-a",
    runner_name: str = "runner-a",
) -> dict:
    return {
        "gpu_model": "l40s",
        "task_id": "task-a",
        "backend": "newton",
        "target_branch": "develop",
        "commit_sha": "commit-a",
        "launch_config_hash": "launch-a",
        "benchmark_contract_hash": "benchmark-a",
        "runtime_contract_hash": runtime_hash,
        "baseline_epoch": 1,
        "sample_index": sample_index,
        "ci_run_id": ci_run_id,
        "ci_runner_name": runner_name,
        "fps": fps,
    }


def test_analyze_records_reports_robust_and_full_spread_metrics() -> None:
    """The report exposes both robust noise and outlier-sensitive spread."""
    records = [
        _record(99.0, 4),
        _record(102.0, 1),
        _record(100.0, 0),
        _record(101.0, 3),
        _record(98.0, 2),
    ]

    buckets, skipped = analyze_seed_variance.analyze_records(records)

    assert skipped == 0
    assert len(buckets) == 1
    bucket = buckets[0]
    assert bucket.ci_run_count == 1
    assert bucket.runner_count == 1
    assert bucket.sample_count == 5
    assert bucket.median_fps == 100.0
    assert bucket.mean_fps == 100.0
    assert bucket.mad_pct == 1.0
    assert bucket.cv_pct == pytest.approx(1.58113883)
    assert bucket.range_pct == 4.0
    assert bucket.first_to_last_pct == -1.0


def test_analyze_records_keeps_runtime_fingerprints_separate() -> None:
    """Environment changes cannot be misreported as runner variance."""
    records = [
        _record(100.0, 0, runtime_hash="runtime-a"),
        _record(101.0, 1, runtime_hash="runtime-a"),
        _record(80.0, 0, runtime_hash="runtime-b"),
        _record(81.0, 1, runtime_hash="runtime-b"),
    ]

    buckets, _ = analyze_seed_variance.analyze_records(records)

    assert len(buckets) == 2
    assert {bucket.runtime_contract_hash for bucket in buckets} == {"runtime-a", "runtime-b"}
    assert [bucket.sample_count for bucket in buckets] == [2, 2]


def test_analyze_records_counts_identical_fps_repetitions() -> None:
    """Equal measurements remain independent samples with zero observed variance."""
    records = [_record(100.0, sample_index) for sample_index in range(5)]

    buckets, _ = analyze_seed_variance.analyze_records(records)

    assert buckets[0].sample_count == 5
    assert buckets[0].mad_pct == 0.0
    assert buckets[0].cv_pct == 0.0
    assert buckets[0].range_pct == 0.0


def test_analyze_records_combines_independent_ci_runs() -> None:
    """Repeated workflow jobs contribute to one same-fingerprint variance bucket."""
    records = [
        _record(100.0, 0, ci_run_id="run-a"),
        _record(101.0, 1, ci_run_id="run-a"),
        _record(99.0, 0, ci_run_id="run-b", runner_name="runner-b"),
        _record(102.0, 1, ci_run_id="run-b", runner_name="runner-b"),
    ]

    buckets, _ = analyze_seed_variance.analyze_records(records)

    assert len(buckets) == 1
    assert buckets[0].ci_run_count == 2
    assert buckets[0].runner_count == 2
    assert buckets[0].sample_count == 4


def test_build_report_skips_invalid_records_and_explains_metrics() -> None:
    """Malformed records are reported without preventing useful output."""
    report, markdown = analyze_seed_variance.build_report(
        [
            _record(100.0, 0),
            {"task_id": "task-a", "backend": "newton", "fps": float("nan")},
            {"task_id": "task-a", "fps": 100.0},
        ]
    )

    assert report["valid_sample_count"] == 1
    assert report["skipped_sample_count"] == 2
    assert "This report is advisory and does not fail CI." in markdown
    assert "**MAD / median**" in markdown
    assert "Skipped 2 malformed or non-finite record(s)." in markdown
