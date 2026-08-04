# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The advisory/blocking exit contract, and diagnostics after a failed benchmark.

``gate_config.blocking`` decides whether the aggregate job's exit code carries
the verdict. In advisory mode it never does: every verdict, including
HARD_FAILURE, exits 0, and the verdict travels through the ``perf-smoke-test``
commit status, the sticky PR comment and the job summary instead. That is what
keeps an unrelated pull request from getting a red check because a registry
outage killed the benchmark.

What advisory mode must *not* do is go quiet. These tests pin both halves: exit
0, and a verdict that still says HARD_FAILURE out loud, with a summary a
developer can read.

Gate malfunctions are the exception and stay fatal in both modes -- if aggregate
found no bench artifacts at all it produced no verdict, and its owners need to
see that.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

import aggregate  # noqa: E402
from contracts import BenchResult  # noqa: E402

_RUNTIME_HASH = "runtime-a"


def _bench_result(*, fps: float | None, info_present: bool, exit_code: int = 0, failure_phase: str | None = None):
    launch_config = {
        "task_id": "Isaac-Cartpole-Direct",
        "backend": "physx",
        "backend_key": "physx",
        "physics_backend": "physx",
        "render_backend": None,
        "gpu_model": "l40s",
        "launch_config_hash": "launch-a",
        "benchmark_contract_hash": "bench-a",
        "baseline_epoch": 1,
    }
    return BenchResult(
        task_id="Isaac-Cartpole-Direct",
        backend="physx",
        physics_backend="physx",
        render_backend=None,
        backend_key="physx",
        preset="default",
        was_retried=False,
        stdout_tail="",
        perf_smoke_test_info_present=info_present,
        raw_fps_mean=fps,
        exit_code=exit_code,
        failure_phase=failure_phase,
        runtime_contract_hash=_RUNTIME_HASH,
        runtime_resources={"gpu_name": "NVIDIA L40S"},
        provenance={"software": {"isaaclab": "3.0.0"}},
        launch_config=launch_config,
        launch_config_hash="launch-a",
        benchmark_contract_hash="bench-a",
        baseline_epoch=1,
    )


def _run(tmp_path: Path, monkeypatch, *, blocking: bool, bench_result=None, write_artifact: bool = True):
    """Drive aggregate.main() in flat-file mode; return (exit_code, outputs, summary)."""
    artifacts_dir = tmp_path / "artifacts"
    baselines_dir = tmp_path / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_file = tmp_path / "verdict_summary.md"
    output_file = tmp_path / "gh_output.txt"

    gate_config = tmp_path / "gate_config.json"
    gate_config.write_text(json.dumps({"blocking": blocking}), encoding="utf-8")

    # These cases exercise one bucket, so declare a one-bucket matrix. Otherwise
    # the completeness check correctly reports 1-of-9 and masks what is under
    # test here. The check itself is covered directly further down.
    monkeypatch.setattr(aggregate, "load_tasks", lambda *a, **k: [object()])

    if write_artifact:
        task_dir = artifacts_dir / "bench-Isaac-Cartpole-Direct-physx"
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "perf_smoke_test_result.json").write_text(json.dumps(bench_result.to_dict()))

    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate.py",
            "--artifacts_dir",
            str(artifacts_dir),
            "--gpu_model",
            "L40S",
            "--baselines_dir",
            str(baselines_dir),
            "--allow_baseline_update",
            "false",
            "--gate_config",
            str(gate_config),
            "--summary_file",
            str(summary_file),
        ],
    )
    exit_code = aggregate.main()

    outputs: dict[str, str] = {}
    if output_file.exists():
        for line in output_file.read_text().splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                outputs[key] = value
    summary = summary_file.read_text(encoding="utf-8") if summary_file.exists() else ""
    return exit_code, outputs, summary


# --- advisory mode: never fails the job, always reports --------------------


@pytest.mark.parametrize(
    "bench_kwargs",
    [
        pytest.param({"fps": None, "info_present": False}, id="no-measurement"),
        pytest.param({"fps": 0.0, "info_present": True}, id="zero-fps"),
        pytest.param(
            {"fps": 100.0, "info_present": True, "exit_code": 1, "failure_phase": "import"},
            id="crashed-after-writing-a-bundle",
        ),
    ],
)
def test_advisory_mode_exits_zero_on_hard_failure(tmp_path, monkeypatch, bench_kwargs) -> None:
    """A crashed benchmark must not fail the check while the gate is advisory."""
    exit_code, outputs, _ = _run(tmp_path, monkeypatch, blocking=False, bench_result=_bench_result(**bench_kwargs))

    assert exit_code == 0, "advisory mode must not fail the job on a benchmark failure"
    assert outputs.get("overall_verdict") == "HARD_FAILURE"


def test_advisory_mode_still_reports_the_failure(tmp_path, monkeypatch) -> None:
    """Advisory must mean 'does not fail the PR', never 'says nothing'."""
    _, outputs, summary = _run(
        tmp_path, monkeypatch, blocking=False, bench_result=_bench_result(fps=None, info_present=False)
    )

    # The commit status is the signal that survives a green job.
    assert outputs.get("status_state") == "failure"
    assert outputs.get("blocking") == "false"
    assert "advisory" in outputs.get("status_description", "").lower()

    # And the summary a developer actually reads must name the failure.
    assert summary, "a summary must be written even when the benchmark failed"
    assert "HARD FAILURE" in summary or "HARD_FAILURE" in summary
    assert "Isaac-Cartpole-Direct" in summary


def test_advisory_mode_is_green_and_quiet_on_a_clean_run(tmp_path, monkeypatch) -> None:
    """A healthy run reports success, so the red status stays meaningful."""
    exit_code, outputs, summary = _run(
        tmp_path, monkeypatch, blocking=False, bench_result=_bench_result(fps=100.0, info_present=True)
    )

    assert exit_code == 0
    assert outputs.get("status_state") == "success"
    # No baseline yet, so the honest verdict is WARN (not a silent PASS).
    assert outputs.get("overall_verdict") == "WARN"
    assert summary


# --- blocking mode: the exit code carries the verdict ----------------------


def test_blocking_mode_fails_on_hard_failure(tmp_path, monkeypatch) -> None:
    """Flipping blocking:true is what makes a crashed benchmark fail the job."""
    exit_code, outputs, _ = _run(
        tmp_path, monkeypatch, blocking=True, bench_result=_bench_result(fps=None, info_present=False)
    )

    assert exit_code == 2
    assert outputs.get("overall_verdict") == "HARD_FAILURE"
    assert outputs.get("blocking") == "true"


def test_blocking_mode_passes_a_healthy_run(tmp_path, monkeypatch) -> None:
    exit_code, outputs, _ = _run(
        tmp_path, monkeypatch, blocking=True, bench_result=_bench_result(fps=100.0, info_present=True)
    )

    assert exit_code == 0
    assert outputs.get("status_state") == "success"


# --- gate malfunctions stay fatal in both modes ----------------------------


@pytest.mark.parametrize("blocking", [False, True], ids=["advisory", "blocking"])
def test_missing_artifacts_fail_in_both_modes(tmp_path, monkeypatch, blocking: bool) -> None:
    """No bench artifacts at all means no verdict was produced -- that is a gate fault."""
    exit_code, _outputs, _summary = _run(tmp_path, monkeypatch, blocking=blocking, write_artifact=False)

    assert exit_code == 1, "a gate that produced no verdict must fail regardless of advisory mode"


# --- the verdict must come from the rows, never from the flags alone -------
#
# has_hard_failure is cleared for crashes excused as CI-image skew, and main()
# only bails when there are ZERO artifacts. Deriving the reported verdict from
# those two booleans therefore produced an affirmative "no meaningful
# performance regression detected" for a run in which nine buckets crashed and
# nothing was measured -- the exact silent-green this gate exists to prevent.

from gate_types import OracleVerdict  # noqa: E402


def _rows(*verdicts):
    return [(SimpleNamespace(verdict=v), None) for v in verdicts]


def test_skew_excused_crashes_do_not_report_pass() -> None:
    """Nine crashed-but-excused buckets must not read as 'no regression'."""
    out = aggregate._verdict_outputs(
        _rows(*([OracleVerdict.HARD_FAILURE] * 9)),
        has_block=False,
        has_hard_failure=False,  # cleared by detect_dependency_skew
        blocking=False,
        expected_buckets=9,
    )

    assert out["overall_verdict"] == "HARD_FAILURE"
    assert out["status_state"] == "failure"
    assert "no meaningful" not in out["status_description"]
    assert "stale" in out["status_description"]


def test_missing_buckets_are_not_graded_on_the_survivors() -> None:
    """One passing bucket out of nine is not a pass."""
    out = aggregate._verdict_outputs(
        _rows(OracleVerdict.PASS), has_block=False, has_hard_failure=False, blocking=False, expected_buckets=9
    )

    assert out["overall_verdict"] == "HARD_FAILURE"
    assert out["status_state"] == "failure"
    assert "only 1 of 9" in out["status_description"]


def test_a_complete_clean_run_still_passes() -> None:
    """The guard must not swallow a genuine pass."""
    out = aggregate._verdict_outputs(
        _rows(*([OracleVerdict.PASS] * 9)),
        has_block=False,
        has_hard_failure=False,
        blocking=False,
        expected_buckets=9,
    )

    assert out["overall_verdict"] == "PASS"
    assert out["status_state"] == "success"
    assert "9 buckets" in out["status_description"]


def test_expected_bucket_count_is_optional() -> None:
    """An unreadable tasks.json disables the completeness check, never fails a run."""
    out = aggregate._verdict_outputs(
        _rows(*([OracleVerdict.PASS] * 3)),
        has_block=False,
        has_hard_failure=False,
        blocking=False,
        expected_buckets=None,
    )

    assert out["overall_verdict"] == "PASS"
