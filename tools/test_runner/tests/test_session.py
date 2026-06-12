# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for test_runner.session: WorkQueue, Collector filters, Reporter, cold-cache."""

from __future__ import annotations

from test_runner.planning import RunnerConfig
from test_runner.session import Collector, Reporter, Session, WorkQueue


def _status(result, *, tests=1, failures=0, errors=0, skipped=0, wall=1.0, elapsed=0.5):
    return {
        "result": result,
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "wall_time": wall,
        "time_elapsed": elapsed,
    }


# ---- WorkQueue ----


def test_workqueue_inactive_iterates_the_given_files():
    wq = WorkQueue(RunnerConfig(workspace_root="/ws"))  # no queue_path
    assert wq.active is False
    assert list(wq.iter(["a.py", "b.py"])) == ["a.py", "b.py"]


def test_workqueue_claim_then_mark_done_roundtrip(tmp_path):
    config = RunnerConfig(workspace_root="/ws", queue_path=str(tmp_path), sim_device="cuda:1")
    (tmp_path / "queue").mkdir()
    (tmp_path / "queue" / "src__pkg__test_x.py").write_text("")
    wq = WorkQueue(config)
    assert wq.active is True
    claimed = list(wq.iter([]))
    assert claimed == ["src/pkg/test_x.py"]  # decoded path
    assert (tmp_path / "inflight" / "cuda-1" / "src__pkg__test_x.py").exists()
    wq.mark_done("src/pkg/test_x.py")
    assert (tmp_path / "done" / "cuda-1" / "src__pkg__test_x.py").exists()
    assert not (tmp_path / "inflight" / "cuda-1" / "src__pkg__test_x.py").exists()


# ---- Collector filters ----


def test_collector_selects_by_filter_and_exclude_pattern():
    c = Collector(RunnerConfig(workspace_root="/ws", filter_pattern="physx"))
    assert c._selected("test_zzz_synthetic.py", "/a/physx/test_zzz_synthetic.py") is True
    assert c._selected("test_zzz_synthetic.py", "/a/newton/test_zzz_synthetic.py") is False

    c = Collector(RunnerConfig(workspace_root="/ws", exclude_pattern="newton"))
    assert c._selected("test_zzz_synthetic.py", "/a/newton/test_zzz_synthetic.py") is False


def test_collector_respects_include_files():
    c = Collector(RunnerConfig(workspace_root="/ws", include_files=frozenset({"test_keep.py"})))
    assert c._selected("test_keep.py", "/a/test_keep.py") is True
    assert c._selected("test_drop.py", "/a/test_drop.py") is False


# ---- Reporter counts ----


def test_reporter_counts_tally_results_and_time():
    files = ["a.py", "b.py", "c.py"]
    status = {"a.py": _status("passed"), "b.py": _status("FAILED"), "c.py": _status("passed (shutdown hanged)")}
    counts = Reporter._counts(files, status)
    assert counts["total"] == 3
    assert counts["passing"] == 2  # both "passed*" results
    assert counts["failing"] == 1
    assert counts["wall"] == 3.0


# ---- Session cold-cache timeout policy ----


def test_cold_cache_buffer_applies_once_to_the_first_camera_file():
    config = RunnerConfig(workspace_root="/ws", default_timeout=600, cold_cache_buffer=700, startup_deadline=120)
    session = Session(config)
    first = session._exec_context("test_cam.py", "enable_cameras=True")
    assert first.timeout == 1300  # 600 + 700
    assert first.startup_deadline == 820  # min(1300, 120 + 700)
    second = session._exec_context("test_cam2.py", "enable_cameras=True")
    assert second.timeout == 600  # buffer already spent
    assert second.startup_deadline == 120


def test_per_file_timeout_override():
    config = RunnerConfig(workspace_root="/ws", default_timeout=600, per_file_timeouts={"test_slow.py": 1200})
    ctx = Session(config)._exec_context("test_slow.py", "")
    assert ctx.timeout == 1200
