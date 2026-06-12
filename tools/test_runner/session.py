# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The test-run lifecycle: collect files, plan units, run them, report.

:class:`Session` is the orchestrator that ``conftest.pytest_sessionstart``
drives. It composes the stages: :class:`Collector` (which files), :class:`WorkQueue`
(how files are handed out across mgpu shards), the planner and unit runner from
the sibling modules (how each file runs), and :class:`Reporter` (merge + summary).
"""

from __future__ import annotations

import contextlib
import os
import re

import pytest
import test_settings  # tools/ module: QUARANTINED_TESTS / CUROBO_TESTS / TESTS_TO_SKIP
from junitparser import JUnitXml
from prettytable import PrettyTable

from test_runner.execution import ExecContext, UnitRunner, merge_statuses
from test_runner.planning import Planner, RunnerConfig


class WorkQueue:
    """Hand test files out across mgpu shards via an atomic-rename work queue.

    The queue is a directory of one file per pending test. A shard claims a test
    by renaming it ``queue/<slug>`` -> ``inflight/<shard>/<slug>`` (POSIX rename is
    atomic, so racing shards are kernel-serialized), then ``-> done/<shard>/<slug>``
    when it finishes. Anything left in ``inflight`` at job-end is a crashed claim.
    When inactive (no queue configured) the runner just iterates its own list.
    """

    def __init__(self, config: RunnerConfig):
        self._dir = config.queue_path
        self._shard = config.sim_device.replace(":", "-")

    @property
    def active(self) -> bool:
        """Whether a shared queue is configured (mgpu) versus plain iteration."""
        return bool(self._dir)

    @staticmethod
    def _slug(test_path: str) -> str:
        """Encode a path as a flat queue filename (``/`` -> ``__``)."""
        return test_path.replace("/", "__")

    def _claim(self) -> str | None:
        """Atomically claim one pending test, or return None when the queue is empty."""
        pending_dir = os.path.join(self._dir, "queue")
        inflight_dir = os.path.join(self._dir, "inflight", self._shard)
        os.makedirs(inflight_dir, exist_ok=True)
        try:
            entries = sorted(os.listdir(pending_dir))  # uncached: another shard may have just taken one
        except FileNotFoundError:
            return None
        for entry in entries:
            try:
                os.rename(os.path.join(pending_dir, entry), os.path.join(inflight_dir, entry))
            except FileNotFoundError:
                continue  # lost the race for this entry; try the next
            return entry.replace("__", "/")
        return None

    def mark_done(self, test_path: str) -> None:
        """Move a finished claim from ``inflight/<shard>/`` to ``done/<shard>/``."""
        entry = self._slug(test_path)
        done_dir = os.path.join(self._dir, "done", self._shard)
        os.makedirs(done_dir, exist_ok=True)
        # Suppress: already moved, or the runner died before marking — the reconciler catches that.
        with contextlib.suppress(FileNotFoundError):
            os.rename(os.path.join(self._dir, "inflight", self._shard, entry), os.path.join(done_dir, entry))

    def iter(self, files: list[str]):
        """Yield files to run: claimed from the queue when active, else ``files`` as-is."""
        if not self.active:
            yield from files
            return
        while (claimed := self._claim()) is not None:
            yield claimed


class Collector:
    """Discover the test files to run, applying filters and file-level sharding."""

    def __init__(self, config: RunnerConfig):
        self._config = config

    def _source_dirs(self) -> list[str]:
        """The roots walked for ``test_*.py`` files."""
        return [os.path.join(self._config.workspace_root, d) for d in ("scripts", "source")]

    def _selected(self, file_name: str, full_path: str) -> bool:
        """Whether one ``test_*.py`` passes the active mode/skip/pattern filters."""
        config = self._config
        if config.quarantined_only:
            return file_name in test_settings.QUARANTINED_TESTS
        if config.curobo_only:
            return file_name in test_settings.CUROBO_TESTS
        # An explicit include overrides the skip list (dedicated jobs run otherwise-skipped tests).
        if file_name in test_settings.TESTS_TO_SKIP and file_name not in config.include_files:
            return False
        if config.filter_pattern and config.filter_pattern not in full_path:
            return False
        if config.exclude_pattern and any(p.strip() in full_path for p in config.exclude_pattern.split(",")):
            return False
        if config.include_files and file_name not in config.include_files:
            return False
        return True

    def _sharded(self, files: list[str]) -> list[str]:
        """Select every Nth file for this file-shard; a no-op unless both shard env are set."""
        config = self._config
        if config.shard_index is None or config.shard_count is None or config.include_files:
            return files
        files = sorted(files)
        selected = [f for i, f in enumerate(files) if i % config.shard_count == config.shard_index]
        print(f"Shard {config.shard_index}/{config.shard_count}: selected {len(selected)} test files")
        return selected

    def _isaacsim_ci_only(self, files: list[str]) -> list[str]:
        """Keep only files containing an ``@pytest.mark.isaacsim_ci`` test."""
        kept = []
        for test_file in files:
            with open(test_file) as f:
                if "@pytest.mark.isaacsim_ci" in f.read():
                    kept.append(test_file)
        return kept

    def discover(self) -> list[str]:
        """Walk the source roots and return the filtered, sharded list of test files."""
        files = []
        for source_dir in self._source_dirs():
            if not os.path.exists(source_dir):
                print(f"Error: source directory not found at {source_dir}")
                pytest.exit("Source directory not found", returncode=1)
            for root, _, names in os.walk(source_dir):
                # install_ci has its own config/conftest and runs via its own action.
                if "install_ci" in root.replace("\\", "/").split("/"):
                    continue
                for name in names:
                    if name.startswith("test_") and name.endswith(".py"):
                        full_path = os.path.join(root, name)
                        if self._selected(name, full_path):
                            files.append(full_path)
        files = self._sharded(files)
        if self._config.isaacsim_ci:
            files = self._isaacsim_ci_only(files)
        return files


class Reporter:
    """Merge per-unit JUnit reports and render the run summary."""

    def __init__(self, config: RunnerConfig):
        self._config = config

    @property
    def _report_path(self) -> str:
        return os.path.join(self._config.report_dir, self._config.result_file)

    def write_empty(self) -> None:
        """Write an empty aggregate report so downstream CI steps find a valid file."""
        os.makedirs(self._config.report_dir, exist_ok=True)
        JUnitXml().write(self._report_path)
        print(f"Wrote empty report to {self._report_path}")

    @staticmethod
    def _counts(files: list[str], status: dict) -> dict:
        """Tally per-result-class file counts plus total wall/test time."""

        def n(result):
            return len([f for f in files if status[f]["result"] == result])

        return {
            "total": len(status),
            "passing": len([f for f in files if status[f]["result"].startswith("passed")]),
            "failing": n("FAILED"),
            "timeout": n("TIMEOUT"),
            "crashed": n("CRASHED"),
            "startup_hang": n("STARTUP_HANG"),
            "wall": sum(status[f]["wall_time"] for f in files),
            "test": sum(status[f]["time_elapsed"] for f in files),
        }

    def _per_file_table(self, files: list[str], status: dict) -> str:
        """One row per file: path, GPU, result, times, pass/total."""
        table = PrettyTable(field_names=["Test Path", "GPU", "Result", "Test (s)", "Wall (s)", "# Tests"])
        table.align["Test Path"] = "l"
        table.align["Test (s)"] = "r"
        table.align["Wall (s)"] = "r"
        for f in files:
            s = status[f]
            passed = s["tests"] - s["failures"] - s["errors"] - s["skipped"]
            table.add_row(
                [
                    f,
                    self._config.sim_device,
                    s["result"],
                    f"{s['time_elapsed']:0.2f}",
                    f"{s['wall_time']:0.2f}",
                    f"{passed}/{s['tests']}",
                ]
            )
        return table.get_string()

    def _per_test_time_table(self, full_report: JUnitXml) -> str:
        """Per-test run times, slowest first; device read from the test id, else the boot device."""
        table = PrettyTable(field_names=["Test", "Device", "Time (s)"])
        table.align["Test"] = "l"
        table.align["Time (s)"] = "r"
        rows = []
        for suite in full_report:
            for case in suite:
                name = f"{case.classname}::{case.name}" if case.classname else case.name
                device = self._config.sim_device
                bracket = re.search(r"\[(.*)\]", name)
                if bracket and (dev := re.search(r"cuda:\d+|\bcpu\b", bracket.group(1))):
                    device = dev.group(0)
                rows.append((name, device, float(case.time) if case.time is not None else 0.0))
        for name, device, elapsed in sorted(rows, key=lambda r: r[2], reverse=True):
            table.add_row([name, device, f"{elapsed:0.3f}"])
        return table.get_string()

    def _summary(self, files: list[str], status: dict, counts: dict, full_report: JUnitXml) -> str:
        """Assemble the printed summary: result tally, per-file table, per-test times."""
        pct = 100 if counts["total"] == 0 else counts["passing"] / counts["total"] * 100
        wall, test = counts["wall"], counts["test"]
        lines = [
            "\n\n===================\nTest Result Summary\n===================",
            f"Total: {counts['total']}",
            f"Passing: {counts['passing']}",
            f"Failing: {counts['failing']}",
            f"Crashed: {counts['crashed']}",
            f"Startup Hang: {counts['startup_hang']}",
            f"Timeout: {counts['timeout']}",
            f"Passing Percentage: {pct:.2f}%",
            f"Total Wall Time: {wall // 3600:.0f}h{wall // 60 % 60:.0f}m{wall % 60:.2f}s",
            f"Total Test Time: {test // 3600:.0f}h{test // 60 % 60:.0f}m{test % 60:.2f}s",
            "\n=======================\nPer File Result Summary\n=======================",
            self._per_file_table(files, status),
            "\n=================\nPer Test Run Time\n=================",
            self._per_test_time_table(full_report),
        ]
        return "\n".join(lines)

    def finalize(self, files: list[str], status: dict, xml_reports: list) -> int:
        """Merge reports, write the aggregate, print the summary, return the exit code.

        Returns 0 when every file passed, else 1.
        """
        print("~~~~~~~~~ Collecting final report...")
        # Merge the in-memory report objects (re-reading the files risks dropping
        # <failure> elements through the junitparser round-trip).
        full_report = JUnitXml()
        for report in xml_reports:
            full_report += report
        os.makedirs(self._config.report_dir, exist_ok=True)
        full_report.write(self._report_path)
        print("~~~~~~~~~~~~ Report written to", self._report_path)

        counts = self._counts(files, status)
        print(self._summary(files, status, counts, full_report))
        clean = counts["failing"] == counts["timeout"] == counts["crashed"] == counts["startup_hang"] == 0
        return 0 if clean else 1


class Session:
    """The per-file test-run lifecycle, driven by ``conftest.pytest_sessionstart``."""

    def __init__(self, config: RunnerConfig):
        self._config = config
        self._collector = Collector(config)
        self._queue = WorkQueue(config)
        self._planner = Planner(config.runtime_mask)
        self._runner = UnitRunner(config)
        self._reporter = Reporter(config)
        self._cold_cache_applied = False

    def _exec_context(self, file_name: str, source: str) -> ExecContext:
        """Per-file execution context, applying the one-time cold-cache timeout bump."""
        config = self._config
        timeout = config.per_file_timeouts.get(file_name, config.default_timeout)
        # The first camera-enabled test compiles shaders on a cold cache (~600 s);
        # give it (and only it) extra time so that isn't misread as a hang.
        is_cold = not self._cold_cache_applied and config.cold_cache_marker in source
        if is_cold:
            timeout += config.cold_cache_buffer
            self._cold_cache_applied = True
            print(f"⏱️  Adding {config.cold_cache_buffer}s cold-cache buffer (timeout now {timeout}s)")
        startup_deadline = min(timeout, config.startup_deadline + (config.cold_cache_buffer if is_cold else 0))
        env = os.environ.copy()
        env["PYTHONFAULTHANDLER"] = "1"
        return ExecContext(file_name=file_name, timeout=timeout, startup_deadline=startup_deadline, env=env)

    def _run_file(self, test_file: str, status: dict, failed: list, reports: list) -> None:
        """Plan ``test_file`` into units, run each, and fold their statuses into ``status``."""
        print(f"\n\n🚀 Running {test_file} independently...\n")
        try:
            with open(test_file) as fh:
                source = fh.read()
        except OSError:
            source = ""
        ctx = self._exec_context(os.path.basename(test_file), source)
        units = self._planner.plan(test_file, source)
        if len(units) > 1:
            print(f"⚙️  device_isolated — invoking {ctx.file_name} once per device ({len(units)} units)")
        merged = None
        for unit in units:
            report, unit_status, was_failure = self._runner.run(unit, ctx)
            if report is not None:
                reports.append(report)
            if was_failure and test_file not in failed:
                failed.append(test_file)
            merged = merge_statuses(merged, unit_status)
        status[test_file] = merged

    def run(self) -> int:
        """Run the lifecycle and return the process exit code.

        Returns:
            0 when nothing failed (including a clean "nothing to run"), else 1.
        """
        files = self._collector.discover()
        if not files:
            # A configured-but-empty scope is success; a bare empty run is an error.
            if self._config.quarantined_only or self._config.filter_pattern:
                print("No tests in scope — nothing to run.")
                self._reporter.write_empty()
                return 0
            print("No test files found in source directory")
            return 1

        print(f"Found {len(files)} test files after filtering")
        status, failed, reports = {}, [], []
        for test_file in self._queue.iter(files):
            self._run_file(test_file, status, failed, reports)
            if self._queue.active:
                self._queue.mark_done(test_file)
        print("~~~~~~~~~~~~ Finished running all tests")
        print("failed tests:", failed)

        # In work-queue mode this container ran only the files it claimed; report on those.
        reported_files = list(status) if self._queue.active else files
        return self._reporter.finalize(reported_files, status, reports)
