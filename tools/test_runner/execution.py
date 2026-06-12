# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""How to run one unit: build its command, drive the subprocess, classify the result.

:class:`UnitRunner` turns a :class:`~test_runner.planning.Unit` into a pytest
subprocess via :class:`~test_runner.process.ProcessCapture`, then resolves the
outcome (startup hang / timeout / crash / pass / fail) into a status dict, with
fresh-process retries for flaky files. The per-file dynamic inputs that vary
within a run live in :class:`ExecContext`; the static knobs come from the
:class:`~test_runner.planning.RunnerConfig`.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass

from test_runner.planning import RunnerConfig, Unit
from test_runner.process import JUnitReport, ProcessCapture

_RESULT_PRIORITY = {
    "STARTUP_HANG": 5,
    "CRASHED": 4,
    "TIMEOUT": 3,
    "FAILED": 2,
    "passed (shutdown hanged)": 1,
    "passed": 0,
}


@dataclass
class ExecContext:
    """Per-file inputs that vary within a run (static knobs are in RunnerConfig).

    Attributes:
        file_name: Basename of the file, used in JUnit naming and messages.
        timeout: Per-unit hard timeout [s] (already cold-cache-adjusted).
        startup_deadline: Per-unit startup-hang deadline [s].
        env: Base environment; :class:`UnitRunner` copies it and adds the unit's mask.
    """

    file_name: str
    timeout: int
    startup_deadline: int
    env: dict


def merge_statuses(prev: dict | None, new: dict) -> dict:
    """Combine per-unit status dicts into a per-file entry.

    Counters sum; ``result`` becomes the more severe of the two by
    :data:`_RESULT_PRIORITY`. Used by the session to fold a split file's units.
    """
    if prev is None:
        return new
    return {
        "errors": prev["errors"] + new["errors"],
        "failures": prev["failures"] + new["failures"],
        "skipped": prev["skipped"] + new["skipped"],
        "tests": prev["tests"] + new["tests"],
        "time_elapsed": prev["time_elapsed"] + new["time_elapsed"],
        "wall_time": prev["wall_time"] + new["wall_time"],
        "result": prev["result"]
        if _RESULT_PRIORITY.get(prev["result"], 0) >= _RESULT_PRIORITY.get(new["result"], 0)
        else new["result"],
    }


class UnitRunner:
    """Run one unit as a pytest subprocess and classify its outcome."""

    def __init__(self, config: RunnerConfig):
        self._config = config
        self._capture = ProcessCapture(config.shutdown_grace_period)

    @staticmethod
    def _status(result, *, errors=0, failures=0, skipped=0, tests=1, time_elapsed=0.0, wall_time=0.0) -> dict:
        """A per-unit status dict (the shape the reporter and merge_statuses expect)."""
        return {
            "errors": errors,
            "failures": failures,
            "skipped": skipped,
            "tests": tests,
            "result": result,
            "time_elapsed": time_elapsed,
            "wall_time": wall_time,
        }

    def _error_result(
        self,
        kind,
        label,
        suffix,
        unit,
        ctx,
        result,
        wall_time,
        cap,
        pre,
        *,
        msg,
        stdout,
        stderr,
        report_file,
        time_elapsed=0.0,
    ):
        """Print diagnostics, write a synthetic error report, and return its failure status."""
        diag = cap.diagnostics(pre)
        print(f"⚠️  {unit.file}{suffix}: {msg}")
        print(diag)
        details = f"{msg}\n\n=== SYSTEM DIAGNOSTICS ===\n{diag}\n\n"
        if stdout:
            details += "=== STDOUT (last 5000 chars) ===\n" + stdout.decode("utf-8", errors="replace")[-5000:] + "\n"
        if stderr:
            details += "=== STDERR (last 5000 chars) ===\n" + stderr.decode("utf-8", errors="replace")[-5000:] + "\n"
        report = JUnitReport.error(kind, label, msg, details)
        report.write(report_file)
        return report, self._status(result, errors=1, tests=1, time_elapsed=time_elapsed, wall_time=wall_time), True

    def _build_cmd(self, unit: Unit, ctx: ExecContext):
        """Build ``(cmd, env, report_file, label, report_suffix)`` for a unit.

        Device-variant selection is the unit's mask, set as ``ISAACLAB_TEST_DEVICES``
        for ``test_devices()`` — never ``-k``. A split unit (either half) or a
        cpu-less shard also injects the selector plugin to drop out-of-scope
        variants (including literal device-param ones the mask cannot narrow).
        """
        config = self._config
        # Split units of one file share a slug, so suffix the report with the mask;
        # a lone unit keeps the suffix-free name the mgpu aggregator unslugs to a path.
        report_suffix = f"-{unit.mask}" if unit.split else ""
        pass_file_label = f"{ctx.file_name}{report_suffix}"
        # Slug the full path (not the basename) so concurrent shards running
        # same-basename files don't collide on the shared workspace mount.
        report_slug = str(unit.file).replace("/", "__").replace("\\", "__")
        report_file = os.path.join(config.report_dir, f"test-reports-{report_slug}{report_suffix}.xml")

        env = dict(ctx.env)
        env["ISAACLAB_TEST_DEVICES"] = unit.mask
        select_by_device = unit.split or unit.mask[:1] == "0"
        if select_by_device:
            env["PYTHONPATH"] = config.tools_dir + os.pathsep + env.get("PYTHONPATH", "")

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-s",
            "-v",  # per-test names in the log: if a file hangs, the last name pinpoints the culprit
            "--no-header",
            f"--config-file={config.config_file}",
            f"--junitxml={report_file}",
            "--tb=short",
        ]
        if select_by_device:
            cmd += ["-p", "test_runner.selector"]
        if config.isaacsim_ci:
            cmd += ["-m", "isaacsim_ci"]
        cmd.append(str(unit.file))
        return cmd, env, report_file, pass_file_label, report_suffix

    def _retry_failed_in_fresh_process(
        self,
        *,
        unit,
        ctx,
        cmd,
        env,
        report_file,
        report,
        errors,
        failures,
        skipped,
        tests,
        time_elapsed,
        returncode,
        stdout,
        stderr,
        kill_reason,
        wall_time,
        pre,
    ):
        """Re-run a failed file in fresh processes (for files with stale-state flakiness)."""
        has_failures = errors > 0 or failures > 0
        attempts = 0
        max_retries = self._config.process_failure_retries.get(ctx.file_name, 0)
        while has_failures and attempts < max_retries:
            attempts += 1
            print(
                f"⚠️  {unit.file}: failed in subprocess"
                f" (attempt {attempts}/{max_retries + 1}), retrying in fresh process..."
            )
            with contextlib.suppress(FileNotFoundError):
                os.remove(report_file)
            returncode, stdout, stderr, kill_reason, wall_time, pre = self._capture.run(
                cmd, ctx.timeout, env, startup_deadline=ctx.startup_deadline, report_file=report_file
            )
            if not os.path.exists(report_file):
                break
            try:
                report, errors, failures, skipped, tests, time_elapsed = JUnitReport.read(report_file, ctx.file_name)
                has_failures = errors > 0 or failures > 0
            except Exception as e:
                print(f"Error reading retry test report {report_file}: {e}")
                report, errors, failures, skipped, tests, time_elapsed = report, 1, 0, 0, 0, 0.0
                has_failures = True
                break
        return report, errors, failures, skipped, tests, time_elapsed, returncode, kill_reason, wall_time, has_failures

    def run(self, unit: Unit, ctx: ExecContext) -> tuple:
        """Run ``unit`` and return ``(xml_report_or_None, status_dict, was_failure)``."""
        cap = self._capture
        cmd, env, report_file, label, suffix = self._build_cmd(unit, ctx)

        # -- launch, retrying startup hangs / hard timeouts --
        returncode, stdout, stderr, kill_reason, wall_time, pre = -1, b"", b"", "", 0.0, ""
        startup_attempts = timeout_attempts = 0
        while True:
            with contextlib.suppress(FileNotFoundError):
                os.remove(report_file)
            returncode, stdout, stderr, kill_reason, wall_time, pre = cap.run(
                cmd, ctx.timeout, env, startup_deadline=ctx.startup_deadline, report_file=report_file
            )
            has_report = os.path.exists(report_file)
            if kill_reason == "startup_hang" and startup_attempts < self._config.startup_hang_retries:
                startup_attempts += 1
                print(
                    f"⚠️  {unit.file}{suffix}: startup hang after {ctx.startup_deadline}s"
                    f" (attempt {startup_attempts}/{self._config.startup_hang_retries + 1}), retrying..."
                )
                if stderr:
                    print("=== STDERR (last 5000 chars) ===")
                    print(stderr.decode("utf-8", errors="replace")[-5000:])
                print(cap.diagnostics(pre))
                continue
            if kill_reason == "timeout" and not has_report and timeout_attempts < self._config.timeout_retries:
                timeout_attempts += 1
                print(
                    f"⚠️  {unit.file}{suffix}: timeout after {ctx.timeout}s"
                    f" (attempt {timeout_attempts}/{self._config.timeout_retries + 1}), retrying..."
                )
                if stdout:
                    print("=== STDOUT (last 5000 chars) ===")
                    print(stdout.decode("utf-8", errors="replace")[-5000:])
                print(cap.diagnostics(pre))
                continue
            break

        # -- classify the outcome --
        has_report = os.path.exists(report_file)
        if kill_reason == "startup_hang":
            return self._error_result(
                "startup_hang",
                label,
                suffix,
                unit,
                ctx,
                "STARTUP_HANG",
                wall_time,
                cap,
                pre,
                msg=f"Startup hang after {ctx.startup_deadline}s (retried {self._config.startup_hang_retries} time(s))",
                stdout=stdout,
                stderr=stderr,
                report_file=report_file,
            )
        if kill_reason == "timeout" and not has_report:
            return self._error_result(
                "timeout",
                label,
                suffix,
                unit,
                ctx,
                "TIMEOUT",
                wall_time,
                cap,
                pre,
                msg=f"Timeout after {ctx.timeout} seconds (retried {timeout_attempts} time(s))",
                stdout=stdout,
                stderr=stderr,
                report_file=report_file,
                time_elapsed=ctx.timeout,
            )
        if not has_report:
            reason = (
                cap.signal_description(-returncode)
                if returncode < 0
                else f"Process exited with code {returncode} but produced no report"
            )
            return self._error_result(
                "crash",
                label,
                suffix,
                unit,
                ctx,
                "CRASHED",
                wall_time,
                cap,
                "",
                msg=reason,
                stdout=stdout,
                stderr=stderr,
                report_file=report_file,
            )

        if kill_reason in ("shutdown_hang", "timeout"):
            print(f"⚠️  {unit.file}{suffix}: shutdown hanged (killed after {wall_time:.0f}s, test completed)")
        try:
            report, errors, failures, skipped, tests, time_elapsed = JUnitReport.read(report_file, label)
        except Exception as e:
            print(f"Error reading test report {report_file}: {e}")
            return None, self._status("FAILED", errors=1, tests=0, wall_time=wall_time), True

        report, errors, failures, skipped, tests, time_elapsed, returncode, kill_reason, wall_time, has_failures = (
            self._retry_failed_in_fresh_process(
                unit=unit,
                ctx=ctx,
                cmd=cmd,
                env=env,
                report_file=report_file,
                report=report,
                errors=errors,
                failures=failures,
                skipped=skipped,
                tests=tests,
                time_elapsed=time_elapsed,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                kill_reason=kill_reason,
                wall_time=wall_time,
                pre=pre,
            )
        )
        shutdown_hanged = kill_reason in ("shutdown_hang", "timeout") and not has_failures
        was_failure = has_failures or (returncode != 0 and not shutdown_hanged)
        result = "passed (shutdown hanged)" if shutdown_hanged else ("FAILED" if has_failures else "passed")
        return (
            report,
            self._status(
                result,
                errors=errors,
                failures=failures,
                skipped=skipped,
                tests=tests,
                time_elapsed=time_elapsed,
                wall_time=wall_time,
            ),
            was_failure,
        )
