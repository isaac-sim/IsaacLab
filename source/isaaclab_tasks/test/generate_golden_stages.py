# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate and validate golden USD stages for rendering correctness tests.

The first pytest run writes missing baselines under ``golden_stages/`` and fails
intentionally (one "Golden stage not found" failure per missing baseline). The second
run verifies the generated goldens against simulation output.

Isaac Sim occasionally deadlocks while tearing down the RTX/Hydra renderer at interpreter
exit (the ``removeHydraEngineImpl - found not in use hydra engine`` warning is the last
thing printed). pytest has already finished and written its JUnit report by then, so this
script never trusts the child's exit code: it recovers the real outcome from the report
and hard-kills the (possibly shutdown-hung) process tree so the pipeline can proceed.

This driver is Linux/POSIX-only: it relies on process-group semantics
(:func:`os.setsid` via ``start_new_session``, :func:`os.getpgid`, :func:`os.killpg`, and
``SIGKILL``) to reap the Kit process tree, none of which exist on Windows. Run it on the
CI platform, not a local Windows checkout.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Safety cap on a single pytest run before its JUnit report appears. Sized for the slowest
# valid rendering run; exceeding it means the child is stuck before finishing collection.
_PYTEST_MAX_RUNTIME_SECONDS = 3600

# Grace period granted for a clean interpreter exit after pytest has written its report.
# A healthy shutdown takes ~15s; anything longer is the Kit teardown deadlock, so we kill.
_SHUTDOWN_GRACE_SECONDS = 90

# Interval between JUnit-report completeness polls while pytest is running.
_POLL_INTERVAL_SECONDS = 2.0

# Substring emitted by ``maybe_save_stage`` when it bootstraps a missing baseline. These
# failures are expected on pass 1 and must not be treated as real regressions.
_BOOTSTRAP_FAILURE_MARKER = "A new baseline was written"


@dataclass
class PytestOutcome:
    """Result of a golden-stage pytest run, recovered from the JUnit report."""

    report_available: bool
    passed: int = 0
    bootstrapped: list[str] = field(default_factory=list)
    real_failures: list[tuple[str, str]] = field(default_factory=list)
    timed_out: bool = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _golden_stage_pytest_node_ids(repo_root: Path) -> tuple[str, ...]:
    """Collect pytest node IDs from the shared rendering test utilities."""
    test_utils_dir = repo_root / "source/isaaclab_tasks/test"
    sys.path.insert(0, str(test_utils_dir))
    from rendering_test_utils import golden_stage_pytest_node_ids

    return golden_stage_pytest_node_ids()


def _check_gpu() -> None:
    """Fail fast when the NVIDIA driver is unavailable."""
    try:
        subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.error("NVIDIA driver/GPU is required to generate golden stages.")
        raise SystemExit(1) from None


def _junit_is_complete(junit_path: Path) -> bool:
    """Return whether ``junit_path`` holds a fully written, parseable report.

    pytest writes the report atomically once the session finishes (before the atexit
    SimulationApp shutdown), so a successful parse means the run is done reporting.
    """
    if not junit_path.exists() or junit_path.stat().st_size == 0:
        return False
    try:
        ET.parse(junit_path)
    except ET.ParseError:
        return False
    return True


def _kill_process_tree(process: subprocess.Popen) -> None:
    """SIGKILL the child's whole process group (Kit spawns helpers that outlive the parent)."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        logger.warning("Process tree did not terminate after SIGKILL.")


def _parse_junit(junit_path: Path) -> PytestOutcome:
    """Turn a JUnit report into a :class:`PytestOutcome`, classifying bootstrap failures."""
    if not _junit_is_complete(junit_path):
        return PytestOutcome(report_available=False)

    outcome = PytestOutcome(report_available=True)
    root = ET.parse(junit_path).getroot()
    for testcase in root.iter("testcase"):
        name = testcase.get("name", "<unknown>")
        problem = testcase.find("failure")
        if problem is None:
            problem = testcase.find("error")
        if problem is None:
            if testcase.find("skipped") is None:
                outcome.passed += 1
            continue

        detail = " ".join(part for part in (problem.get("message", ""), problem.text or "") if part)
        if _BOOTSTRAP_FAILURE_MARKER in detail:
            outcome.bootstrapped.append(name)
        else:
            outcome.real_failures.append((name, detail.strip()))
    return outcome


def _run_pytest(repo_root: Path, node_ids: tuple[str, ...], junit_path: Path) -> PytestOutcome:
    """Run the stage golden tests through Isaac Lab's Python wrapper.

    Runs every case (no ``-x``) so a single pass bootstraps every missing baseline. The
    child is started in its own session so a shutdown deadlock can be killed as a group,
    and the outcome is read back from ``junit_path`` rather than the child's exit code.
    """
    env = os.environ.copy()
    env.pop("CONDA_PREFIX", None)
    env.pop("VIRTUAL_ENV", None)

    command = [
        str(repo_root / "isaaclab.sh"),
        "-p",
        "-m",
        "pytest",
        *node_ids,
        "-v",
        "-s",
        f"--junit-xml={junit_path}",
    ]

    process = subprocess.Popen(command, cwd=repo_root, env=env, start_new_session=True)

    deadline = time.monotonic() + _PYTEST_MAX_RUNTIME_SECONDS
    report_ready = False
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        if _junit_is_complete(junit_path):
            report_ready = True
            break
        time.sleep(_POLL_INTERVAL_SECONDS)

    if process.poll() is None:
        if not report_ready:
            logger.warning(
                "pytest produced no report within %ds; killing the stuck process tree.",
                _PYTEST_MAX_RUNTIME_SECONDS,
            )
            _kill_process_tree(process)
            outcome = _parse_junit(junit_path)
            outcome.timed_out = True
            return outcome

        # pytest is done and has reported; only the Isaac Sim shutdown remains. Allow a
        # short grace period for a clean exit, then kill the (deadlocked) renderer teardown.
        try:
            process.wait(timeout=_SHUTDOWN_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning("Isaac Sim did not shut down within %ds; killing it.", _SHUTDOWN_GRACE_SECONDS)
            _kill_process_tree(process)

    return _parse_junit(junit_path)


def _format_failures(failures: list[tuple[str, str]]) -> str:
    """Render a compact, readable list of failing cases for logging."""
    lines = []
    for name, detail in failures:
        first_line = detail.splitlines()[0] if detail else "(no detail)"
        lines.append(f"  - {name}: {first_line}")
    return "\n".join(lines)


def main() -> int:
    """Generate missing golden stages, then verify them."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    repo_root = _repo_root()
    node_ids = _golden_stage_pytest_node_ids(repo_root)
    if not node_ids:
        logger.error("No golden-stage rendering tests are configured.")
        return 1

    _check_gpu()

    logger.info("Running %d golden-stage rendering test(s).", len(node_ids))

    with tempfile.TemporaryDirectory(prefix="golden_stage_junit_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        logger.info("Pass 1: write missing golden stages (bootstrap failures expected)...")
        first = _run_pytest(repo_root, node_ids, tmp_path / "pass1.xml")

        if not first.report_available:
            logger.error("Pass 1 produced no pytest report; the run crashed before tests finished.")
            return 1
        if first.real_failures:
            logger.error("Pass 1 had unexpected failures:\n%s", _format_failures(first.real_failures))
            return 1
        if not first.bootstrapped:
            if first.passed == 0:
                logger.error(
                    "Pass 1 had no passing, failing, or bootstrapping tests."
                    " All tests were likely skipped (e.g., Isaac Sim failed to initialise)."
                )
                return 1
            logger.info("Golden stages already exist and match. Nothing to generate.")
            return 0

        logger.info("Bootstrapped %d golden stage(s). Verifying...", len(first.bootstrapped))

        logger.info("Pass 2: verify generated golden stages...")
        second = _run_pytest(repo_root, node_ids, tmp_path / "pass2.xml")

    if not second.report_available:
        logger.error("Pass 2 produced no pytest report; unable to verify generated golden stages.")
        return 1
    if second.bootstrapped:
        logger.error(
            "Pass 2 still had to bootstrap goldens, which points to a path mismatch:\n%s",
            "\n".join(f"  - {name}" for name in second.bootstrapped),
        )
        return 1
    if second.real_failures:
        logger.error("Pass 2 failed to verify golden stages:\n%s", _format_failures(second.real_failures))
        return 1

    logger.info("Golden stages generated under source/isaaclab_tasks/test/golden_stages/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
