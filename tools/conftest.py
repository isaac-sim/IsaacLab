# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import os
import re

import pytest
from junitparser import JUnitXml
from prettytable import PrettyTable

# Local imports
import test_settings as test_settings  # isort: skip
from _device_exec import COLD_CACHE_BUFFER, STARTUP_DEADLINE, _UnitContext, merge_unit_status, run_unit  # isort: skip
from _device_plan import is_isolated as _is_isolated, plan_units  # isort: skip


def pytest_ignore_collect(collection_path, config):
    # Skip collection and run each test script individually
    return True


def _slugify_test_path(test_path):
    """Encode a test path as a flat queue entry name.

    The queue uses one file per pending test. Slashes are not legal inside a
    filename, so we encode the relative path by replacing ``/`` with ``__``.
    The decoder is :func:`_unslugify_queue_entry`.
    """
    return test_path.replace("/", "__")


def _unslugify_queue_entry(entry_name):
    """Reverse of :func:`_slugify_test_path`."""
    return entry_name.replace("__", "/")


def _claim_queued_file(queue_dir):
    """Atomically claim one pending test from the work-queue directory.

    The queue is a directory of files (one per pending test); the shard claims
    one by renaming it from ``queue/`` into its private ``inflight/cuda-N/``.
    POSIX rename is atomic on the same filesystem, so two shards racing on the
    same source file are serialized by the kernel: exactly one rename succeeds,
    the other gets ``FileNotFoundError`` and tries the next entry.

    On success, the test's queue entry is now sitting in ``inflight/cuda-N/``;
    the caller is expected to move it to ``done/cuda-N/`` after the per-test
    pytest invocation exits with a clean result, leaving anything still in
    ``inflight/`` at job-end as recoverable evidence of a crashed test.

    Args:
        queue_dir: Path to the shared work-queue root. Must contain a
            ``queue/`` subdir (pending entries) and an ``inflight/<shard>/``
            subdir for this shard (claim destination).

    Returns:
        The decoded test path for the claimed file, or ``None`` when the
        queue is empty.
    """
    shard = os.environ.get("ISAACLAB_SIM_DEVICE", "cuda").replace(":", "-")
    pending_dir = os.path.join(queue_dir, "queue")
    inflight_dir = os.path.join(queue_dir, "inflight", shard)
    os.makedirs(inflight_dir, exist_ok=True)

    # Listdir is intentionally not cached: another shard may have just removed
    # an entry we'd otherwise try. We pay one listdir per claim attempt; with
    # N≤20 entries this is microseconds.
    try:
        entries = sorted(os.listdir(pending_dir))
    except FileNotFoundError:
        return None

    for entry in entries:
        src = os.path.join(pending_dir, entry)
        dst = os.path.join(inflight_dir, entry)
        try:
            os.rename(src, dst)
        except FileNotFoundError:
            # Lost the race for this entry; another shard claimed it first.
            # Continue to the next entry in our (potentially stale) listing.
            continue
        except OSError:
            # Any other rename failure (e.g. permission) is a hard error.
            raise
        return _unslugify_queue_entry(entry)

    return None


def _mark_queued_file_done(queue_dir, test_path):
    """Move a successfully-completed claim from ``inflight/cuda-N/`` to ``done/cuda-N/``.

    Called by the test runner after a per-file pytest invocation exits cleanly.
    The inflight residual is what the post-run reconciler uses to detect
    crashed shards: anything still in ``inflight/`` at job-end is an orphan.
    """
    shard = os.environ.get("ISAACLAB_SIM_DEVICE", "cuda").replace(":", "-")
    entry = _slugify_test_path(test_path)
    src = os.path.join(queue_dir, "inflight", shard, entry)
    dst_dir = os.path.join(queue_dir, "done", shard)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, entry)
    # Suppress: already moved (idempotent) or the runner crashed before we
    # could mark done — the reconciler catches the second case.
    with contextlib.suppress(FileNotFoundError):
        os.rename(src, dst)


def _queued_files(queue_dir):
    """Yield files claimed from the shared work queue until it is empty."""
    while True:
        claimed = _claim_queued_file(queue_dir)
        if claimed is None:
            return
        yield claimed


def run_individual_tests(test_files, workspace_root, isaacsim_ci):
    """Run each test file separately, ensuring one finishes before starting the next.

    When ``ISAACLAB_TEST_QUEUE`` names a shared work-queue directory, files are
    claimed from it (work-stealing across sibling shard containers) instead of
    iterating ``test_files``; each file still runs once, on this container's GPU.

    The planner turns each file into one or more units (a device_isolated file
    splits into one unit per device when the runtime spans more than one); the
    executor runs each unit with the unit's mask set as ``ISAACLAB_TEST_DEVICES``.
    """
    failed_tests = []
    test_status = {}
    xml_reports = []
    cold_cache_applied = False

    queue_path = os.environ.get("ISAACLAB_TEST_QUEUE", "")
    file_source = _queued_files(queue_path) if queue_path else test_files

    # The runner's device set. Unset (single-GPU CI) is cpu + cuda:0, matching
    # test_devices()'s default; an mgpu shard sets a single-device mask. The
    # planner splits a device_isolated file only when this spans more than one.
    runtime_mask = os.environ.get("ISAACLAB_TEST_DEVICES") or "110"

    for test_file in file_source:
        print(f"\n\n\U0001f680 Running {test_file} independently...\n")
        file_name = os.path.basename(test_file)
        env = os.environ.copy()
        env["PYTHONFAULTHANDLER"] = "1"

        timeout = test_settings.PER_TEST_TIMEOUTS.get(file_name, test_settings.DEFAULT_TIMEOUT)

        # Read the test file once for cold-cache and device-isolation detection.
        try:
            with open(test_file) as fh:
                test_content = fh.read()
        except OSError:
            test_content = ""

        # The first camera-enabled test in a fresh container compiles shaders
        # (~600 s).  Give it extra time so that doesn't look like a test timeout.
        is_cold_cache_test = not cold_cache_applied and "enable_cameras=True" in test_content
        if is_cold_cache_test:
            timeout += COLD_CACHE_BUFFER
            cold_cache_applied = True
            print(f"\u23f1\ufe0f  Adding {COLD_CACHE_BUFFER}s cold-cache buffer (timeout now {timeout}s)")

        extra = COLD_CACHE_BUFFER if is_cold_cache_test else 0
        startup_deadline = min(timeout, STARTUP_DEADLINE + extra)

        ctx = _UnitContext(
            test_file=test_file,
            file_name=file_name,
            workspace_root=workspace_root,
            isaacsim_ci=isaacsim_ci,
            timeout=timeout,
            startup_deadline=startup_deadline,
            env=env,
        )

        units = plan_units([test_file], runtime_mask, is_isolated=lambda f: _is_isolated(f, source=test_content))
        if len(units) > 1:
            print(f"\u2699\ufe0f  device_isolated \u2014 invoking {file_name} once per device ({len(units)} units)")

        merged_status: dict | None = None
        for unit in units:
            report, status, was_failure = run_unit(ctx, unit)
            if report is not None:
                xml_reports.append(report)
            if was_failure and test_file not in failed_tests:
                failed_tests.append(test_file)
            merged_status = merge_unit_status(merged_status, status)

        assert merged_status is not None  # the unit list is never empty
        test_status[test_file] = merged_status

        # In work-queue mode, move the claim from inflight/<shard>/ to done/<shard>/
        # so the post-run reconciler can tell "ran to completion" from "claimed but
        # crashed mid-test". A claim left in inflight at job-end is a silent drop.
        if queue_path:
            _mark_queued_file_done(queue_path, test_file)

    print("~~~~~~~~~~~~ Finished running all tests")

    return failed_tests, test_status, xml_reports


def _collect_test_files(
    source_dirs,
    filter_pattern,
    exclude_pattern,
    include_files,
    quarantined_only,
    curobo_only,
):
    """Collect test files from source directories, applying all active filters."""
    test_files = []
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"Error: source directory not found at {source_dir}")
            pytest.exit("Source directory not found", returncode=1)

        for root, _, files in os.walk(source_dir):
            # source/isaaclab/test/install_ci/ has its own pytest config and conftest.
            # It is run via .github/actions/install-ci-run, never via this collector,
            # so skip the whole subtree to keep install_ci tests out of build.yaml jobs.
            if "install_ci" in root.replace("\\", "/").split("/"):
                continue

            for file in files:
                if not (file.startswith("test_") and file.endswith(".py")):
                    continue

                # Mode-exclusive filters (each bypasses TESTS_TO_SKIP)
                if quarantined_only:
                    if file not in test_settings.QUARANTINED_TESTS:
                        continue
                elif curobo_only:
                    if file not in test_settings.CUROBO_TESTS:
                        continue
                else:
                    # An explicit include_files entry overrides TESTS_TO_SKIP, allowing
                    # dedicated jobs (e.g. test-environments-training) to run tests that
                    # are otherwise excluded from general CI runs.
                    if file in test_settings.TESTS_TO_SKIP and file not in include_files:
                        print(f"Skipping {file} as it's in the skip list")
                        continue

                full_path = os.path.join(root, file)

                if filter_pattern and filter_pattern not in full_path:
                    print(f"Skipping {full_path} (does not match include pattern: {filter_pattern})")
                    continue
                if exclude_pattern and any(p.strip() in full_path for p in exclude_pattern.split(",")):
                    print(f"Skipping {full_path} (matches exclude pattern: {exclude_pattern})")
                    continue
                if include_files and file not in include_files:
                    print(f"Skipping {full_path} (not in include files list)")
                    continue

                test_files.append(full_path)

    # Apply file-level sharding: sort deterministically, then select every Nth file.
    # Skip when include_files is set — in that case the test's own conftest handles
    # sharding at the test-item level (e.g. parametrized test cases).
    shard_index = os.environ.get("TEST_SHARD_INDEX", "")
    shard_count = os.environ.get("TEST_SHARD_COUNT", "")
    if shard_index and shard_count and not include_files:
        shard_index = int(shard_index)
        shard_count = int(shard_count)
        test_files.sort()
        test_files = [f for i, f in enumerate(test_files) if i % shard_count == shard_index]
        print(f"Shard {shard_index}/{shard_count}: selected {len(test_files)} test files")

    return test_files


def _write_empty_report():
    """Write an empty JUnit XML report so downstream CI steps find a valid file."""
    os.makedirs("tests", exist_ok=True)
    result_file = os.environ.get("TEST_RESULT_FILE", "full_report.xml")
    report = JUnitXml()
    report.write(f"tests/{result_file}")
    print(f"Wrote empty report to tests/{result_file}")


def pytest_sessionstart(session):
    """Intercept pytest startup to execute tests in the correct order."""
    # Get the workspace root directory (one level up from tools)
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_dirs = [
        os.path.join(workspace_root, "scripts"),
        os.path.join(workspace_root, "source"),
    ]

    # Get filter pattern from environment variable or command line
    filter_pattern = os.environ.get("TEST_FILTER_PATTERN", "")
    exclude_pattern = os.environ.get("TEST_EXCLUDE_PATTERN", "")
    include_files_str = os.environ.get("TEST_INCLUDE_FILES", "")
    quarantined_only = os.environ.get("TEST_QUARANTINED_ONLY", "false") == "true"
    curobo_only = os.environ.get("TEST_CUROBO_ONLY", "false") == "true"

    isaacsim_ci = os.environ.get("ISAACSIM_CI_SHORT", "false") == "true"

    # Parse include files list (comma-separated paths)
    include_files = set()
    if include_files_str:
        for f in include_files_str.split(","):
            f = f.strip()
            if f:
                include_files.add(os.path.basename(f))

    # Also try to get from pytest config
    if hasattr(session.config, "option") and hasattr(session.config.option, "filter_pattern"):
        filter_pattern = filter_pattern or getattr(session.config.option, "filter_pattern", "")
    if hasattr(session.config, "option") and hasattr(session.config.option, "exclude_pattern"):
        exclude_pattern = exclude_pattern or getattr(session.config.option, "exclude_pattern", "")

    print("=" * 50)
    print("CONFTEST.PY DEBUG INFO")
    print("=" * 50)
    print(f"Filter pattern: '{filter_pattern}'")
    print(f"Exclude pattern: '{exclude_pattern}'")
    print(f"Include files: {include_files if include_files else 'none'}")
    print(f"Quarantined-only mode: {quarantined_only}")
    print(f"Curobo-only mode: {curobo_only}")
    print(f"TEST_FILTER_PATTERN env var: '{os.environ.get('TEST_FILTER_PATTERN', 'NOT_SET')}'")
    print(f"TEST_EXCLUDE_PATTERN env var: '{os.environ.get('TEST_EXCLUDE_PATTERN', 'NOT_SET')}'")
    print(f"TEST_INCLUDE_FILES env var: '{os.environ.get('TEST_INCLUDE_FILES', 'NOT_SET')}'")
    print(f"TEST_QUARANTINED_ONLY env var: '{os.environ.get('TEST_QUARANTINED_ONLY', 'NOT_SET')}'")
    print(f"TEST_CUROBO_ONLY env var: '{os.environ.get('TEST_CUROBO_ONLY', 'NOT_SET')}'")
    print("=" * 50)

    # Get all test files in the source directories
    test_files = _collect_test_files(
        source_dirs,
        filter_pattern,
        exclude_pattern,
        include_files,
        quarantined_only,
        curobo_only,
    )

    if isaacsim_ci:
        new_test_files = []
        for test_file in test_files:
            with open(test_file) as f:
                if "@pytest.mark.isaacsim_ci" in f.read():
                    new_test_files.append(test_file)
        test_files = new_test_files

    if not test_files:
        if quarantined_only:
            print("No quarantined tests configured — nothing to run.")
            _write_empty_report()
            pytest.exit("No quarantined tests configured", returncode=0)
        if filter_pattern:
            print(f"No test files found matching filter pattern '{filter_pattern}' — nothing to run.")
            _write_empty_report()
            pytest.exit("No test files found for filter", returncode=0)
        print("No test files found in source directory")
        pytest.exit("No test files found", returncode=1)

    print(f"Found {len(test_files)} test files after filtering:")
    for test_file in test_files:
        print(f"  - {test_file}")

    # Run all tests individually
    failed_tests, test_status, xml_reports = run_individual_tests(test_files, workspace_root, isaacsim_ci)

    # In work-queue mode this container ran only the files it claimed; report on those.
    if os.environ.get("ISAACLAB_TEST_QUEUE"):
        test_files = list(test_status)

    print("failed tests:", failed_tests)

    # Collect reports
    print("~~~~~~~~~ Collecting final report...")

    # Merge in-memory report objects collected during the test run.  Reading the
    # on-disk files again risks losing <failure> elements if the junitparser
    # read/write round-trip does not preserve them faithfully.
    full_report = JUnitXml()
    for xml_report in xml_reports:
        print(xml_report)
        full_report += xml_report
    print("~~~~~~~~~~~~ Writing final report...")
    # write content to full report
    result_file = os.environ.get("TEST_RESULT_FILE", "full_report.xml")
    full_report_path = f"tests/{result_file}"
    # Ensure the directory exists even when this shard claimed zero files
    # from the work queue (per-test JUnit XMLs are what normally create
    # ``tests/``; with no tests run there is nothing to create it).
    os.makedirs("tests", exist_ok=True)
    print(f"Using result file: {result_file}")
    full_report.write(full_report_path)
    print("~~~~~~~~~~~~ Report written to", full_report_path)

    # print test status in a nice table
    # Calculate the number and percentage of passing tests
    num_tests = len(test_status)
    num_passing = len([p for p in test_files if test_status[p]["result"].startswith("passed")])
    num_failing = len([p for p in test_files if test_status[p]["result"] == "FAILED"])
    num_timeout = len([p for p in test_files if test_status[p]["result"] == "TIMEOUT"])
    num_crashed = len([p for p in test_files if test_status[p]["result"] == "CRASHED"])
    num_startup_hang = len([p for p in test_files if test_status[p]["result"] == "STARTUP_HANG"])

    if num_tests == 0:
        passing_percentage = 100
    else:
        passing_percentage = num_passing / num_tests * 100

    # Print summaries of test results
    summary_str = "\n\n"
    summary_str += "===================\n"
    summary_str += "Test Result Summary\n"
    summary_str += "===================\n"

    summary_str += f"Total: {num_tests}\n"
    summary_str += f"Passing: {num_passing}\n"
    summary_str += f"Failing: {num_failing}\n"
    summary_str += f"Crashed: {num_crashed}\n"
    summary_str += f"Startup Hang: {num_startup_hang}\n"
    summary_str += f"Timeout: {num_timeout}\n"
    summary_str += f"Passing Percentage: {passing_percentage:.2f}%\n"

    total_wall = sum(test_status[test_path]["wall_time"] for test_path in test_files)
    total_test = sum(test_status[test_path]["time_elapsed"] for test_path in test_files)

    summary_str += f"Total Wall Time: {total_wall // 3600:.0f}h{total_wall // 60 % 60:.0f}m{total_wall % 60:.2f}s\n"
    summary_str += f"Total Test Time: {total_test // 3600:.0f}h{total_test // 60 % 60:.0f}m{total_test % 60:.2f}s"

    # GPU this run used (the shard's boot device); ``cuda:0`` when unset.
    run_device = os.environ.get("ISAACLAB_SIM_DEVICE") or "cuda:0"

    summary_str += "\n\n=======================\n"
    summary_str += "Per File Result Summary\n"
    summary_str += "=======================\n"

    per_file_result_table = PrettyTable(field_names=["Test Path", "GPU", "Result", "Test (s)", "Wall (s)", "# Tests"])
    per_file_result_table.align["Test Path"] = "l"
    per_file_result_table.align["Test (s)"] = "r"
    per_file_result_table.align["Wall (s)"] = "r"
    for test_path in test_files:
        num_tests_passed = (
            test_status[test_path]["tests"]
            - test_status[test_path]["failures"]
            - test_status[test_path]["errors"]
            - test_status[test_path]["skipped"]
        )
        per_file_result_table.add_row(
            [
                test_path,
                run_device,
                test_status[test_path]["result"],
                f"{test_status[test_path]['time_elapsed']:0.2f}",
                f"{test_status[test_path]['wall_time']:0.2f}",
                f"{num_tests_passed}/{test_status[test_path]['tests']}",
            ]
        )

    summary_str += per_file_result_table.get_string()

    # Per-test run times, slowest first, from the merged JUnit report. The
    # device is read from the test id params (e.g. ``...[size0-cuda:1]``),
    # falling back to the run's boot device.
    summary_str += "\n\n=================\n"
    summary_str += "Per Test Run Time\n"
    summary_str += "=================\n"

    per_test_time_table = PrettyTable(field_names=["Test", "Device", "Time (s)"])
    per_test_time_table.align["Test"] = "l"
    per_test_time_table.align["Time (s)"] = "r"
    test_times = []
    for suite in full_report:
        for case in suite:
            full_name = f"{case.classname}::{case.name}" if case.classname else case.name
            device = run_device
            bracket = re.search(r"\[(.*)\]", full_name)
            if bracket:
                dev_match = re.search(r"cuda:\d+|\bcpu\b", bracket.group(1))
                if dev_match:
                    device = dev_match.group(0)
            elapsed = float(case.time) if case.time is not None else 0.0
            test_times.append((full_name, device, elapsed))
    for full_name, device, elapsed in sorted(test_times, key=lambda row: row[2], reverse=True):
        per_test_time_table.add_row([full_name, device, f"{elapsed:0.3f}"])

    summary_str += per_test_time_table.get_string()

    # Print summary to console and log file
    print(summary_str)

    # Exit pytest after custom execution to prevent normal pytest from overwriting our report
    pytest.exit(
        "Custom test execution completed",
        returncode=0 if (num_failing == 0 and num_timeout == 0 and num_crashed == 0 and num_startup_hang == 0) else 1,
    )
