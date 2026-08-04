# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Group test files that can share one Kit app into a single pytest invocation.

A test file that boots Kit at module scope pays Kit startup on its own, and the runner
gives every file its own subprocess, so a directory of 23 such files boots Kit 23 times.
Files migrated to :func:`~isaaclab.test.launch.launch_kit` share the app when they land in
one process, which turns those 23 boots into one.

Only files carrying the same launch profile may be grouped. ``kit`` and ``kit_cameras``
cannot share a process in either direction: cameras cannot be enabled after startup, and a
camera-enabled app is not a substitute for a plain one because some tests assert that
offscreen rendering is off. Anything whose behaviour depends on having a process to itself
stays on the per-file path.

This module is deliberately free of ``os`` and ``subprocess`` calls: the grouping and the
report demultiplexing are pure functions over paths and strings, so they can be exercised on
any platform, unlike the POSIX-only process machinery in ``tools/conftest.py``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

BATCH_ENV_VAR = "ISAACLAB_TEST_BATCH_KIT"
"""Environment variable that opts a run into batching. Unset keeps the per-file path."""

BATCH_SIZE_ENV_VAR = "ISAACLAB_TEST_BATCH_SIZE"
"""Environment variable overriding :data:`DEFAULT_BATCH_SIZE`."""

DEFAULT_BATCH_SIZE = 12
"""Files per batch.

Bounded so that one crash cannot cost a whole lane, and so accumulated GPU memory in a long
shared process does not become its own failure mode.
"""

BATCH_TIMEOUT_CUTOFF = 2000
"""Files whose own timeout reaches this stay unbatched.

A batch's timeout is the sum of its members', so one file hanging consumes the whole budget.
The long-running files are also the ones where Kit startup is a rounding error, so excluding
them removes most of the risk and almost none of the benefit.
"""

# `kit` must not match `kit_cameras` or `kit_solo`.
_MARK_KIT = re.compile(r"pytest\.mark\.kit(?![\w])")
_MARK_CAMERAS = re.compile(r"pytest\.mark\.kit_cameras\b")
_MARK_SOLO = re.compile(r"pytest\.mark\.kit_solo\b")


@dataclass
class Batch:
    """One pytest invocation covering one or more test files.

    Attributes:
        profile: Launch profile shared by every member, or None for an unbatched file.
        files: Test files to hand to pytest, in invocation order.
        index: Position among the batches of this profile. Part of :attr:`label`, which
            becomes a JUnit report filename, so two batches of the same profile and size
            cannot write to the same path.
    """

    profile: str | None
    files: list[str] = field(default_factory=list)
    index: int = 0

    @property
    def is_batched(self) -> bool:
        """Whether this covers more than one file."""
        return len(self.files) > 1

    @property
    def label(self) -> str:
        """Short identifier used in logs and JUnit report filenames."""
        return f"batch-{self.profile}-{self.index}-{len(self.files)}files" if self.is_batched else self.files[0]


def batching_enabled(env: dict | None = None) -> bool:
    """Whether the run opted into batching via :data:`BATCH_ENV_VAR`."""
    env = os.environ if env is None else env
    return env.get(BATCH_ENV_VAR, "").strip().lower() in ("1", "true", "yes")


def batch_size(env: dict | None = None) -> int:
    """Resolve the per-batch file cap, falling back to :data:`DEFAULT_BATCH_SIZE`."""
    env = os.environ if env is None else env
    raw = env.get(BATCH_SIZE_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_BATCH_SIZE
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_BATCH_SIZE
    return value if value > 0 else DEFAULT_BATCH_SIZE


def file_profile(source: str) -> str | None:
    """Return the launch profile a test file declares, or None if it cannot be batched.

    Args:
        source: The test file's text. Markers are matched against the source rather than by
            importing the module, because importing a Kit-dependent module boots Kit.

    Returns:
        ``"kit_cameras"``, ``"kit"``, or None when the file is unmarked or opts out.
    """
    if _MARK_SOLO.search(source):
        return None
    if _MARK_CAMERAS.search(source):
        return "kit_cameras"
    if _MARK_KIT.search(source):
        return "kit"
    return None


def group_test_files(
    test_files: list[str],
    sources: dict[str, str],
    *,
    unbatchable: set[str] | None = None,
    max_size: int = DEFAULT_BATCH_SIZE,
) -> list[Batch]:
    """Partition ``test_files`` into batches, preserving the given order.

    Files that cannot be grouped -- unmarked, ``kit_solo``, or listed in ``unbatchable`` --
    each become a batch of one, which is exactly the current per-file behaviour.

    Args:
        test_files: Test file paths, in the order the runner would execute them.
        sources: Map from a path in ``test_files`` to that file's text. A path missing from
            the map is treated as unbatchable rather than assumed safe.
        unbatchable: Paths to keep on the per-file path regardless of their markers.
        max_size: Maximum files per batch.

    Returns:
        Batches covering every input file exactly once, in input order.
    """
    unbatchable = unbatchable or set()
    batches: list[Batch] = []
    pending: dict[str, Batch] = {}
    counts: dict[str, int] = {}

    def flush(profile: str) -> None:
        if profile in pending:
            batches.append(pending.pop(profile))

    for path in test_files:
        source = sources.get(path)
        profile = None if source is None or path in unbatchable else file_profile(source)

        if profile is None:
            batches.append(Batch(profile=None, files=[path]))
            continue

        current = pending.get(profile)
        if current is None:
            current = Batch(profile=profile, index=counts.get(profile, 0))
            counts[profile] = current.index + 1
            pending[profile] = current
        current.files.append(path)
        if len(current.files) >= max_size:
            flush(profile)

    # Emit any partially filled batches in a stable order.
    for profile in sorted(pending):
        batches.append(pending[profile])
    return batches


def _testcase_files(report, batch_files: list[str]) -> dict[str, list]:
    """Map each batch member to the testcases attributed to it in a JUnit report.

    JUnit ``classname`` encodes the dotted module path, so a file is matched by its stem.
    Where two members share a stem the match is ambiguous and those testcases are dropped
    from the per-file split rather than assigned to the wrong file.
    """
    stems: dict[str, list[str]] = {}
    for path in batch_files:
        stem = os.path.splitext(os.path.basename(path))[0]
        stems.setdefault(stem, []).append(path)

    per_file: dict[str, list] = {path: [] for path in batch_files}
    for suite in report:
        for case in suite:
            classname = getattr(case, "classname", "") or ""
            name = getattr(case, "name", "") or ""
            for part in reversed(classname.split(".")):
                owners = stems.get(part)
                if owners and len(owners) == 1:
                    per_file[owners[0]].append(case)
                    break
            else:
                # Fall back to the test name for parametrized ids that carry the module.
                for stem, owners in stems.items():
                    if len(owners) == 1 and stem in name:
                        per_file[owners[0]].append(case)
                        break
    return per_file


def split_batch_status(
    report,
    batch_files: list[str],
    *,
    wall_time: float,
    batch_result: str,
) -> dict[str, dict]:
    """Attribute a batch's JUnit report back to its individual files.

    The summary table, the failed-file list, and the per-file JUnit artifact are all keyed by
    file, so a batch has to be taken apart again before its results are reported.

    A file with no testcases in the report never ran -- the shared process died before
    reaching it -- and is marked with ``batch_result`` so the caller can re-run it.

    Args:
        report: Parsed JUnit XML for the whole batch.
        batch_files: The batch's members.
        wall_time: Wall seconds for the whole batch, shared out across members that ran.
        batch_result: Result to record for members that produced no testcases.

    Returns:
        Map from file path to a status dict of the same shape the per-file path produces.
    """
    per_file = _testcase_files(report, batch_files)
    ran = [path for path, cases in per_file.items() if cases]
    share = wall_time / len(ran) if ran else 0.0

    statuses: dict[str, dict] = {}
    for path in batch_files:
        cases = per_file[path]
        if not cases:
            statuses[path] = {
                "errors": 1,
                "failures": 0,
                "skipped": 0,
                "tests": 1,
                "result": batch_result,
                "time_elapsed": 0.0,
                "wall_time": 0.0,
            }
            continue

        errors = failures = skipped = 0
        elapsed = 0.0
        for case in cases:
            elapsed += float(getattr(case, "time", 0.0) or 0.0)
            result = getattr(case, "result", None) or []
            kinds = {type(entry).__name__ for entry in result}
            if "Error" in kinds:
                errors += 1
            elif "Failure" in kinds:
                failures += 1
            elif "Skipped" in kinds:
                skipped += 1

        statuses[path] = {
            "errors": errors,
            "failures": failures,
            "skipped": skipped,
            "tests": len(cases),
            "result": "FAILED" if (errors or failures) else "passed",
            "time_elapsed": elapsed,
            "wall_time": share,
        }
    return statuses
