# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collection of the OVRTX renderer log for tests.

OVRTX logs natively, through a handle it opens on
:attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.log_file_path` and keeps for the lifetime of the process
(``keep_system_alive=True``). Pytest cannot capture that handle -- pointing it at ``/dev/stdout`` so pytest
would is what wrote blocks of NUL bytes into CI logs, because pytest rewinds and truncates the file it
captures into while the renderer's own offset keeps advancing. So the log stays a file, and this module is
the one place that reads it back:

* loaded as a pytest plugin by the repo-root ``conftest.py``, it claims the log for the test process and
  replays each test's share of it into pytest's capture, where a failing test's report picks it up;
* imported by ``tools/conftest.py``, it quotes a bounded tail in the crash, hang, or timeout report of a
  process that died before it could replay anything.

Both readers are needed: the replay attributes output to the test that produced it and stays out of the job
log for passing tests, and neither is possible from inside a process that segfaults, aborts, is OOM-killed,
or is SIGKILLed for hanging.

Both are also bounded, because what they report lands in the job log and in the JUnit XML, and both cover
the log alone. So when :data:`LOG_DIR_ENV_VAR` names a directory, everything a test left in the renderer's
own directory -- its log, whole, and any dump beside it -- is additionally saved there for CI to upload as
an artifact, which is what a diagnosis reads when the quoted tail is not enough.
"""

import contextlib
import os
import re
import shutil
import tempfile

import pytest

LOG_PATH = os.path.join(tempfile.gettempdir(), "ovrtx_renderer.log")
"""Where the renderer logs.

Mirrors the default of :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.log_file_path`, spelled out here so
loading this module does not import ``isaaclab_ov``.
"""

LOG_LIMIT_BYTES = 1024 * 1024
"""Maximum bytes of renderer log shown at once; earlier bytes are reported as omitted.

The log is verbose, and what is shown lands in the job log and in JUnit XML. An unbounded read would put a
multi-megabyte log in both -- the oversized-log problem the NUL blocks caused, with real text. This is
sized to hold a whole rendering test's log rather than its tail, since the tests this exists for log a few
hundred kilobytes each; the copy saved under :data:`LOG_DIR_ENV_VAR` covers the runs that outgrow it.
"""

LOG_DIR_ENV_VAR = "ISAACLAB_OVRTX_LOG_DIR"
"""Environment variable naming the directory each test's renderer output is saved under, whole.

Set per pytest invocation by ``tools/conftest.py``, to a directory under ``tests/`` that CI collects as a
job artifact. Unset by default, which leaves a local run writing nothing beyond the replay.
"""

_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
"""Everything a test node ID may hold that a file name should not, e.g. ``/``, ``::``, and ``[param]``."""


def log_size(path):
    """Return the size of ``path`` in bytes, or 0 when it does not exist yet."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def format_log_section(path, label, start=0, limit=LOG_LIMIT_BYTES):
    """Return the renderer log from ``start`` onwards as a report section.

    Args:
        path: Log file to read.
        label: What the section is reported against, e.g. a test name or a test file.
        start: Offset the reported range begins at. A log shorter than this was re-opened and rewritten, so
            the offset no longer describes its contents and the whole file is reported instead.
        limit: Maximum bytes to show, counted back from the end of the file.

    Returns:
        The section, or ``""`` when the range holds nothing -- the case for every test that never builds an
        OVRTX renderer, since nothing writes the file then.
    """
    size = log_size(path)
    if size < start:
        start = 0
    omitted = max(0, size - start - limit)
    try:
        with open(path, "rb") as handle:
            handle.seek(start + omitted)
            chunk = handle.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""
    if not chunk:
        return ""
    header = f"----- OVRTX renderer log: {label} -----"
    if omitted:
        header += f"\n[{omitted} earlier bytes omitted; last {limit} bytes follow]"
    return f"{header}\n{chunk}"


def _slugify(name):
    """Return ``name`` as a file name.

    Names here are test node IDs, which carry the path separators and parameter brackets a file name
    cannot.
    """
    return _UNSAFE_NAME_CHARS.sub("_", name).strip("_")


def _unused_dir(directory, stem):
    """Create and return ``<directory>/<stem>.<n>/``, numbered past any attempt already saved there.

    A retry reruns a test in a fresh process against the directory the first attempt wrote to, so
    overwriting would drop the attempt that failed in favour of the one that came after it.
    """
    attempt = 0
    candidate = os.path.join(directory, f"{stem}.{attempt}")

    while os.path.exists(candidate):
        attempt += 1
        candidate = os.path.join(directory, f"{stem}.{attempt}")
    os.makedirs(candidate)
    return candidate


def pytest_configure(config):
    """Claim the renderer log for this session, before any test imports the renderer.

    Whatever is at the path belongs to a session that has ended, so it is dropped rather than reasoned
    about: the byte offsets recorded per test below are only meaningful against this session's own output.
    The log is left in place at session end instead, since a crashed run is diagnosed by reading it after
    the process is gone; ``tools/conftest.py`` clears it before starting the next one.
    """
    with contextlib.suppress(OSError):
        os.remove(LOG_PATH)


@pytest.fixture(autouse=True)
def _echo_ovrtx_log(request):
    """Replay what the renderer logged during the test, and save what it wrote whole when asked.

    A no-op for tests that never build one, since nothing is written then. The log is written for the
    lifetime of the process, so only the range this test added is replayed, and pytest shows it with the
    test that failed rather than in the log of a passing run. The saved copy is the whole log plus any dump
    the renderer left beside it, under a directory of this test's own.

    The save runs first so that the artifact holds the log even if the replay cannot print it.
    """
    # The node ID names both, rather than the node name (``test_alpha[param]``, no test file). The saved
    # copy is read outside the report that says which file the test came from, so it needs the file, and
    # naming the replay the same way keeps one test's two records greppable by the same string.
    label = request.node.nodeid

    start = log_size(LOG_PATH)
    yield
    # The log exists only once the renderer has written to it, while the env var is set for every test in
    # the run, so a test that never built one has nothing to save.
    if (directory := os.environ.get(LOG_DIR_ENV_VAR)) and os.path.exists(LOG_PATH):
        destination = _unused_dir(directory, _slugify(label))
        # The renderer writes more than its log -- a crash leaves a dump beside it -- so everything in its
        # directory goes, bar the subdirectories a shared temp directory collects from other processes.
        for entry in os.scandir(os.path.dirname(LOG_PATH)):
            with contextlib.suppress(OSError):
                if entry.is_file():
                    shutil.copy(entry.path, destination)

    if section := format_log_section(LOG_PATH, label, start=start):
        print(f"\n{section}", end="")
