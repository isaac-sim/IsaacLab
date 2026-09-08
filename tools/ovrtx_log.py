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
own directory -- its own share of the log, uncapped, and any dump beside it -- is additionally saved there
for CI to upload as an artifact, which is what a diagnosis reads when the quoted tail is not enough. The
test that a crash, hang, or timeout killed never reaches the fixture that saves, so ``tools/conftest.py``
saves what that process left behind on its behalf -- the one test in the run whose output would otherwise
be missing from the artifact is the one the artifact exists for.
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
"""Environment variable naming the directory each test's renderer output is saved under, uncapped.

Set per pytest invocation by ``tools/conftest.py``, to a directory under ``tests/`` that CI collects as a
job artifact. Unset by default, which leaves a local run writing nothing beyond the replay.
"""

_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
"""Everything a test name may hold that a file name should not, e.g. the ``[param]`` of a parametrization."""


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

    Names here are test names, and a parametrized one carries whatever its parameters are spelled with.
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


def save_output(directory, label, start=0):
    """Save what the renderer wrote under ``directory``, in a directory named after ``label``.

    Args:
        directory: Where to save. Created if it does not exist yet.
        label: What the saved directory is named after, i.e. the test the output belongs to.
        start: Offset the saved log begins at, so that a per-test copy holds the range that test added
            rather than everything the process logged before it. A log shorter than this was re-opened
            and rewritten, so the offset no longer describes its contents and the whole file is saved,
            as :func:`format_log_section` does with the same argument.

    Returns:
        The directory written, or ``None`` when the log holds nothing past ``start``. That covers both a
        test that never builds a renderer, which writes no log at all, and one that runs after a test
        which did, since the log it finds already there is not its to save.
    """
    size = log_size(LOG_PATH)
    if size < start:
        start = 0
    if size <= start:
        return None

    log_name = os.path.basename(LOG_PATH)
    destination = _unused_dir(directory, _slugify(label))
    with open(LOG_PATH, "rb") as source, open(os.path.join(destination, log_name), "wb") as saved:
        source.seek(start)
        shutil.copyfileobj(source, saved)

    # The renderer writes more than its log -- a crash leaves a dump beside it -- so the rest of its
    # directory goes too, bar the subdirectories a shared temp directory collects from other processes.
    for entry in os.scandir(os.path.dirname(LOG_PATH)):
        with contextlib.suppress(OSError):
            if entry.is_file() and entry.name != log_name:
                shutil.copy(entry.path, destination)
    return destination


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
    """Replay what the renderer logged during the test, and save what it wrote when asked.

    A no-op for tests that never build one, since nothing is written then. The log is written for the
    lifetime of the process, so only the range this test added is replayed and saved: pytest shows it with
    the test that failed rather than in the log of a passing run, and the artifact holds one copy of each
    test's own output rather than a growing copy of everything logged before it.

    The save runs first so that the artifact holds the log even if the replay cannot print it, and is
    suppressed so that the reverse cannot happen either: this runs in the teardown of every test that
    rendered, so a full or unwritable artifact directory would otherwise turn each of them into an error
    and take the replay below down with it. An artifact matters less than the test result and the replay,
    as it does to ``_make_crash_pass_result`` in ``tools/conftest.py``.
    """
    label = request.node.name

    start = log_size(LOG_PATH)
    yield
    if directory := os.environ.get(LOG_DIR_ENV_VAR):
        with contextlib.suppress(OSError):
            save_output(directory, label, start=start)

    if section := format_log_section(LOG_PATH, label, start=start):
        print(f"\n{section}", end="")
