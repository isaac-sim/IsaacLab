# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest plugin for shared Kit sessions across multiple test files.

Loaded via ``-p _kit_session_plugin`` in batch subprocess commands built by
``tools/conftest.py`` when :envvar:`ISAACLAB_KIT_SESSION_FILES` is greater
than 1.  It should never be loaded in single-file subprocess invocations.

Responsibilities
----------------
1. **Kit session reuse** — ``pytest_configure`` monkey-patches
   :class:`~isaaclab.app.AppLauncher.__init__` so that the first call in the
   process starts Kit normally while every subsequent call (from other test
   file modules imported during collection) copies all attributes from the
   first instance instead of launching a second :class:`SimulationApp`.  The
   patch is applied before pytest begins collecting and importing test modules,
   which is the only window in which it can intercept module-level
   ``AppLauncher(...)`` calls.

2. **Per-file JUnit XML reports** — ``pytest_runtest_logreport`` accumulates
   test results and the module-scoped autouse fixture ``_kit_module_fence``
   writes one ``tests/test-reports-<slug>.xml`` per test file in its teardown
   phase.  This replaces the ``--junitxml`` flag (which would write a single
   combined report) and preserves the per-file report layout expected by
   ``tools/conftest.py``.

3. **Kit/USD state reset** — ``_kit_module_fence`` calls
   :func:`_reset_kit_state` in teardown so residual USD prims, timeline
   position, and physics state from file *N* do not bleed into file *N+1*.
"""

from __future__ import annotations

import os

import pytest
from junitparser import Error, Failure, JUnitXml, Skipped, TestCase, TestSuite


# ---------------------------------------------------------------------------
# Kit state reset
# ---------------------------------------------------------------------------


def _reset_kit_state() -> None:
    """Reset Kit/USD state between test files in a shared session.

    Opens a fresh empty stage and stops the timeline so that prims, physics
    state, and timeline position from the previous test file do not bleed into
    the next one.  All errors are suppressed — a reset failure should not
    abort the entire batch run.
    """
    try:
        import omni.timeline

        omni.timeline.get_timeline_interface().stop()
    except Exception:
        pass

    try:
        import omni.usd

        omni.usd.get_context().new_stage()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# AppLauncher singleton patch
# ---------------------------------------------------------------------------

_shared_launcher: object = None


def pytest_configure(config) -> None:
    """Patch AppLauncher.__init__ to reuse the first Kit session."""
    global _shared_launcher

    try:
        from isaaclab.app import AppLauncher
    except Exception:
        return

    original_init = AppLauncher.__init__

    def _shared_init(self, launcher_args=None, **kwargs):
        global _shared_launcher
        if _shared_launcher is not None:
            self.__dict__.update(_shared_launcher.__dict__)
            return
        original_init(self, launcher_args, **kwargs)
        _shared_launcher = self
        # Guard: prevent test files from closing the shared SimulationApp.
        # Any call to simulation_app.close() in a batch file's teardown would
        # shut down Kit for all remaining files in the batch, causing cryptic
        # import/startup errors instead of a clear failure message.
        app = getattr(self, "app", None)
        if app is not None:
            app.close = lambda *a, **kw: None

    AppLauncher.__init__ = _shared_init


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _kit_session():
    """Hold the shared Kit session alive for the entire batch."""
    yield


@pytest.fixture(scope="module", autouse=True)
def _kit_module_fence(request):
    """Write the per-file JUnit XML and reset Kit state after each test file."""
    yield
    _write_report(str(request.fspath))
    _reset_kit_state()


# ---------------------------------------------------------------------------
# Hooks: accumulate test results and flush reports
# ---------------------------------------------------------------------------

# nodeid -> TestCase that started setup but whose call phase has not yet been recorded.
_pending: dict[str, TestCase] = {}
# filepath -> TestSuite for all test cases from that file so far.
_suites: dict[str, TestSuite] = {}
# Paths whose JUnit XML has already been written (guards against double-write).
_written: set[str] = set()


def pytest_runtest_logreport(report) -> None:
    """Record individual test phase outcomes into the per-file suite."""
    filepath = str(report.fspath)
    nodeid = report.nodeid

    if report.when == "setup":
        if report.passed:
            case = TestCase(name=_case_name(nodeid), classname=_classname(nodeid))
            case.time = 0.0
            _pending[nodeid] = case
        elif report.failed:
            case = TestCase(name=_case_name(nodeid), classname=_classname(nodeid))
            case.time = report.duration
            case.result = Error(message=str(report.longrepr))
            _suite(filepath).add_testcase(case)
        elif report.skipped:
            case = TestCase(name=_case_name(nodeid), classname=_classname(nodeid))
            case.time = report.duration
            case.result = Skipped()
            _suite(filepath).add_testcase(case)

    elif report.when == "call":
        case = _pending.pop(nodeid, None)
        if case is None:
            case = TestCase(name=_case_name(nodeid), classname=_classname(nodeid))
            case.time = 0.0
        case.time = (case.time or 0.0) + report.duration
        if report.failed:
            case.result = Failure(message=str(report.longrepr))
        elif report.skipped:
            case.result = Skipped()
        # passed: leave result unset — no child element means "passed" in JUnit XML
        _suite(filepath).add_testcase(case)

    elif report.when == "teardown" and report.failed:
        # Teardown errors are appended as a separate Error case so they
        # appear in the report without overwriting the call-phase result.
        case = TestCase(
            name=f"{_case_name(nodeid)}[teardown]",
            classname=_classname(nodeid),
        )
        case.time = report.duration
        case.result = Error(message=str(report.longrepr))
        _suite(filepath).add_testcase(case)


def pytest_sessionfinish(session, exitstatus) -> None:
    """Drain _pending and flush any per-file report not yet written.

    Tests whose setup passed but whose call phase was never recorded (e.g.
    session aborted via ``--maxfail`` or ``KeyboardInterrupt``) are written as
    ``Error`` entries so they appear in the JUnit output instead of vanishing.
    """
    for nodeid, case in list(_pending.items()):
        filepath = nodeid.split("::")[0]
        case.result = Error(message="Test interrupted: call phase never executed")
        _suite(filepath).add_testcase(case)
    _pending.clear()

    for filepath in list(_suites):
        _write_report(filepath)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _case_name(nodeid: str) -> str:
    parts = nodeid.split("::")
    return parts[-1] if len(parts) > 1 else nodeid


def _classname(nodeid: str) -> str:
    parts = nodeid.split("::")
    if len(parts) >= 3:
        # path/test_foo.py::TestClass::test_method  ->  TestClass
        return parts[-2]
    if len(parts) == 2:
        # path/test_foo.py::test_function  ->  test_foo
        return os.path.splitext(os.path.basename(parts[0]))[0]
    return ""


def _suite(filepath: str) -> TestSuite:
    if filepath not in _suites:
        name = os.path.splitext(os.path.basename(filepath))[0]
        _suites[filepath] = TestSuite(name=name)
    return _suites[filepath]


def _write_report(filepath: str) -> None:
    if filepath in _written:
        return
    suite = _suites.get(filepath)
    if suite is None:
        return
    os.makedirs("tests", exist_ok=True)
    slug = filepath.replace("/", "__").replace("\\", "__")
    report_path = f"tests/test-reports-{slug}.xml"
    xml = JUnitXml()
    xml.add_testsuite(suite)
    xml.write(report_path)
    _written.add(filepath)
