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

2. **Per-file JUnit XML reports** — :class:`_KitSessionPlugin` accumulates
   test results via ``pytest_runtest_logreport`` and the module-scoped autouse
   fixture ``_kit_module_fence`` writes one ``tests/test-reports-<slug>.xml``
   per test file in its teardown phase.  This replaces the ``--junitxml`` flag
   (which would write a single combined report) and preserves the per-file
   report layout expected by ``tools/conftest.py``.

3. **Kit/USD state reset** — the same ``_kit_module_fence`` fixture calls
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

# Holds the first AppLauncher instance created in this process.
_shared_launcher: object = None


def _patch_app_launcher() -> None:
    """Monkey-patch AppLauncher.__init__ to reuse the first Kit session.

    ``pytest_configure`` (and therefore this function) fires before pytest
    imports any test module, so the patch is in place before the first
    module-level ``simulation_app = AppLauncher(...).app`` call executes.

    The first call to the patched ``__init__`` runs the original
    initialisation normally and saves the resulting instance.  Every
    subsequent call copies all attributes from the saved instance, making the
    new ``AppLauncher`` object indistinguishable from the first without
    starting an additional :class:`isaacsim.simulation_app.SimulationApp`.
    """
    global _shared_launcher

    try:
        from isaaclab.app import AppLauncher
    except Exception:
        return

    original_init = AppLauncher.__init__

    def _shared_init(self, launcher_args=None, **kwargs):
        global _shared_launcher
        if _shared_launcher is not None:
            # Reuse the existing Kit session: copy all instance state.
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
# Pytest plugin registration
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Register the plugin and patch AppLauncher before test collection."""
    _patch_app_launcher()
    plugin = _KitSessionPlugin()
    config.pluginmanager.register(plugin, "kit_session_plugin")


# ---------------------------------------------------------------------------
# Plugin implementation
# ---------------------------------------------------------------------------


class _KitSessionPlugin:
    """Accumulate per-file test results and write JUnit XMLs on file transitions.

    The ``_kit_module_fence`` fixture (module-scoped, autouse) drives both the
    report write and the Kit state reset at the end of each test file.
    ``pytest_runtest_logreport`` accumulates individual test outcomes into
    per-file :class:`~junitparser.TestSuite` objects while tests are running.
    ``pytest_sessionfinish`` flushes any suite that was not yet written
    (safety net for error paths where the fixture teardown was skipped).
    """

    def __init__(self):
        # nodeid -> TestCase that started setup but whose call phase has not
        # yet been recorded.
        self._pending: dict[str, TestCase] = {}
        # filepath -> TestSuite for all test cases from that file so far.
        self._suites: dict[str, TestSuite] = {}
        # Paths whose JUnit XML has already been written (guards against
        # double-write if fixture teardown and sessionfinish both fire).
        self._written: set[str] = set()

    # -- module-scoped autouse fixture --------------------------------------

    @pytest.fixture(scope="module", autouse=True)
    def _kit_module_fence(self, request):
        """Write the per-file report and reset Kit state after each test file.

        The setup half (before ``yield``) is intentionally empty — AppLauncher
        is already alive courtesy of the batch session.  The teardown half
        writes the JUnit XML for the completed file and opens a fresh USD
        stage so the next file starts with a clean world.
        """
        yield
        filepath = str(request.fspath)
        self._write_report(filepath)
        _reset_kit_state()

    # -- hook: accumulate test results -------------------------------------

    def pytest_runtest_logreport(self, report) -> None:
        """Record individual test phase outcomes into the per-file suite."""
        filepath = str(report.fspath)
        nodeid = report.nodeid

        if report.when == "setup":
            if report.passed:
                case = TestCase(name=self._case_name(nodeid), classname=self._classname(nodeid))
                case.time = 0.0
                self._pending[nodeid] = case
            elif report.failed:
                case = TestCase(name=self._case_name(nodeid), classname=self._classname(nodeid))
                case.time = report.duration
                case.result = Error(message=str(report.longrepr))
                self._suite(filepath).add_testcase(case)
            elif report.skipped:
                case = TestCase(name=self._case_name(nodeid), classname=self._classname(nodeid))
                case.time = report.duration
                case.result = Skipped()
                self._suite(filepath).add_testcase(case)

        elif report.when == "call":
            case = self._pending.pop(nodeid, None)
            if case is None:
                case = TestCase(name=self._case_name(nodeid), classname=self._classname(nodeid))
                case.time = 0.0
            case.time = (case.time or 0.0) + report.duration
            if report.failed:
                case.result = Failure(message=str(report.longrepr))
            elif report.skipped:
                case.result = Skipped()
            # passed: leave result unset — no child element means "passed" in JUnit XML
            self._suite(filepath).add_testcase(case)

        elif report.when == "teardown" and report.failed:
            # Teardown errors are appended as a separate Error case so they
            # appear in the report without overwriting the call-phase result.
            case = TestCase(
                name=f"{self._case_name(nodeid)}[teardown]",
                classname=self._classname(nodeid),
            )
            case.time = report.duration
            case.result = Error(message=str(report.longrepr))
            self._suite(filepath).add_testcase(case)

    # -- hook: safety-net flush -------------------------------------------

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        """Flush any per-file report not yet written (e.g. fixture teardown skipped).

        Also drains ``_pending``: tests whose setup passed but whose call phase
        was never recorded (e.g. session aborted via ``--maxfail`` or
        ``KeyboardInterrupt``) are written as ``Error`` entries so they appear
        in the JUnit output instead of silently vanishing.
        """
        # Drain orphaned setup-passed cases before flushing suites.
        for nodeid, case in list(self._pending.items()):
            filepath = nodeid.split("::")[0]
            case.result = Error(message="Test interrupted: call phase never executed")
            self._suite(filepath).add_testcase(case)
        self._pending.clear()

        for filepath in list(self._suites):
            self._write_report(filepath)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _case_name(nodeid: str) -> str:
        parts = nodeid.split("::")
        return parts[-1] if len(parts) > 1 else nodeid

    @staticmethod
    def _classname(nodeid: str) -> str:
        """Derive a JUnit classname from a pytest node ID."""
        parts = nodeid.split("::")
        if len(parts) >= 3:
            # path/test_foo.py::TestClass::test_method  ->  TestClass
            return parts[-2]
        if len(parts) == 2:
            # path/test_foo.py::test_function  ->  test_foo
            return os.path.splitext(os.path.basename(parts[0]))[0]
        return ""

    def _suite(self, filepath: str) -> TestSuite:
        if filepath not in self._suites:
            name = os.path.splitext(os.path.basename(filepath))[0]
            self._suites[filepath] = TestSuite(name=name)
        return self._suites[filepath]

    def _write_report(self, filepath: str) -> None:
        if filepath in self._written:
            return
        suite = self._suites.get(filepath)
        if suite is None:
            return
        os.makedirs("tests", exist_ok=True)
        slug = filepath.replace("/", "__").replace("\\", "__")
        report_path = f"tests/test-reports-{slug}.xml"
        xml = JUnitXml()
        xml.add_testsuite(suite)
        xml.write(report_path)
        self._written.add(filepath)
