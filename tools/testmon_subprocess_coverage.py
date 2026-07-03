# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Testmon subprocess coverage helpers.

pytest-testmon 2.x only tracks coverage in the pytest process. Tests that spawn
child Python processes (for example ``./isaaclab.sh -p scripts/.../train.py``)
need:

1. A ``.pth`` shim in ``site-packages`` so child interpreters call
   :func:`coverage.process_startup`.
2. ``COVERAGE_PROCESS_START`` pointing at a config with ``parallel = true`` and
   the :mod:`testmon_coverage_context` plugin, so lines are tagged with the
   active test's node id.
3. Child data merged into Testmon's in-memory coverage via
   :meth:`coverage.CoverageData.update` before each batch is read.

Import the pytest hook functions into ``conftest.py``.
"""

from __future__ import annotations

import os
import sysconfig
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest

_TMONTMP_DIR = ".tmontmp"
_COVERAGE_PROCESS_START = "COVERAGE_PROCESS_START"
_COVERAGE_CONTEXT = "COVERAGE_CONTEXT"
_STATE_KEY = pytest.StashKey["SubprocessCoverageState"]()

_SHIM_FILENAME = "testmon_subprocess.pth"
_SHIM_CONTENT = "import coverage; coverage.process_startup()\n"


@dataclass(frozen=True)
class SubprocessCoverageState:
    """Session state for subprocess coverage collection."""

    rc_path: Path
    data_prefix: Path
    shim_path: Path | None
    created_shim: bool
    enabled: bool


def _python_lib_omit_patterns() -> list[str]:
    return [os.path.join(value, "*") for key, value in sysconfig.get_paths().items() if key.endswith("lib")]


def _is_testmon_collecting(config: pytest.Config) -> bool:
    tm_conf = getattr(config, "testmon_config", None)
    return tm_conf is not None and tm_conf.collect


def _site_packages_dir() -> Path | None:
    purelib = sysconfig.get_paths().get("purelib")
    if not purelib:
        return None
    path = Path(purelib)
    return path if path.is_dir() else None


def _subprocess_shim_available(site_dir: Path) -> Path | None:
    """Return an existing ``.pth`` that calls ``process_startup``, if any."""
    for pth_file in site_dir.glob("*.pth"):
        try:
            if "process_startup" in pth_file.read_text(encoding="utf-8", errors="replace"):
                return pth_file
        except OSError:
            continue
    return None


def _install_subprocess_shim() -> tuple[Path | None, bool]:
    """Ensure a ``process_startup`` ``.pth`` shim is on the interpreter path."""
    try:
        import coverage  # noqa: F401
    except ImportError:
        return None, False

    site_dir = _site_packages_dir()
    if site_dir is None:
        return None, False

    existing = _subprocess_shim_available(site_dir)
    if existing is not None:
        return existing, False

    target = site_dir / _SHIM_FILENAME
    try:
        target.write_text(_SHIM_CONTENT, encoding="utf-8")
    except OSError:
        return None, False
    return target, True


def setup_subprocess_coverage(config: pytest.Config) -> SubprocessCoverageState | None:
    """Configure ``COVERAGE_PROCESS_START`` for the pytest session."""
    if not _is_testmon_collecting(config):
        return None

    shim_path, created_shim = _install_subprocess_shim()
    if shim_path is None:
        warnings.warn(
            "testmon subprocess coverage: could not install a process_startup .pth shim into "
            "site-packages, so code executed in child Python processes will not be tracked.",
            stacklevel=1,
        )
        state = SubprocessCoverageState(
            rc_path=Path(), data_prefix=Path(), shim_path=None, created_shim=False, enabled=False
        )
        config.stash[_STATE_KEY] = state
        return state

    rootdir = Path(config.rootpath)
    tmontmp = rootdir / _TMONTMP_DIR
    tmontmp.mkdir(exist_ok=True)

    tools_dir = rootdir / "tools"
    pythonpath_entries = [str(tools_dir)]
    if existing := os.environ.get("PYTHONPATH"):
        pythonpath_entries.append(existing)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    data_prefix = tmontmp / f"subprocess-{os.getpid()}"
    rc_path = tmontmp / f"subprocess-{os.getpid()}.coveragerc"
    rc_path.write_text(
        "\n".join(
            [
                "[run]",
                f"data_file = {data_prefix}",
                "parallel = true",
                "plugins = testmon_coverage_context",
                "include =",
                f"    {os.path.join(str(rootdir), '*')}",
                "omit =",
                *(f"    {pattern}" for pattern in _python_lib_omit_patterns()),
                "",
            ]
        ),
        encoding="utf-8",
    )
    os.environ[_COVERAGE_PROCESS_START] = str(rc_path.resolve())
    state = SubprocessCoverageState(
        rc_path=rc_path, data_prefix=data_prefix, shim_path=shim_path, created_shim=created_shim, enabled=True
    )
    config.stash[_STATE_KEY] = state
    return state


def _child_data_files(data_prefix: Path) -> list[Path]:
    if not data_prefix.parent.is_dir():
        return []
    return sorted(p for p in data_prefix.parent.glob(f"{data_prefix.name}.*") if p.is_file())


def combine_subprocess_coverage(config: pytest.Config) -> None:
    """Merge child-process coverage into Testmon's in-memory coverage data."""
    state = config.stash.get(_STATE_KEY, None)
    if state is None or not state.enabled:
        return

    child_files = _child_data_files(state.data_prefix)
    if not child_files:
        return

    collect_plugin = config.pluginmanager.get_plugin("TestmonCollect")
    if collect_plugin is None:
        return

    cov = collect_plugin.testmon.cov
    if cov is None:
        return

    from coverage import CoverageData

    was_started = cov._started
    if was_started:
        cov.stop()
    target = cov.get_data()
    for child_file in child_files:
        child = CoverageData(basename=str(child_file))
        try:
            child.read()
        except Exception:
            continue
        target.update(child)
        child_file.unlink(missing_ok=True)
    if was_started:
        cov.start()


def teardown_subprocess_coverage(config: pytest.Config) -> None:
    """Remove temporary subprocess coverage configuration."""
    state = config.stash.get(_STATE_KEY, None)
    if state is None:
        return

    os.environ.pop(_COVERAGE_PROCESS_START, None)
    os.environ.pop(_COVERAGE_CONTEXT, None)
    if state.enabled and state.rc_path.is_file():
        state.rc_path.unlink(missing_ok=True)
    if state.enabled:
        for leftover in _child_data_files(state.data_prefix):
            leftover.unlink(missing_ok=True)
    if state.created_shim and state.shim_path is not None and state.shim_path.is_file():
        state.shim_path.unlink(missing_ok=True)
    if _STATE_KEY in config.stash:
        del config.stash[_STATE_KEY]


def pytest_sessionstart(session: pytest.Session) -> None:
    setup_subprocess_coverage(session.config)


def pytest_runtest_setup(item: pytest.Item) -> None:
    if os.environ.get(_COVERAGE_PROCESS_START):
        os.environ[_COVERAGE_CONTEXT] = item.nodeid


def pytest_runtest_teardown(item: pytest.Item) -> None:
    os.environ.pop(_COVERAGE_CONTEXT, None)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    if call.when == "teardown":
        combine_subprocess_coverage(item.config)
    yield


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    teardown_subprocess_coverage(session.config)
