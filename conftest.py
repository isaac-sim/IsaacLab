# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Repo-root pytest configuration shared by every ``source/`` and ``scripts/`` test.

Collects the markers applied to each test and records them in the JUnit XML report so that CI
can carry them into the uploaded test artifact.

Level markers (``unit`` / ``integration`` / ``benchmark``) are applied per file via a module-level ``pytestmark``
and registered in the repo-root ``pyproject.toml``. Select them with the standard ``-m`` syntax,
e.g. ``pytest -m unit source/isaaclab/test`` or ``pytest -m "not unit" source/isaaclab/test``.
"""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

import pytest

# Modules skipped during collection because they declare an extra that is not installed.
_MISSING_EXTRAS_KEY = pytest.StashKey[list]()


def pytest_addoption(parser):
    """Add repository-wide test runtime selection options."""
    parser.addoption(
        "--without-kit",
        action="store_true",
        default=False,
        help="Do not collect modules marked as requiring the Kit runtime.",
    )
    parser.addoption(
        "--run-ci-tests",
        action="store_true",
        default=False,
        help="Collect expensive tests that are omitted from the fast local test profile.",
    )
    parser.addoption(
        "--ignore-missing-extras",
        action="store_true",
        default=False,
        help=(
            "Do not fail when a collected module declares 'requires_extra' for an optional"
            " dependency extra that is not installed. Intended for repository-wide sweeps,"
            " which cannot install mutually conflicting extras at once."
        ),
    )


def _run_ci_tests(config) -> bool:
    """Return whether the complete CI test profile was explicitly requested."""
    return config.getoption("--run-ci-tests") or os.environ.get("ISAACLAB_RUN_CI_TESTS") == "1"


def module_markers(source: str) -> dict[str, list[str]]:
    """Return the module-level ``pytestmark`` markers, mapped to their string arguments.

    The module is parsed rather than imported, so the answer is available before any
    import-time side effect can run. The single-marker form, the list form, and marker
    calls such as ``pytest.mark.requires_extra("ov")`` are all recognized.

    Args:
        source: Contents of the test module.

    Returns:
        Marker names mapped to their literal string arguments, empty if the module does not parse.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    markers: dict[str, list[str]] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
            value = statement.value
        else:
            continue
        if value is None or not any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in targets):
            continue
        for node in ast.walk(value):
            # Match the ``pytest.mark.<name>`` attribute chain, with or without a call.
            if isinstance(node, ast.Call) and _is_pytest_mark(node.func):
                args = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                markers.setdefault(node.func.attr, []).extend(args)
            elif _is_pytest_mark(node):
                markers.setdefault(node.attr, [])
    return markers


def _is_pytest_mark(node: ast.AST) -> bool:
    """Whether the node is a ``pytest.mark.<name>`` attribute chain."""
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
    )


# Extras that cannot all co-resolve (see the conflicts table in pyproject.toml), mapped to
# the module that proves the extra is installed.
_EXTRA_SENTINELS = {
    "mimic": "isaaclab_mimic",
    "opencv": "cv2",
    "ov": "ovphysx",
    "ovphysx": "ovphysx",
    "ovrtx": "ovrtx",
    "rerun": "rerun",
    "teleop": "isaaclab_teleop",
    "viser": "viser",
}


def _missing_extras(required: list[str]) -> list[str]:
    """Return the declared extras whose sentinel module cannot be imported."""
    missing = []
    for extra in required:
        sentinel = _EXTRA_SENTINELS.get(extra, extra)
        try:
            found = importlib.util.find_spec(sentinel) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(extra)
    return missing


def pytest_ignore_collect(collection_path, config):
    """Skip non-local test modules before importing them.

    A regular marker expression deselects tests after their module is imported,
    which is too late for modules that launch Kit or perform costly setup at
    import time. Kit-only and entirely CI-only files therefore use module-level
    ``pytestmark`` assignments that this hook can detect from source.
    """
    path = Path(str(collection_path))
    if path.suffix != ".py" or not path.name.startswith("test_"):
        return None
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None

    run_ci_tests = _run_ci_tests(config)
    # install_ci performs real installs and needs a built wheel. Its own pytest.ini makes it
    # the rootdir when run directly, so this only excludes it from repository-wide sweeps.
    if "install_ci" in path.parts and not run_ci_tests:
        return True

    markers = module_markers(source)
    if "requires_kit" in markers:
        return config.getoption("--without-kit") or not run_ci_tests
    if "ci_only" in markers and not run_ci_tests:
        return True
    if "requires_extra" in markers:
        missing = _missing_extras(markers["requires_extra"])
        if missing:
            # Record rather than skip: an absent extra is reported as a hard error at the
            # end of collection so the suite is never silently dropped.
            config.stash.setdefault(_MISSING_EXTRAS_KEY, []).append((str(path), missing))
            return True
    return None


def _uninstalled_extras(config) -> list[tuple[str, list[str]]]:
    """Modules that were not collected because a declared extra is missing."""
    if config.getoption("--ignore-missing-extras"):
        return []
    return sorted(config.stash.get(_MISSING_EXTRAS_KEY, []))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Report suites that were not run because a declared extra is missing."""
    skipped = _uninstalled_extras(config)
    if not skipped:
        return
    extras = sorted({extra for _, missing in skipped for extra in missing})
    install = " ".join(f"--extra {extra}" for extra in extras)
    terminalreporter.section(f"{len(skipped)} module(s) not run: optional extras missing", red=True)
    for path, missing in skipped:
        terminalreporter.line(f"  {path} needs: {', '.join(missing)}")
    terminalreporter.line("")
    terminalreporter.line(f"  uv run --extra test {install} --locked python -m pytest <path>")
    terminalreporter.line("Extras that cannot co-resolve must be run as separate invocations; pass")
    terminalreporter.line("--ignore-missing-extras to accept the gap in a repository-wide sweep.")


def pytest_sessionfinish(session, exitstatus):
    """Fail the session when a declared extra was missing, without hiding other results."""
    if _uninstalled_extras(session.config) and exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_collection_modifyitems(config, items):
    """Record each collected test's registered markers for the JUnit report.

    Each item's marker set (taken from its ``@pytest.mark`` annotations) is recorded
    as a ``user_properties`` entry so it is emitted into the JUnit XML report as
    ``<property name="markers" value="...">``. CI's ``upload-omni-github-test-results``
    action reads that property and carries the markers into the uploaded test artifact.

    Only markers registered in the repo-root ``pyproject.toml`` (e.g. ``unit``,
    ``integration``, ``benchmark``, ``rendering``) are recorded; pytest's built-in structural marks
    (``parametrize``, ``skip``, ``usefixtures``, ...) are excluded so they do not leak
    into the artifact's ``test_type`` field.
    """
    if not _run_ci_tests(config):
        selected = []
        deselected = []
        for item in items:
            if item.get_closest_marker("ci_only") is None:
                selected.append(item)
            else:
                deselected.append(item)
        if deselected:
            items[:] = selected
            config.hook.pytest_deselected(items=deselected)

    registered = {entry.split(":", 1)[0].strip() for entry in config.getini("markers")}
    for item in items:
        markers = {mark.name for mark in item.iter_markers() if mark.name in registered}
        item.user_properties.append(("markers", ",".join(sorted(markers))))
