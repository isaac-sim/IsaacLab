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

import os
import re
from pathlib import Path

_REQUIRES_KIT_PATTERN = re.compile(r"^pytestmark\s*=\s*pytest\.mark\.requires_kit\s*$", re.MULTILINE)
_CI_ONLY_PATTERN = re.compile(r"^pytestmark\s*=\s*pytest\.mark\.ci_only\s*$", re.MULTILINE)


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


def _run_ci_tests(config) -> bool:
    """Return whether the complete CI test profile was explicitly requested."""
    return config.getoption("--run-ci-tests") or os.environ.get("ISAACLAB_RUN_CI_TESTS") == "1"


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
    if _REQUIRES_KIT_PATTERN.search(source) is not None:
        return config.getoption("--without-kit") or not run_ci_tests
    if _CI_ONLY_PATTERN.search(source) is not None:
        return not run_ci_tests
    return None


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
