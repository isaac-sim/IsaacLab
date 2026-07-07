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

Also wires the Testmon subprocess-coverage hooks (see :mod:`tools.testmon_subprocess_coverage`)
into the session so that code executed in child Python processes is attributed to the
active test. The hooks are no-ops unless pytest-testmon is collecting coverage.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> Generator[None, None, None]:
    """Keep ``smoke`` tests selected under Testmon and record markers for the JUnit report.

    Captures the ``smoke``-marked items before other hooks run, then yields so that
    Testmon (and any other ``modifyitems`` hooks) can filter the collection. Afterwards it
    re-adds any ``smoke`` tests Testmon dropped, and records each remaining item's registered
    markers as a ``user_properties`` entry so it is emitted into the JUnit XML report as
    ``<property name="markers" value="...">``. CI's ``upload-omni-github-test-results``
    action reads that property and carries the markers into the uploaded test artifact.

    Only markers registered in the repo-root ``pyproject.toml`` (e.g. ``unit``,
    ``integration``, ``benchmark``, ``rendering``) are recorded; pytest's built-in structural marks
    (``parametrize``, ``skip``, ``usefixtures``, ...) are excluded so they do not leak
    into the artifact's ``test_type`` field.
    """
    always_items = [item for item in items if item.get_closest_marker("smoke") is not None]
    yield

    if config.getoption("testmon_forceselect", default=False):
        selected = set(items)
        items.extend(item for item in always_items if item not in selected)

    registered = {entry.split(":", 1)[0].strip() for entry in config.getini("markers")}
    for item in items:
        markers = {mark.name for mark in item.iter_markers() if mark.name in registered}
        item.user_properties.append(("markers", ",".join(sorted(markers))))


_TOOLS_DIR = Path(__file__).resolve().parent / "tools"
if _TOOLS_DIR.is_dir():
    if str(_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(_TOOLS_DIR))

    # Guard the import too: when ``tools/`` is absent (partial checkout, artefact-only
    # CI image) the hooks should stay unregistered no-ops rather than aborting the whole
    # test collection with a ``ModuleNotFoundError``.
    with contextlib.suppress(ImportError):
        from testmon_subprocess_coverage import (  # noqa: F401
            pytest_runtest_makereport,
            pytest_runtest_setup,
            pytest_runtest_teardown,
            pytest_sessionfinish,
            pytest_sessionstart,
        )
