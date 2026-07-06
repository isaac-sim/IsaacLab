# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Repo-root pytest configuration shared by every ``source/`` and ``scripts/`` test.

Collects the markers applied to each test and records them in the JUnit XML report so that CI
can carry them into the uploaded test artifact.

Level markers (``unit`` / ``integration``) are applied per file via a module-level ``pytestmark``
and registered in the repo-root ``pyproject.toml``. Select them with the standard ``-m`` syntax,
e.g. ``pytest -m unit source/isaaclab/test`` or ``pytest -m "not unit" source/isaaclab/test``.
"""

from __future__ import annotations


def pytest_collection_modifyitems(config, items):
    """Record each collected test's intent markers for the JUnit report.

    Each item's marker set (taken from its ``@pytest.mark`` annotations) is recorded
    as a ``user_properties`` entry so it is emitted into the JUnit XML report as
    ``<property name="markers" value="...">``. CI's ``upload-omni-github-test-results``
    action reads that property and carries the markers into the uploaded test artifact.
    """
    for item in items:
        markers = {mark.name for mark in item.iter_markers()}
        item.user_properties.append(("markers", ",".join(sorted(markers))))
