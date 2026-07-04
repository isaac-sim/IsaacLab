# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Repo-root pytest configuration shared by every ``source/`` and ``scripts/`` test.

Auto-applies *intent* markers (``unit``, ``integration``, ``rendering``, ``training``,
``performance``, ``benchmark``) to each collected test based on its file path, so tests
are categorizable and targetable without hand-annotating hundreds of files.

The markers are registered in the repo-root ``pyproject.toml``. Selection works with the
standard ``-m`` expression syntax, e.g.::

    pytest -m rendering source/isaaclab/test
    pytest -m "unit and not performance" source/isaaclab/test

Every test receives exactly one *level* marker (``unit`` or ``integration``) and, when its
path matches, one or more *flavor* markers (``rendering`` / ``training`` / ``performance`` /
``benchmark``). Classification is a best-effort heuristic on the path: the flavor markers map
to precise, stable tokens, while the unit/integration split defaults to ``integration`` (the
common case in Isaac Lab, where most tests launch the simulator) and only downgrades to
``unit`` for directories and files known to exercise isolated logic.

This lives at the repository root because the custom test runner in ``tools/conftest.py`` drives
each test file in its own subprocess ``pytest`` invocation rooted here; a root ``conftest.py`` is
therefore the single location discovered for all of them. ``tools/changelog`` and
``source/isaaclab/test/install_ci`` set their own rootdir and are unaffected.
"""

from __future__ import annotations

# Directory names (as path segments) whose tests exercise isolated logic and do not launch the
# simulator. Tests under these directories get the ``unit`` level marker instead of ``integration``.
_UNIT_DIR_SEGMENTS = frozenset({"utils", "cli", "deps", "test_mock_interfaces"})

# Filename substrings that mark rendering / camera / visualizer pipeline tests.
_RENDERING_TOKENS = ("render", "camera", "tiled", "rtx", "visualizer", "ppisp")


def _intent_markers(path: str) -> set[str]:
    """Return the set of intent marker names for a test at ``path``.

    Args:
        path: Filesystem path of the test file (any separator style).

    Returns:
        Marker names to apply: exactly one level marker (``unit``/``integration``) plus any
        matching flavor markers (``rendering``/``training``/``performance``/``benchmark``).
    """
    path = path.replace("\\", "/").lower()
    segments = set(path.split("/"))
    name = path.rsplit("/", 1)[-1]

    markers: set[str] = set()

    # -- flavor markers (a test may carry several) --
    if any(token in name for token in _RENDERING_TOKENS):
        markers.add("rendering")
    if "train" in name:  # matches "train" and "training"
        markers.add("training")
    if "performance" in segments or "perf" in name:
        markers.add("performance")
    if segments & {"benchmark", "benchmarks", "benchmarking"}:
        markers.add("benchmark")

    # -- level marker (exactly one) --
    is_unit = bool(segments & _UNIT_DIR_SEGMENTS) or name.endswith(("_kernels.py", "_kernel.py"))
    markers.add("unit" if is_unit else "integration")

    return markers


def pytest_collection_modifyitems(config, items):
    """Attach path-derived intent markers to every collected test item.

    Besides registering the markers for ``-m`` selection, each item's marker set is
    recorded as a ``user_properties`` entry so it is emitted into the JUnit XML report as
    ``<property name="markers" value="...">``. CI's ``upload-omni-github-test-results``
    action reads that property and carries the markers into the uploaded test artifact.
    """
    for item in items:
        # ``item.path`` is a pathlib.Path on modern pytest; fall back to the legacy ``fspath``.
        path = str(getattr(item, "path", None) or item.fspath)
        markers = _intent_markers(path)
        for marker in markers:
            item.add_marker(marker)
        item.user_properties.append(("markers", ",".join(sorted(markers))))
