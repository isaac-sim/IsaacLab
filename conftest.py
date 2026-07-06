# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared pytest configuration for repository tests.

Wires the Testmon subprocess-coverage hooks (see :mod:`tools.testmon_subprocess_coverage`)
into the session so that code executed in child Python processes is attributed to the
active test. The hooks are no-ops unless pytest-testmon is collecting coverage.
"""

import contextlib
import sys
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> Generator[None, None, None]:
    """Keep tests marked ``smoke`` selected when Testmon filters the collection."""
    always_items = [item for item in items if item.get_closest_marker("smoke") is not None]
    yield

    if config.getoption("testmon_forceselect", default=False):
        selected = set(items)
        items.extend(item for item in always_items if item not in selected)


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
