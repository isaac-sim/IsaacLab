# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared pytest configuration for repository tests."""

from collections.abc import Generator

import pytest


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> Generator[None, None, None]:
    """Keep tests marked ``always`` selected when Testmon filters the collection."""
    always_items = [item for item in items if item.get_closest_marker("always")]
    yield

    if config.getoption("testmon_forceselect", default=False):
        selected = set(items)
        items.extend(item for item in always_items if item not in selected)
