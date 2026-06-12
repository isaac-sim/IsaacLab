# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""pytest plugin: drop device-agnostic tests from this run.

Deselects every collected test that is not parametrized over ``device``. It
carries no mask logic of its own. The executor (``tools/conftest.py``) injects
it via ``-p _agnostic_select`` only for a unit whose mask lacks cpu — i.e. an
mgpu shard, or the cuda unit of a can't-mix split — because in those units a
device-agnostic test would either run redundantly on every shard or run a second
time alongside the cpu unit. Device variant selection itself is handled upstream
by ``test_devices()`` reading the unit's ``ISAACLAB_TEST_DEVICES`` mask, so this
plugin only has to remove the paramless remainder.
"""

from __future__ import annotations


def _has_device_param(item) -> bool:
    """Return whether a collected item is parametrized over ``device``."""
    callspec = getattr(item, "callspec", None)
    return callspec is not None and "device" in callspec.params


def pytest_collection_modifyitems(config, items):
    drop = [item for item in items if not _has_device_param(item)]
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = [item for item in items if _has_device_param(item)]
