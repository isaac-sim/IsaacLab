# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Workspace-root pytest hooks.

The only hook today is the multi-GPU shard skip: when a shard pins itself to a
non-default cuda device via :envvar:`ISAACLAB_SIM_DEVICE` (e.g. ``cuda:2``),
skip any test that is not parametrized over a ``device`` argument. Such tests
exercise device-agnostic logic that single-GPU CI already covers on
``cuda:0``, so running them again on every non-default shard duplicates work
without surfacing any new failure mode.

The hook is a no-op outside the multi-GPU lane (``ISAACLAB_SIM_DEVICE`` unset
or equal to ``cuda:0``), so this file does not change pytest behavior under
single-GPU CI.
"""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    sim_device = os.environ.get("ISAACLAB_SIM_DEVICE", "")
    if not sim_device.startswith("cuda:") or sim_device == "cuda:0":
        return
    skip_marker = pytest.mark.skip(reason=f"non-device-parametrized; covered by single-GPU lane (shard={sim_device})")
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None or "device" not in callspec.params:
            item.add_marker(skip_marker)
