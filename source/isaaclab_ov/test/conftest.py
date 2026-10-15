# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Every test in this package exercises the OV libraries."""

import pytest


def pytest_itemcollected(item):
    # Runs before ``-m`` deselects, so ``-m ov`` keeps these tests. The CI runner finds
    # this marker by reading this file, as it does a module-level ``pytestmark``.
    item.add_marker(pytest.mark.ov)
