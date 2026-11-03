# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve prebuilt assets lazily for simulator tests, leaving recipe tests independent."""

import os
import sys

import pytest

# Match the suite-root helper setup when tests use pytest's importlib mode.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from microduck_asset_test_utils import prebuilt_usd_path  # isort: skip


@pytest.fixture(scope="module")
def microduck_walk_usd():
    from isaaclab_tasks.contrib.microduck.assets import MICRODUCK_USD_PATH

    return prebuilt_usd_path(MICRODUCK_USD_PATH)


@pytest.fixture(scope="module")
def microduck_allcollisions_usd():
    from isaaclab_tasks.contrib.microduck.assets import MICRODUCK_ALLCOLLISIONS_USD_PATH

    return prebuilt_usd_path(MICRODUCK_ALLCOLLISIONS_USD_PATH)


@pytest.fixture(scope="module")
def microduck_rollers_usd():
    from isaaclab_tasks.contrib.microduck.assets import MICRODUCK_ROLLERS_USD_PATH

    return prebuilt_usd_path(MICRODUCK_ROLLERS_USD_PATH)


@pytest.fixture(scope="module")
def microduck_fullcontact_backlash_usd():
    from isaaclab_tasks.contrib.microduck.assets import MICRODUCK_FULLCONTACT_BACKLASH_USD_PATH

    return prebuilt_usd_path(MICRODUCK_FULLCONTACT_BACKLASH_USD_PATH)
