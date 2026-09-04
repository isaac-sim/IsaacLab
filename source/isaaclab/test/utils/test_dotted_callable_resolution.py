# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections import Counter

import pytest

from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString

pytestmark = pytest.mark.unit


@configclass
class _NestedCallableCfg:
    updater: str = "collections:Counter.update"


def test_configclass_wraps_nested_callable_reference():
    """Configclass should wrap dotted callable references as lazy resolvable strings."""
    cfg = _NestedCallableCfg()

    assert isinstance(cfg.updater, ResolvableString)
    counter = Counter()
    cfg.updater(counter, {"value": 2})
    assert counter["value"] == 2
