# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from isaaclab_mimic.datagen import utils

pytestmark = pytest.mark.unit


def test_interactive_update_returns_nested_inputs(monkeypatch):
    event_term = SimpleNamespace(params={"outer": {"inner": 1.0}, "top": 2.0})
    param_config = {
        "outer": {"inner": (0.0, 2.0, 0.1)},
        "top": (0.0, 3.0, 0.1),
    }
    monkeypatch.setattr(utils, "get_parameter_input", lambda param_name, *args, **kwargs: param_name)

    inputs = utils.interactive_update_randomizable_params(event_term, "randomize", param_config)

    assert inputs == [
        (["outer", "inner"], "outer.inner"),
        (["top"], "top"),
    ]
