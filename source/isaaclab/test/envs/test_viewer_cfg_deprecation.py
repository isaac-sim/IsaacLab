# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the ViewerCfg deprecation shim in isaaclab.envs.common.

No simulation context or Kit app required — pure Python.
"""

from __future__ import annotations

import warnings

import pytest

from isaaclab.envs.common import ViewerCfg


def test_viewer_cfg_default_no_warning():
    """ViewerCfg() with all defaults must not emit a DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        ViewerCfg()  # must not raise


# One tuple field and one non-tuple field: the check loops over every field with these two comparisons.
@pytest.mark.parametrize("field_value", [{"eye": (1.0, 2.0, 3.0)}, {"origin_type": "env"}], ids=["eye", "origin_type"])
def test_viewer_cfg_custom_eye_warns(field_value):
    """ViewerCfg with a non-default field value must emit DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="ViewerCfg is deprecated"):
        ViewerCfg(**field_value)


def test_viewer_cfg_default_eye_no_warning():
    """Passing the default eye value explicitly must not trigger a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        ViewerCfg(eye=(7.5, 7.5, 7.5))
