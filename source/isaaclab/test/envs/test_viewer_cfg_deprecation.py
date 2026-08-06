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


def test_viewer_cfg_custom_eye_warns():
    """ViewerCfg(eye=...) with a non-default value must emit DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="ViewerCfg is deprecated"):
        ViewerCfg(eye=(1.0, 2.0, 3.0))


def test_viewer_cfg_custom_lookat_warns():
    """ViewerCfg(lookat=...) with a non-default value must emit DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="ViewerCfg is deprecated"):
        ViewerCfg(lookat=(1.0, 0.0, 0.0))


def test_viewer_cfg_custom_resolution_warns():
    """ViewerCfg(resolution=...) with a non-default value must emit DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="ViewerCfg is deprecated"):
        ViewerCfg(resolution=(640, 480))


def test_viewer_cfg_custom_origin_type_warns():
    """ViewerCfg(origin_type=...) with a non-default value must emit DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="ViewerCfg is deprecated"):
        ViewerCfg(origin_type="env")


def test_viewer_cfg_default_eye_no_warning():
    """Passing the default eye value explicitly must not trigger a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        ViewerCfg(eye=(7.5, 7.5, 7.5))


def test_viewer_cfg_warning_is_deprecation_warning_subclass():
    """The emitted warning must be a DeprecationWarning (not just a UserWarning)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ViewerCfg(eye=(0.0, 0.0, 1.0))

    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
