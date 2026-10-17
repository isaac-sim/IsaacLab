# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the ViewerCfg deprecation shim in isaaclab.envs.common.

No simulation context or Kit app required — pure Python.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
from isaaclab_visualizers.kit import KitVisualizerCfg

from isaaclab.envs.common import ViewerCfg, _apply_deprecated_viewer_cfg
from isaaclab.visualizers import VisualizerBackendCfg, VisualizerCfg


def test_viewer_cfg_default_no_warning():
    """ViewerCfg() with all defaults must not emit a DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        ViewerCfg()  # must not raise
        ViewerCfg(eye=(7.5, 7.5, 7.5))


# One tuple field and one non-tuple field: the check loops over every field with these two comparisons.
@pytest.mark.parametrize("field_value", [{"eye": (1.0, 2.0, 3.0)}, {"origin_type": "env"}], ids=["eye", "origin_type"])
def test_viewer_cfg_custom_eye_warns(field_value):
    """ViewerCfg with a non-default field value must emit DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="ViewerCfg is deprecated"):
        ViewerCfg(**field_value)


@pytest.mark.parametrize(
    "case,expected_path",
    [
        ("world", None),
        ("asset_root", "robot"),
        ("asset_body", "robot/panda_hand"),
        ("defaults", None),
        ("configured", None),
    ],
)
def test_deprecated_viewer_forwards_only_to_unconfigured_views(case, expected_path):
    """Common fields migrate once for all viewers; legacy asset tracking belongs to Kit."""
    viewer = ViewerCfg()
    if case != "defaults":
        viewer.eye, viewer.lookat = (1.0, 2.0, 3.0), (0.0, 0.0, 1.0)
        viewer.origin_type = "world" if case == "configured" else case
        viewer.asset_name, viewer.body_name = "robot", "panda_hand"
    shared = VisualizerCfg()
    if case == "configured":
        shared.eye = (8.0, 0.0, 5.0)
    # Environments receive concrete viewer configs after launch has expanded backend selection.
    views = [shared.replace(backend=backend) for backend in (KitVisualizerCfg(), VisualizerBackendCfg())]
    env_cfg = SimpleNamespace(viewer=viewer, sim=SimpleNamespace(visualizer_cfgs=views))
    _apply_deprecated_viewer_cfg(env_cfg)
    for view in views:
        if case in ("defaults", "configured"):
            assert (view.eye, view.lookat, view.window) == (shared.eye, shared.lookat, shared.window)
        else:
            assert (view.eye, view.lookat, view.window.size) == (viewer.eye, viewer.lookat, viewer.resolution)
    assert views[0].backend.origin_type == ("asset" if expected_path else "world")
    assert views[0].backend.origin_track_path == expected_path
