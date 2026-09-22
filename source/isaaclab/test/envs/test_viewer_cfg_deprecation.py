# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the ``ViewerCfg`` deprecation shim in :mod:`isaaclab.envs.common`."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

from isaaclab.envs.common import ViewerCfg, _apply_deprecated_viewer_cfg

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "overrides",
    [{"eye": (1.0, 2.0, 3.0)}, {"lookat": (1.0, 0.0, 0.0)}, {"resolution": (640, 480)}, {"origin_type": "env"}],
    ids=["eye", "lookat", "resolution", "origin_type"],
)
def test_viewer_cfg_non_default_field_warns(overrides):
    """A non-default camera field emits a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="ViewerCfg is deprecated"):
        ViewerCfg(**overrides)


@pytest.mark.parametrize("overrides", [{}, {"eye": (7.5, 7.5, 7.5)}], ids=["defaults", "explicit_default"])
def test_viewer_cfg_default_no_warning(overrides):
    """Default values, even when passed explicitly, stay silent so unmigrated task configs do not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        ViewerCfg(**overrides)


def _make_env_cfg(eye=(7.5, 7.5, 7.5), **viewer_fields):
    viewer = ViewerCfg()
    viewer.eye = eye
    for name, value in viewer_fields.items():
        setattr(viewer, name, value)
    return SimpleNamespace(viewer=viewer, sim=SimpleNamespace(default_visualizer_cfg=None))


def test_apply_deprecated_viewer_noop_when_defaults():
    """Default viewer values are not forwarded to the visualizer configuration."""
    env_cfg = _make_env_cfg()
    _apply_deprecated_viewer_cfg(env_cfg)
    assert env_cfg.sim.default_visualizer_cfg is None


@pytest.mark.parametrize(
    ("viewer_fields", "expected_origin_type", "expected_track_path"),
    [
        ({}, "world", None),
        ({"origin_type": "asset_root", "asset_name": "robot"}, "asset", "robot"),
        ({"origin_type": "asset_body", "asset_name": "robot", "body_name": "panda_hand"}, "asset", "robot/panda_hand"),
    ],
    ids=["world", "asset_root", "asset_body"],
)
def test_apply_deprecated_viewer_sets_visualizer_cfg(viewer_fields, expected_origin_type, expected_track_path):
    """Non-default viewer values are forwarded, mapping the asset origin types onto ``origin_track_path``."""
    env_cfg = _make_env_cfg(eye=(1.0, 2.0, 3.0), **viewer_fields)
    _apply_deprecated_viewer_cfg(env_cfg)
    cfg = env_cfg.sim.default_visualizer_cfg
    assert cfg is not None
    assert cfg.eye == (1.0, 2.0, 3.0)
    assert getattr(cfg, "origin_type", "world") == expected_origin_type
    assert getattr(cfg, "origin_track_path", None) == expected_track_path


def test_apply_deprecated_viewer_skips_when_default_visualizer_cfg_already_set():
    """An existing visualizer configuration is never overwritten by the shim."""
    existing_cfg = object()
    env_cfg = _make_env_cfg(eye=(1.0, 2.0, 3.0))
    env_cfg.sim.default_visualizer_cfg = existing_cfg
    _apply_deprecated_viewer_cfg(env_cfg)
    assert env_cfg.sim.default_visualizer_cfg is existing_cfg
