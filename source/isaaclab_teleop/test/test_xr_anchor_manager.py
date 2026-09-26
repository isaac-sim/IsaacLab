# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for XR anchor configuration."""

from types import SimpleNamespace

import pytest
from isaaclab_teleop import xr_anchor_manager
from isaaclab_teleop.xr_cfg import XrCfg


def test_near_plane_reaches_xr_profiles(monkeypatch: pytest.MonkeyPatch):
    """A configured near plane must reach both profile settings used by XR sessions."""
    values = {}

    class Settings:
        def set_float(self, path: str, value: float) -> None:
            values[path] = value

        def set_string(self, path: str, value: str) -> None:
            values[path] = value

    settings = Settings()
    monkeypatch.setattr(
        xr_anchor_manager, "carb", SimpleNamespace(settings=SimpleNamespace(get_settings=lambda: settings))
    )
    monkeypatch.setattr(xr_anchor_manager, "XRCore", None)
    monkeypatch.setattr(xr_anchor_manager, "_xr_anchor_prim_exists", lambda _: True)

    xr_anchor_manager.XrAnchorManager(XrCfg(near_plane=0.05))

    assert values["/persistent/xr/render/nearPlane"] == pytest.approx(0.05)
    assert values["/persistent/xr/profile/ar/render/nearPlane"] == pytest.approx(0.05)
    assert values["/persistent/xr/profile/vr/render/nearPlane"] == pytest.approx(0.05)
