# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for rendering mode resolution on renderer configs (no Kit required)."""

from __future__ import annotations

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.rendering_mode.rendering_mode_cfg import RenderingModeCfg
from isaaclab.rendering_mode.rendering_mode_utils import (
    apply_mode_profile_to_renderer_cfg,
    apply_newton_warp_mode_cfg_to_renderer_cfg,
    resolve_rendering_mode_name_for_renderer_cfg,
)


class _FakeSettings:
    def __init__(self, explicit: bool, mode: str):
        self._data = {
            "/isaaclab/rendering/rendering_mode/explicit": explicit,
            "/isaaclab/rendering/rendering_mode": mode,
        }

    def get(self, key: str):
        return self._data[key]


def test_resolve_renderer_mode_cli_explicit_overrides_cfg():
    r_cfg = RendererCfg(renderer_type="isaac_rtx", rendering_mode="quality")
    settings = _FakeSettings(explicit=True, mode="performance")
    assert resolve_rendering_mode_name_for_renderer_cfg(settings.get, r_cfg) == "performance"


def test_resolve_renderer_mode_uses_cfg_when_cli_not_explicit():
    r_cfg = RendererCfg(renderer_type="isaac_rtx", rendering_mode="balanced")
    settings = _FakeSettings(explicit=False, mode="")
    assert resolve_rendering_mode_name_for_renderer_cfg(settings.get, r_cfg) == "balanced"


def test_apply_newton_warp_overrides_mutate_cfg():
    mode_cfg = RenderingModeCfg(
        newton_warp_enable_shadows=True,
        newton_warp_max_distance=42.0,
    )
    from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg

    ren = NewtonWarpRendererCfg()
    assert ren.enable_shadows is False
    assert ren.max_distance == 1000.0
    apply_newton_warp_mode_cfg_to_renderer_cfg(ren, mode_cfg)
    assert ren.enable_shadows is True
    assert ren.max_distance == 42.0


def test_apply_mode_profile_kit_branch_calls_set_setting():
    recorded: list[tuple[str, object]] = []

    def get_setting(key: str):
        return _FakeSettings(explicit=False, mode="").get(key)

    def set_setting(name: str, value: object) -> None:
        recorded.append((name, value))

    r_cfg = RendererCfg(renderer_type="isaac_rtx", rendering_mode="performance")
    mode_cfgs = {"performance": RenderingModeCfg(rendering_mode_preset="performance")}
    apply_mode_profile_to_renderer_cfg(get_setting, set_setting, r_cfg, mode_cfgs, logger=__import__("logging").getLogger(__name__))
    assert any(k == "/rtx/shadows/enabled" for k, _ in recorded)
