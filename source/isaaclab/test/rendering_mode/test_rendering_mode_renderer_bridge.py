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
    resolve_rendering_mode_cfg,
    resolve_rendering_mode_name_for_renderer_cfg,
)


class _FakeSettings:
    def __init__(self, explicit: bool, mode):
        self._data: dict[str, object] = {
            "/isaaclab/rendering/rendering_mode/explicit": explicit,
            "/isaaclab/rendering/rendering_mode": mode,
        }

    def get(self, key: str):
        return self._data.get(key)


def test_resolve_renderer_mode_cli_explicit_overrides_cfg():
    r_cfg = RendererCfg(renderer_type="isaac_rtx", rendering_mode="quality")
    settings = _FakeSettings(explicit=True, mode="performance")
    assert resolve_rendering_mode_name_for_renderer_cfg(settings.get, r_cfg) == "performance"


def test_resolve_renderer_mode_cli_explicit_coerces_carb_dict():
    """Some Kit builds return a subtree dict from carb get(); profile name may be embedded."""
    r_cfg = RendererCfg(renderer_type="isaac_rtx", rendering_mode="performance")
    settings = _FakeSettings(explicit=True, mode={"leaf": "quality"})
    assert resolve_rendering_mode_name_for_renderer_cfg(settings.get, r_cfg) == "quality"


def test_resolve_renderer_mode_cli_explicit_coerces_nested_carb_dict():
    r_cfg = RendererCfg(renderer_type="isaac_rtx", rendering_mode="performance")
    settings = _FakeSettings(explicit=True, mode={"outer": {"inner": "balanced"}})
    assert resolve_rendering_mode_name_for_renderer_cfg(settings.get, r_cfg) == "balanced"


def test_resolve_renderer_mode_cli_explicit_coerces_carb_dict_value_key():
    """Carb may expose the profile under a ``value`` (or similar) leaf."""
    r_cfg = RendererCfg(renderer_type="isaac_rtx", rendering_mode="balanced")
    settings = _FakeSettings(explicit=True, mode={"value": "performance"})
    assert resolve_rendering_mode_name_for_renderer_cfg(settings.get, r_cfg) == "performance"


def test_cli_explicit_unreadable_carb_dict_without_profile_string_returns_none():
    """When generic get() yields a dict with no embedded profile strings, resolution fails."""
    r_cfg = RendererCfg(renderer_type="isaac_rtx", rendering_mode="performance")
    settings = _FakeSettings(explicit=True, mode={"flags": True, "count": 2})
    assert resolve_rendering_mode_name_for_renderer_cfg(settings.get, r_cfg) is None


def test_resolve_renderer_mode_uses_cfg_when_cli_not_explicit():
    r_cfg = RendererCfg(renderer_type="isaac_rtx", rendering_mode="balanced")
    settings = _FakeSettings(explicit=False, mode="")
    assert resolve_rendering_mode_name_for_renderer_cfg(settings.get, r_cfg) == "balanced"


def test_resolve_rendering_mode_cfg_rejects_non_str_mode_name():
    """Profile lookup keys must be str (guards against bad carb.get() types)."""
    log = __import__("logging").getLogger(__name__)
    assert resolve_rendering_mode_cfg({}, {"q": RenderingModeCfg()}, log) is None


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
