# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Isaac RTX renderer utility functions."""

import logging
from unittest.mock import MagicMock, patch

import pytest
from isaaclab_physx.renderers.isaac_rtx_renderer_utils import (
    SIMPLE_SHADING_MODES,
    configure_isaac_rtx_settings,
    resolve_simple_shading_mode,
)
from packaging.version import Version

MODULE = "isaaclab_physx.renderers.isaac_rtx_renderer_utils"

# The production code branches on ``isaac_sim_version.major >= 6``.
# Use exact boundary values so the tests break loudly if the threshold changes.
_SIM_AT_THRESHOLD = Version("6.0.0")
_SIM_BELOW_THRESHOLD = Version("5.9.9")


class TestResolveSimpleShadingMode:
    """Tests for resolve_simple_shading_mode."""

    def test_returns_none_when_no_shading_requested(self):
        assert resolve_simple_shading_mode(["rgb", "depth"]) is None

    def test_returns_none_for_empty_data_types(self):
        assert resolve_simple_shading_mode([]) is None

    @pytest.mark.parametrize(
        "data_type,expected_mode",
        [
            ("simple_shading_constant_diffuse", 0),
            ("simple_shading_diffuse_mdl", 1),
            ("simple_shading_full_mdl", 2),
        ],
    )
    def test_returns_correct_mode_for_each_shading_type(self, data_type, expected_mode):
        assert resolve_simple_shading_mode([data_type]) == expected_mode

    def test_returns_correct_mode_mixed_with_other_types(self):
        assert resolve_simple_shading_mode(["depth", "simple_shading_full_mdl", "rgb"]) == 2

    def test_warns_and_picks_first_when_multiple_modes(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = resolve_simple_shading_mode(["simple_shading_constant_diffuse", "simple_shading_full_mdl"])
        assert result == SIMPLE_SHADING_MODES["simple_shading_constant_diffuse"]
        assert "Multiple simple shading modes requested" in caplog.text

    def test_no_warning_for_single_mode(self, caplog):
        with caplog.at_level(logging.WARNING):
            resolve_simple_shading_mode(["simple_shading_diffuse_mdl"])
        assert "Multiple simple shading modes requested" not in caplog.text


class TestConfigureIsaacRtxSettings:
    """Tests for configure_isaac_rtx_settings."""

    def test_noop_when_kit_unavailable(self):
        with patch(f"{MODULE}.has_kit", return_value=False) as mock_has_kit:
            configure_isaac_rtx_settings(["depth"])
            mock_has_kit.assert_called_once()

    @pytest.fixture()
    def _sim6_env(self):
        """Patch has_kit, version, and settings for Isaac Sim at the 6.0 threshold."""
        mock_settings = MagicMock()
        mock_settings.get.return_value = False
        with (
            patch(f"{MODULE}.has_kit", return_value=True),
            patch(f"{MODULE}.get_isaac_sim_version", return_value=_SIM_AT_THRESHOLD),
            patch(f"{MODULE}.get_settings_manager", return_value=mock_settings),
        ):
            yield mock_settings

    def test_disables_color_render_when_no_rgb(self, _sim6_env):
        settings = _sim6_env
        configure_isaac_rtx_settings(["depth", "normals"])
        settings.set_bool.assert_any_call("/rtx/sdg/force/disableColorRender", True)

    def test_does_not_disable_color_render_when_rgb_requested(self, _sim6_env):
        settings = _sim6_env
        configure_isaac_rtx_settings(["rgb", "depth"])
        calls = [c for c in settings.set_bool.call_args_list if c[0][0] == "/rtx/sdg/force/disableColorRender"]
        assert not calls

    def test_does_not_disable_color_render_when_rgba_requested(self, _sim6_env):
        settings = _sim6_env
        configure_isaac_rtx_settings(["rgba"])
        calls = [c for c in settings.set_bool.call_args_list if c[0][0] == "/rtx/sdg/force/disableColorRender"]
        assert not calls

    def test_gui_overrides_fast_path(self):
        mock_settings = MagicMock()
        mock_settings.get.return_value = True  # GUI active
        with (
            patch(f"{MODULE}.has_kit", return_value=True),
            patch(f"{MODULE}.get_isaac_sim_version", return_value=_SIM_AT_THRESHOLD),
            patch(f"{MODULE}.get_settings_manager", return_value=mock_settings),
        ):
            configure_isaac_rtx_settings(["depth"])
            mock_settings.set_bool.assert_any_call("/rtx/sdg/force/disableColorRender", False)

    @pytest.fixture()
    def _pre6_env(self):
        """Patch has_kit, version, and settings for Isaac Sim just below the 6.0 threshold."""
        mock_settings = MagicMock()
        with (
            patch(f"{MODULE}.has_kit", return_value=True),
            patch(f"{MODULE}.get_isaac_sim_version", return_value=_SIM_BELOW_THRESHOLD),
            patch(f"{MODULE}.get_settings_manager", return_value=mock_settings),
        ):
            yield mock_settings

    def test_warns_about_albedo_on_old_sim(self, _pre6_env, caplog):
        with caplog.at_level(logging.WARNING):
            configure_isaac_rtx_settings(["albedo"])
        assert "Albedo annotator is only supported in Isaac Sim 6.0+" in caplog.text

    def test_warns_about_simple_shading_on_old_sim(self, _pre6_env, caplog):
        with caplog.at_level(logging.WARNING):
            configure_isaac_rtx_settings(["simple_shading_constant_diffuse"])
        assert "Simple shading annotators are only supported in Isaac Sim 6.0+" in caplog.text

    def test_no_warning_for_supported_types_on_old_sim(self, _pre6_env, caplog):
        with caplog.at_level(logging.WARNING):
            configure_isaac_rtx_settings(["rgb", "depth"])
        assert caplog.text == ""

    def test_no_disableColorRender_on_old_sim(self, _pre6_env):
        settings = _pre6_env
        configure_isaac_rtx_settings(["depth"])
        calls = [c for c in settings.set_bool.call_args_list if c[0][0] == "/rtx/sdg/force/disableColorRender"]
        assert not calls
