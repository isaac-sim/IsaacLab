# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for SDF collision configuration and application logic."""

import re
from unittest.mock import MagicMock, patch

from newton import GeoType, ModelBuilder, ShapeFlags


class TestBuildSdfOnMesh:
    """Tests for NewtonManager._build_sdf_on_mesh."""

    @staticmethod
    def _make_sdf_cfg(max_resolution=256, narrow_band_range=(-0.1, 0.1), target_voxel_size=None):
        cfg = MagicMock()
        cfg.max_resolution = max_resolution
        cfg.narrow_band_range = narrow_band_range
        cfg.target_voxel_size = target_voxel_size
        return cfg

    def test_none_mesh_is_noop(self):
        """Passing None as mesh should not raise."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        sdf_cfg = self._make_sdf_cfg()
        # Should not raise
        NewtonManager._build_sdf_on_mesh(None, sdf_cfg, None, "test_label")

    def test_builds_sdf_with_max_resolution(self):
        """SDF is built on mesh with max_resolution and narrow_band_range."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        mesh = MagicMock()
        mesh.sdf = None
        sdf_cfg = self._make_sdf_cfg(max_resolution=128)

        NewtonManager._build_sdf_on_mesh(mesh, sdf_cfg, None, "test_label")

        mesh.build_sdf.assert_called_once_with(narrow_band_range=(-0.1, 0.1), max_resolution=128)

    def test_skips_rebuild_when_sdf_already_exists(self):
        """Existing SDF on mesh is preserved (shared by reference from prototype)."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        mesh = MagicMock()
        mesh.sdf = "existing_sdf"
        sdf_cfg = self._make_sdf_cfg()

        NewtonManager._build_sdf_on_mesh(mesh, sdf_cfg, None, "test_label")

        mesh.clear_sdf.assert_not_called()
        mesh.build_sdf.assert_not_called()

    def test_target_voxel_size_takes_precedence(self):
        """When target_voxel_size is set, it is passed alongside max_resolution."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        mesh = MagicMock()
        mesh.sdf = None
        sdf_cfg = self._make_sdf_cfg(max_resolution=256, target_voxel_size=0.005)

        NewtonManager._build_sdf_on_mesh(mesh, sdf_cfg, None, "test_label")

        call_kwargs = mesh.build_sdf.call_args[1]
        assert call_kwargs["target_voxel_size"] == 0.005
        assert call_kwargs["max_resolution"] == 256

    def test_resolution_override_by_pattern(self):
        """Per-pattern resolution override is applied when label matches."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        mesh = MagicMock()
        mesh.sdf = None
        sdf_cfg = self._make_sdf_cfg(max_resolution=256)
        res_overrides = [(re.compile(".*elbow.*"), 128)]

        NewtonManager._build_sdf_on_mesh(mesh, sdf_cfg, res_overrides, "/World/Robot/elbow_link/collision")

        call_kwargs = mesh.build_sdf.call_args[1]
        assert call_kwargs["max_resolution"] == 128

    def test_resolution_override_no_match_uses_global(self):
        """When label doesn't match any override, global max_resolution is used."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        mesh = MagicMock()
        mesh.sdf = None
        sdf_cfg = self._make_sdf_cfg(max_resolution=256)
        res_overrides = [(re.compile(".*elbow.*"), 128)]

        NewtonManager._build_sdf_on_mesh(mesh, sdf_cfg, res_overrides, "/World/Robot/wrist_link/collision")

        call_kwargs = mesh.build_sdf.call_args[1]
        assert call_kwargs["max_resolution"] == 256

    def test_resolution_override_first_match_wins(self):
        """First matching pattern in res_overrides determines resolution."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        mesh = MagicMock()
        mesh.sdf = None
        sdf_cfg = self._make_sdf_cfg(max_resolution=256)
        res_overrides = [
            (re.compile(".*link.*"), 64),
            (re.compile(".*elbow.*"), 128),
        ]

        NewtonManager._build_sdf_on_mesh(mesh, sdf_cfg, res_overrides, "/World/Robot/elbow_link/collision")

        call_kwargs = mesh.build_sdf.call_args[1]
        assert call_kwargs["max_resolution"] == 64  # ".*link.*" matches first


class TestApplySdfConfig:
    """Tests for NewtonManager._apply_sdf_config shape index collection and patching."""

    @staticmethod
    def _make_builder(bodies, shapes):
        """Create a minimal ModelBuilder-like mock.

        Args:
            bodies: List of body label strings.
            shapes: List of dicts with keys: body_idx, label, geo_type, flags, source.
        """
        builder = MagicMock(spec=ModelBuilder)
        builder.body_label = bodies
        builder.shape_count = len(shapes)
        builder.shape_body = [s["body_idx"] for s in shapes]
        builder.shape_label = [s["label"] for s in shapes]
        builder.shape_type = [s["geo_type"] for s in shapes]
        builder.shape_flags = [s["flags"] for s in shapes]
        builder.shape_source = [s.get("source", MagicMock()) for s in shapes]
        builder.shape_transform = [None] * len(shapes)
        builder.shape_scale = [None] * len(shapes)
        builder.shape_margin = [0.0] * len(shapes)
        builder.shape_material_kh = [0.0] * len(shapes)
        return builder

    def test_sdf_cfg_none_is_noop(self):
        """When sdf_cfg is None, _apply_sdf_config does nothing."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        builder = MagicMock()
        with patch.object(NewtonManager, "_build_sdf_on_mesh") as mock_build:
            with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as mock_pm:
                mock_pm._cfg = MagicMock()
                mock_pm._cfg.sdf_cfg = None
                NewtonManager._apply_sdf_config(builder)
        mock_build.assert_not_called()

    def test_no_patterns_warns_and_returns(self):
        """When both body_patterns and shape_patterns are None, warns and returns."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        builder = MagicMock()
        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as mock_pm:
            sdf_cfg = MagicMock()
            sdf_cfg.max_resolution = 256
            sdf_cfg.target_voxel_size = None
            sdf_cfg.body_patterns = None
            sdf_cfg.shape_patterns = None
            sdf_cfg.hydroelastic_cfg = None
            mock_pm._cfg = MagicMock()
            mock_pm._cfg.sdf_cfg = sdf_cfg

            with patch("isaaclab_newton.physics.newton_manager.logger") as mock_logger:
                NewtonManager._apply_sdf_config(builder)
                mock_logger.warning.assert_called_once()
                assert "no body_patterns or shape_patterns" in mock_logger.warning.call_args[0][0]

    def test_no_resolution_warns_and_returns(self):
        """When neither max_resolution nor target_voxel_size is set, warns and returns."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        builder = MagicMock()
        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as mock_pm:
            sdf_cfg = MagicMock()
            sdf_cfg.max_resolution = None
            sdf_cfg.target_voxel_size = None
            mock_pm._cfg = MagicMock()
            mock_pm._cfg.sdf_cfg = sdf_cfg

            with patch("isaaclab_newton.physics.newton_manager.logger") as mock_logger:
                NewtonManager._apply_sdf_config(builder)
                mock_logger.warning.assert_called_once()
                assert "neither max_resolution nor target_voxel_size" in mock_logger.warning.call_args[0][0]

    def test_shape_patterns_match_shape_label(self):
        """shape_patterns should match against shape_label, not body_label."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        builder = self._make_builder(
            bodies=["/World/Robot/base", "/World/Robot/arm"],
            shapes=[
                {
                    "body_idx": 0,
                    "label": "/World/Robot/base/gear_mesh",
                    "geo_type": GeoType.MESH,
                    "flags": ShapeFlags.COLLIDE_SHAPES,
                },
                {
                    "body_idx": 1,
                    "label": "/World/Robot/arm/plain_mesh",
                    "geo_type": GeoType.MESH,
                    "flags": ShapeFlags.COLLIDE_SHAPES,
                },
            ],
        )
        # Mock the mesh sources
        for s in builder.shape_source:
            s.sdf = None

        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as mock_pm:
            sdf_cfg = MagicMock()
            sdf_cfg.max_resolution = 256
            sdf_cfg.target_voxel_size = None
            sdf_cfg.body_patterns = None
            sdf_cfg.shape_patterns = [".*gear.*"]
            sdf_cfg.pattern_resolutions = None
            sdf_cfg.margin = None
            sdf_cfg.use_visual_meshes = False
            sdf_cfg.hydroelastic_cfg = None
            sdf_cfg.narrow_band_range = (-0.1, 0.1)
            mock_pm._cfg = MagicMock()
            mock_pm._cfg.sdf_cfg = sdf_cfg

            NewtonManager._apply_sdf_config(builder)

        # Only gear_mesh should have had SDF built
        builder.shape_source[0].build_sdf.assert_called_once()
        builder.shape_source[1].build_sdf.assert_not_called()

    def test_body_patterns_match_body_label(self):
        """body_patterns should match against body_label and apply SDF to all shapes under that body."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        builder = self._make_builder(
            bodies=["/World/Robot/arm", "/World/Robot/base"],
            shapes=[
                {
                    "body_idx": 0,
                    "label": "/World/Robot/arm/mesh_a",
                    "geo_type": GeoType.MESH,
                    "flags": ShapeFlags.COLLIDE_SHAPES,
                },
                {
                    "body_idx": 0,
                    "label": "/World/Robot/arm/mesh_b",
                    "geo_type": GeoType.MESH,
                    "flags": ShapeFlags.COLLIDE_SHAPES,
                },
                {
                    "body_idx": 1,
                    "label": "/World/Robot/base/mesh_c",
                    "geo_type": GeoType.MESH,
                    "flags": ShapeFlags.COLLIDE_SHAPES,
                },
            ],
        )
        for s in builder.shape_source:
            s.sdf = None

        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as mock_pm:
            sdf_cfg = MagicMock()
            sdf_cfg.max_resolution = 256
            sdf_cfg.target_voxel_size = None
            sdf_cfg.body_patterns = [".*arm.*"]
            sdf_cfg.shape_patterns = None
            sdf_cfg.pattern_resolutions = None
            sdf_cfg.margin = None
            sdf_cfg.use_visual_meshes = False
            sdf_cfg.hydroelastic_cfg = None
            sdf_cfg.narrow_band_range = (-0.1, 0.1)
            mock_pm._cfg = MagicMock()
            mock_pm._cfg.sdf_cfg = sdf_cfg

            NewtonManager._apply_sdf_config(builder)

        # Both arm shapes should get SDF, base should not
        builder.shape_source[0].build_sdf.assert_called_once()
        builder.shape_source[1].build_sdf.assert_called_once()
        builder.shape_source[2].build_sdf.assert_not_called()

    def test_non_mesh_shapes_are_skipped(self):
        """Non-MESH shapes (e.g., BOX, SPHERE) should never get SDF even if patterns match."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        builder = self._make_builder(
            bodies=["/World/Robot/base"],
            shapes=[
                {
                    "body_idx": 0,
                    "label": "/World/Robot/base/box",
                    "geo_type": GeoType.BOX,
                    "flags": ShapeFlags.COLLIDE_SHAPES,
                },
                {
                    "body_idx": 0,
                    "label": "/World/Robot/base/mesh",
                    "geo_type": GeoType.MESH,
                    "flags": ShapeFlags.COLLIDE_SHAPES,
                },
            ],
        )
        for s in builder.shape_source:
            s.sdf = None

        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as mock_pm:
            sdf_cfg = MagicMock()
            sdf_cfg.max_resolution = 256
            sdf_cfg.target_voxel_size = None
            sdf_cfg.body_patterns = [".*base.*"]
            sdf_cfg.shape_patterns = None
            sdf_cfg.pattern_resolutions = None
            sdf_cfg.margin = None
            sdf_cfg.use_visual_meshes = False
            sdf_cfg.hydroelastic_cfg = None
            sdf_cfg.narrow_band_range = (-0.1, 0.1)
            mock_pm._cfg = MagicMock()
            mock_pm._cfg.sdf_cfg = sdf_cfg

            NewtonManager._apply_sdf_config(builder)

        # Box should not get SDF, mesh should
        builder.shape_source[0].build_sdf.assert_not_called()
        builder.shape_source[1].build_sdf.assert_called_once()

    def test_visual_shapes_skipped_when_use_visual_meshes_false(self):
        """Shapes without COLLIDE_SHAPES flag should not be patched when use_visual_meshes=False."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        builder = self._make_builder(
            bodies=["/World/Robot/arm"],
            shapes=[
                {
                    "body_idx": 0,
                    "label": "/World/Robot/arm/visual",
                    "geo_type": GeoType.MESH,
                    "flags": ShapeFlags.VISIBLE,
                },
            ],
        )
        for s in builder.shape_source:
            s.sdf = None

        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as mock_pm:
            sdf_cfg = MagicMock()
            sdf_cfg.max_resolution = 256
            sdf_cfg.target_voxel_size = None
            sdf_cfg.body_patterns = [".*arm.*"]
            sdf_cfg.shape_patterns = None
            sdf_cfg.pattern_resolutions = None
            sdf_cfg.margin = None
            sdf_cfg.use_visual_meshes = False
            sdf_cfg.hydroelastic_cfg = None
            sdf_cfg.narrow_band_range = (-0.1, 0.1)
            mock_pm._cfg = MagicMock()
            mock_pm._cfg.sdf_cfg = sdf_cfg

            NewtonManager._apply_sdf_config(builder)

        # Visual-only shape should not get SDF built
        builder.shape_source[0].build_sdf.assert_not_called()

    def test_margin_applied_when_set(self):
        """When sdf_cfg.margin is set, it should be applied to matched collision shapes."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        builder = self._make_builder(
            bodies=["/World/Robot/arm"],
            shapes=[
                {
                    "body_idx": 0,
                    "label": "/World/Robot/arm/mesh",
                    "geo_type": GeoType.MESH,
                    "flags": ShapeFlags.COLLIDE_SHAPES,
                },
            ],
        )
        for s in builder.shape_source:
            s.sdf = None

        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as mock_pm:
            sdf_cfg = MagicMock()
            sdf_cfg.max_resolution = 256
            sdf_cfg.target_voxel_size = None
            sdf_cfg.body_patterns = [".*arm.*"]
            sdf_cfg.shape_patterns = None
            sdf_cfg.pattern_resolutions = None
            sdf_cfg.margin = 0.02
            sdf_cfg.use_visual_meshes = False
            sdf_cfg.hydroelastic_cfg = None
            sdf_cfg.narrow_band_range = (-0.1, 0.1)
            mock_pm._cfg = MagicMock()
            mock_pm._cfg.sdf_cfg = sdf_cfg

            NewtonManager._apply_sdf_config(builder)

        assert builder.shape_margin[0] == 0.02

    def test_margin_none_leaves_default(self):
        """When sdf_cfg.margin is None, shape_margin should not be modified."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        builder = self._make_builder(
            bodies=["/World/Robot/arm"],
            shapes=[
                {
                    "body_idx": 0,
                    "label": "/World/Robot/arm/mesh",
                    "geo_type": GeoType.MESH,
                    "flags": ShapeFlags.COLLIDE_SHAPES,
                },
            ],
        )
        for s in builder.shape_source:
            s.sdf = None
        original_margin = builder.shape_margin[0]

        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as mock_pm:
            sdf_cfg = MagicMock()
            sdf_cfg.max_resolution = 256
            sdf_cfg.target_voxel_size = None
            sdf_cfg.body_patterns = [".*arm.*"]
            sdf_cfg.shape_patterns = None
            sdf_cfg.pattern_resolutions = None
            sdf_cfg.margin = None
            sdf_cfg.use_visual_meshes = False
            sdf_cfg.hydroelastic_cfg = None
            sdf_cfg.narrow_band_range = (-0.1, 0.1)
            mock_pm._cfg = MagicMock()
            mock_pm._cfg.sdf_cfg = sdf_cfg

            NewtonManager._apply_sdf_config(builder)

        assert builder.shape_margin[0] == original_margin
