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
        NewtonManager._build_sdf_on_mesh(None, sdf_cfg, None, "test_label")

    def test_builds_sdf_with_max_resolution(self):
        """SDF is built on mesh with max_resolution and narrow_band_range."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        mesh = MagicMock()
        mesh.sdf = None
        sdf_cfg = self._make_sdf_cfg(max_resolution=128)

        NewtonManager._build_sdf_on_mesh(mesh, sdf_cfg, None, "test_label")

        mesh.build_sdf.assert_called_once_with(narrow_band_range=(-0.1, 0.1), max_resolution=128)

    def test_clears_existing_sdf_before_rebuild(self):
        """Existing SDF on mesh is cleared before building a new one."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        mesh = MagicMock()
        mesh.sdf = "existing_sdf"
        sdf_cfg = self._make_sdf_cfg()

        NewtonManager._build_sdf_on_mesh(mesh, sdf_cfg, None, "test_label")

        mesh.clear_sdf.assert_called_once()
        mesh.build_sdf.assert_called_once()

    def test_target_voxel_size_passed_alongside_resolution(self):
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
        builder.shape_type = [s["geo_type"] for s in shapes]
        builder.shape_body = [s["body_idx"] for s in shapes]
        builder.shape_label = [s["label"] for s in shapes]
        builder.shape_flags = [s["flags"] for s in shapes]
        builder.shape_source = [s.get("source") for s in shapes]
        builder.shape_margin = [0.0] * len(shapes)
        builder.shape_material_kh = [0.0] * len(shapes)
        return builder

    @staticmethod
    def _make_cfg(
        body_patterns=None,
        shape_patterns=None,
        max_resolution=256,
        k_hydro=None,
        hydroelastic_shape_patterns=None,
    ):
        cfg = MagicMock()
        cfg.sdf_cfg = MagicMock()
        cfg.sdf_cfg.max_resolution = max_resolution
        cfg.sdf_cfg.target_voxel_size = None
        cfg.sdf_cfg.narrow_band_range = (-0.1, 0.1)
        cfg.sdf_cfg.margin = None
        cfg.sdf_cfg.body_patterns = body_patterns
        cfg.sdf_cfg.shape_patterns = shape_patterns
        cfg.sdf_cfg.pattern_resolutions = None
        cfg.sdf_cfg.use_visual_meshes = False
        cfg.sdf_cfg.k_hydro = k_hydro
        cfg.sdf_cfg.hydroelastic_shape_patterns = hydroelastic_shape_patterns
        return cfg

    def test_no_sdf_cfg_is_noop(self):
        """_apply_sdf_config returns early when sdf_cfg is None."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        builder = MagicMock(spec=ModelBuilder)
        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as pm:
            pm._cfg = MagicMock()
            pm._cfg.sdf_cfg = None
            NewtonManager._apply_sdf_config(builder)
        assert not builder.method_calls

    def test_no_patterns_warns(self):
        """_apply_sdf_config warns when no patterns are set."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        builder = MagicMock(spec=ModelBuilder)
        cfg = self._make_cfg(body_patterns=None, shape_patterns=None)
        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as pm:
            pm._cfg = cfg
            with patch("isaaclab_newton.physics.newton_manager.logger") as mock_logger:
                NewtonManager._apply_sdf_config(builder)
                mock_logger.warning.assert_called()

    def test_body_pattern_collects_shapes(self):
        """Shapes under matching bodies are collected for SDF."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        bodies = ["/World/Robot/elbow", "/World/Robot/wrist"]
        shapes = [
            {
                "body_idx": 0,
                "label": "/World/Robot/elbow/col",
                "geo_type": GeoType.MESH,
                "flags": ShapeFlags.COLLIDE_SHAPES,
                "source": MagicMock(sdf=None),
            },
            {
                "body_idx": 1,
                "label": "/World/Robot/wrist/col",
                "geo_type": GeoType.MESH,
                "flags": ShapeFlags.COLLIDE_SHAPES,
                "source": MagicMock(sdf=None),
            },
        ]
        builder = self._make_builder(bodies, shapes)
        cfg = self._make_cfg(body_patterns=[".*elbow.*"])

        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as pm:
            pm._cfg = cfg
            NewtonManager._apply_sdf_config(builder)

        shapes[0]["source"].build_sdf.assert_called_once()
        shapes[1]["source"].build_sdf.assert_not_called()

    def test_hydroelastic_flag_set_when_k_hydro(self):
        """HYDROELASTIC flag is set on matched shapes when k_hydro is provided."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        bodies = ["/World/Robot/elbow"]
        shapes = [
            {
                "body_idx": 0,
                "label": "/World/Robot/elbow/col",
                "geo_type": GeoType.MESH,
                "flags": ShapeFlags.COLLIDE_SHAPES,
                "source": MagicMock(sdf=None),
            },
        ]
        builder = self._make_builder(bodies, shapes)
        cfg = self._make_cfg(body_patterns=[".*elbow.*"], k_hydro=1e10)

        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as pm:
            pm._cfg = cfg
            NewtonManager._apply_sdf_config(builder)

        assert builder.shape_flags[0] & ShapeFlags.HYDROELASTIC
        assert builder.shape_material_kh[0] == 1e10

    def test_hydroelastic_shape_patterns_filter(self):
        """hydroelastic_shape_patterns limits which shapes get HYDROELASTIC flag."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        bodies = ["/World/Robot/elbow", "/World/Robot/wrist"]
        shapes = [
            {
                "body_idx": 0,
                "label": "/World/Robot/elbow/col",
                "geo_type": GeoType.MESH,
                "flags": ShapeFlags.COLLIDE_SHAPES,
                "source": MagicMock(sdf=None),
            },
            {
                "body_idx": 1,
                "label": "/World/Robot/wrist/col",
                "geo_type": GeoType.MESH,
                "flags": ShapeFlags.COLLIDE_SHAPES,
                "source": MagicMock(sdf=None),
            },
        ]
        builder = self._make_builder(bodies, shapes)
        cfg = self._make_cfg(
            body_patterns=[".*"],
            k_hydro=1e10,
            hydroelastic_shape_patterns=[".*elbow.*"],
        )

        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as pm:
            pm._cfg = cfg
            NewtonManager._apply_sdf_config(builder)

        shapes[0]["source"].build_sdf.assert_called_once()
        shapes[1]["source"].build_sdf.assert_called_once()
        assert builder.shape_flags[0] & ShapeFlags.HYDROELASTIC
        assert not (builder.shape_flags[1] & ShapeFlags.HYDROELASTIC)

    def test_shape_pattern_matching(self):
        """shape_patterns directly matches shape labels."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        bodies = ["/World/Robot/body"]
        shapes = [
            {
                "body_idx": 0,
                "label": "/World/Robot/body/Gear_col",
                "geo_type": GeoType.MESH,
                "flags": ShapeFlags.COLLIDE_SHAPES,
                "source": MagicMock(sdf=None),
            },
            {
                "body_idx": 0,
                "label": "/World/Robot/body/frame_col",
                "geo_type": GeoType.MESH,
                "flags": ShapeFlags.COLLIDE_SHAPES,
                "source": MagicMock(sdf=None),
            },
        ]
        builder = self._make_builder(bodies, shapes)
        cfg = self._make_cfg(shape_patterns=[".*Gear.*"])

        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as pm:
            pm._cfg = cfg
            NewtonManager._apply_sdf_config(builder)

        shapes[0]["source"].build_sdf.assert_called_once()
        shapes[1]["source"].build_sdf.assert_not_called()

    def test_non_mesh_shapes_skipped(self):
        """Non-mesh shapes are never collected for SDF."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        bodies = ["/World/Robot/elbow"]
        shapes = [
            {
                "body_idx": 0,
                "label": "/World/Robot/elbow/box",
                "geo_type": GeoType.BOX,
                "flags": ShapeFlags.COLLIDE_SHAPES,
                "source": None,
            },
        ]
        builder = self._make_builder(bodies, shapes)
        cfg = self._make_cfg(body_patterns=[".*elbow.*"])

        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as pm:
            pm._cfg = cfg
            NewtonManager._apply_sdf_config(builder)


class TestCreateSdfCollisionFromVisual:
    """Tests for NewtonManager._create_sdf_collision_from_visual."""

    @staticmethod
    def _make_builder(bodies, shapes):
        """Create a minimal ModelBuilder-like mock."""
        builder = MagicMock(spec=ModelBuilder)
        builder.body_label = bodies
        builder.shape_count = len(shapes)
        builder.shape_type = [s["geo_type"] for s in shapes]
        builder.shape_body = [s["body_idx"] for s in shapes]
        builder.shape_label = [s["label"] for s in shapes]
        builder.shape_flags = [s["flags"] for s in shapes]
        builder.shape_source = [s.get("source") for s in shapes]
        builder.shape_transform = [s.get("xform", (0, 0, 0, 0, 0, 0, 1)) for s in shapes]
        builder.shape_scale = [s.get("scale", (1, 1, 1)) for s in shapes]
        builder.shape_margin = [0.0] * len(shapes)
        builder.shape_material_kh = [0.0] * len(shapes)
        return builder

    @staticmethod
    def _make_sdf_cfg(k_hydro=None, margin=None):
        cfg = MagicMock()
        cfg.max_resolution = 256
        cfg.target_voxel_size = None
        cfg.narrow_band_range = (-0.1, 0.1)
        cfg.margin = margin
        cfg.k_hydro = k_hydro
        return cfg

    def test_creates_collision_from_visual_mesh(self):
        """A body with only a visual mesh (no COLLIDE_SHAPES) gets a new collision shape."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        bodies = ["/World/Robot/arm"]
        mesh = MagicMock(sdf=None)
        shapes = [
            {
                "body_idx": 0,
                "label": "/World/Robot/arm/visual",
                "geo_type": GeoType.MESH,
                "flags": 0,
                "source": mesh,
            },
        ]
        builder = self._make_builder(bodies, shapes)
        sdf_cfg = self._make_sdf_cfg()

        num_added, num_hydro = NewtonManager._create_sdf_collision_from_visual(builder, {0}, sdf_cfg, None)

        assert num_added == 1
        assert num_hydro == 0
        builder.add_shape_mesh.assert_called_once()
        mesh.build_sdf.assert_called_once()

    def test_skips_body_with_existing_collision(self):
        """A body that already has a COLLIDE_SHAPES shape is not given a visual-mesh collision."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        bodies = ["/World/Robot/arm"]
        shapes = [
            {
                "body_idx": 0,
                "label": "/World/Robot/arm/collision",
                "geo_type": GeoType.MESH,
                "flags": ShapeFlags.COLLIDE_SHAPES,
                "source": MagicMock(sdf=None),
            },
        ]
        builder = self._make_builder(bodies, shapes)
        sdf_cfg = self._make_sdf_cfg()

        num_added, _ = NewtonManager._create_sdf_collision_from_visual(builder, {0}, sdf_cfg, None)

        assert num_added == 0
        builder.add_shape_mesh.assert_not_called()

    def test_warns_when_no_visual_mesh_source(self):
        """A body with no mesh source logs a warning and is skipped."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        bodies = ["/World/Robot/arm"]
        shapes = [
            {
                "body_idx": 0,
                "label": "/World/Robot/arm/visual",
                "geo_type": GeoType.MESH,
                "flags": 0,
                "source": None,
            },
        ]
        builder = self._make_builder(bodies, shapes)
        sdf_cfg = self._make_sdf_cfg()

        with patch("isaaclab_newton.physics.newton_manager.logger") as mock_logger:
            num_added, _ = NewtonManager._create_sdf_collision_from_visual(builder, {0}, sdf_cfg, None)
            mock_logger.warning.assert_called()

        assert num_added == 0

    def test_k_hydro_sets_hydroelastic_on_new_shape(self):
        """When k_hydro is set, the new collision shape gets is_hydroelastic=True."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        bodies = ["/World/Robot/arm"]
        mesh = MagicMock(sdf=None)
        shapes = [
            {
                "body_idx": 0,
                "label": "/World/Robot/arm/visual",
                "geo_type": GeoType.MESH,
                "flags": 0,
                "source": mesh,
            },
        ]
        builder = self._make_builder(bodies, shapes)
        sdf_cfg = self._make_sdf_cfg(k_hydro=1e10)

        num_added, num_hydro = NewtonManager._create_sdf_collision_from_visual(builder, {0}, sdf_cfg, None)

        assert num_added == 1
        assert num_hydro == 1
        call_kwargs = builder.add_shape_mesh.call_args[1]
        assert call_kwargs["cfg"].is_hydroelastic is True

    def test_hydro_patterns_filter_respected(self):
        """hydroelastic patterns filter which visual-mesh shapes get HYDROELASTIC."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        bodies = ["/World/Robot/arm", "/World/Robot/leg"]
        mesh_arm = MagicMock(sdf=None)
        mesh_leg = MagicMock(sdf=None)
        shapes = [
            {
                "body_idx": 0,
                "label": "/World/Robot/arm/visual",
                "geo_type": GeoType.MESH,
                "flags": 0,
                "source": mesh_arm,
            },
            {
                "body_idx": 1,
                "label": "/World/Robot/leg/visual",
                "geo_type": GeoType.MESH,
                "flags": 0,
                "source": mesh_leg,
            },
        ]
        builder = self._make_builder(bodies, shapes)
        sdf_cfg = self._make_sdf_cfg(k_hydro=1e10)
        hydro_patterns = [re.compile(".*arm.*")]

        num_added, num_hydro = NewtonManager._create_sdf_collision_from_visual(
            builder, {0, 1}, sdf_cfg, None, hydro_patterns
        )

        assert num_added == 2
        assert num_hydro == 1  # only arm matches hydro pattern
