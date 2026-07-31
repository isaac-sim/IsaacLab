# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests: the Newton cloner imports visual-only geometry only when it is drawn."""

import newton
from isaaclab_newton.cloner import newton_clone_utils
from isaaclab_newton.cloner import replicate as replicate_module
from isaaclab_newton.cloner.newton_clone_utils import build_source_builders
from newton import ShapeFlags

from pxr import Usd, UsdGeom, UsdPhysics

_SOURCE = "/World/Asset"


def _make_stage() -> Usd.Stage:
    """A rigid body with one collider cube and one visual-only cube."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    xform = UsdGeom.Xform.Define(stage, _SOURCE)
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())

    collider = UsdGeom.Cube.Define(stage, f"{_SOURCE}/collision")
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    UsdGeom.Cube.Define(stage, f"{_SOURCE}/visual")
    return stage


def _shape_counts(builder: newton.ModelBuilder) -> tuple[int, int]:
    """Number of (colliding, visual-only) shapes in the builder."""
    colliding = sum(1 for flags in builder.shape_flags if flags & ShapeFlags.COLLIDE_SHAPES)
    return colliding, len(builder.shape_flags) - colliding


class _CountingStage:
    """Stage proxy that counts the prim lookups a pass performs."""

    def __init__(self, stage: Usd.Stage):
        self._stage = stage
        self.prim_lookups = 0

    def GetPrimAtPath(self, path: str) -> Usd.Prim:  # noqa: N802  (USD API spelling)
        self.prim_lookups += 1
        return self._stage.GetPrimAtPath(path)


def _build(stage: Usd.Stage, **kwargs) -> newton.ModelBuilder:
    return build_source_builders(
        stage,
        [_SOURCE],
        create_builder=lambda: newton.ModelBuilder(up_axis=newton.Axis.Z),
        schema_resolvers=[],
        **kwargs,
    )[_SOURCE]


class TestClonerVisualShapeImport:
    """``build_source_builders`` must keep colliders regardless of the visual-shape flag."""

    def test_load_visual_shapes_imports_visual_only_geometry(self):
        """The default import keeps the visual-only cube alongside the collider."""
        colliding, visual_only = _shape_counts(_build(_make_stage(), load_visual_shapes=True))
        assert colliding == 1
        assert visual_only == 1

    def test_skipping_visual_shapes_keeps_colliders(self):
        """Disabling visual shapes drops the visual-only cube and nothing else."""
        colliding, visual_only = _shape_counts(_build(_make_stage(), load_visual_shapes=False))
        assert colliding == 1
        assert visual_only == 0

    def test_skipping_visual_shapes_skips_collider_visibility_resolution(self):
        """Without visual shapes no collider is hidden, so the restore pass must not run.

        The pass resolves USD visibility and purpose once per collider shape, which is pure
        overhead in a run that imports no visual geometry: the flags it would set are set
        already, and nothing draws them.
        """
        stage = _make_stage()
        builder = _build(stage, load_visual_shapes=False)
        path_shape_map = {f"{_SOURCE}/collision": 0}
        flags_before = list(builder.shape_flags)

        headless_stage = _CountingStage(stage)
        newton_clone_utils._restore_visible_colliders_without_visual_shapes(
            builder, headless_stage, path_shape_map, load_visual_shapes=False
        )
        assert headless_stage.prim_lookups == 0
        assert list(builder.shape_flags) == flags_before

        # The same call with the flag on does reach the stage, so the guard is what saved it.
        rendering_stage = _CountingStage(stage)
        newton_clone_utils._restore_visible_colliders_without_visual_shapes(
            builder, rendering_stage, path_shape_map, load_visual_shapes=True
        )
        assert rendering_stage.prim_lookups == 1
        # ...and the collider was visible either way, so skipping it changed nothing.
        assert list(builder.shape_flags) == flags_before


class _StubSim:
    def __init__(self, is_rendering: bool, can_render_rgb_array: bool, visual_shapes_required: bool = False):
        self.is_rendering = is_rendering
        self._can_render_rgb_array = can_render_rgb_array
        self.visual_shapes_required = visual_shapes_required

    def can_render_rgb_array(self) -> bool:
        return self._can_render_rgb_array


class TestRendererWantsVisualShapes:
    """The auto mode must follow whatever will actually draw the shapes."""

    def _patch_sim(self, monkeypatch, sim):
        import isaaclab.sim as sim_module

        monkeypatch.setattr(sim_module.SimulationContext, "instance", staticmethod(lambda: sim))

    def test_headless_run_skips_visual_shapes(self, monkeypatch):
        """Nothing rendering means nothing needs the visual-only shapes."""
        self._patch_sim(monkeypatch, _StubSim(is_rendering=False, can_render_rgb_array=False))
        assert replicate_module._renderer_wants_visual_shapes() is False

    def test_offscreen_capture_keeps_visual_shapes(self, monkeypatch):
        """An ``rgb_array`` capture draws the shapes even without a viewer window."""
        self._patch_sim(monkeypatch, _StubSim(is_rendering=False, can_render_rgb_array=True))
        assert replicate_module._renderer_wants_visual_shapes() is True

    def test_active_viewer_keeps_visual_shapes(self, monkeypatch):
        """A running viewer draws the shapes."""
        self._patch_sim(monkeypatch, _StubSim(is_rendering=True, can_render_rgb_array=False))
        assert replicate_module._renderer_wants_visual_shapes() is True

    def test_camera_sensor_keeps_visual_shapes(self, monkeypatch):
        """A camera on a non-Kit renderer draws the shapes without flipping any render setting."""
        self._patch_sim(
            monkeypatch, _StubSim(is_rendering=False, can_render_rgb_array=False, visual_shapes_required=True)
        )
        assert replicate_module._renderer_wants_visual_shapes() is True

    def test_missing_simulation_context_keeps_visual_shapes(self, monkeypatch):
        """Without a simulation context there is nothing to infer from, so import them."""
        self._patch_sim(monkeypatch, None)
        assert replicate_module._renderer_wants_visual_shapes() is True
