# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated renderer method aliases still reach the sensor API."""

from __future__ import annotations

from typing import Any

import pytest

from isaaclab.renderers.base_renderer import BaseRenderer

pytestmark = pytest.mark.kitless


class _RendererStub(BaseRenderer):
    """Minimal backend so tests can override one pose-update method."""

    def __init__(self) -> None:
        super().__init__(object(), sim=object())
        self.received = None

    def supported_output_types(self) -> dict:
        return {}

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        return None

    def create_render_data(self, spec: Any) -> Any:
        return None

    def set_outputs(self, render_data: Any, output_data: dict) -> None:
        return None

    def update_transforms(self) -> None:
        return None

    def update_geometries(self) -> None:
        return None

    def render(self, render_data: Any) -> None:
        return None

    def read_output(self, render_data: Any, camera_data: Any) -> None:
        return None

    def cleanup(self, render_data: Any) -> None:
        return None


def test_update_camera_subclass_receives_update_sensor():
    """Legacy backend pose updates remain reachable through update_sensor."""

    class _LegacyRenderer(_RendererStub):
        def update_camera(
            self,
            render_data: Any,
            positions: Any,
            orientations: Any,
            intrinsics: Any,
        ) -> None:
            self.received = (render_data, positions, orientations, intrinsics)

    renderer = _LegacyRenderer()

    with pytest.warns(
        DeprecationWarning, match="_LegacyRenderer.update_camera is deprecated. Implement update_sensor instead."
    ):
        renderer.update_sensor("data", "pos", "quat", "K")

    assert renderer.received == ("data", "pos", "quat", "K")


def test_update_camera_forwards_to_subclass_update_sensor():
    """A backend that implements update_sensor still accepts the deprecated update_camera name."""

    class _ModernRenderer(_RendererStub):
        def update_sensor(
            self,
            render_data: Any,
            positions: Any,
            orientations: Any,
            intrinsics: Any | None = None,
        ) -> None:
            self.received = (render_data, positions, orientations, intrinsics)

    renderer = _ModernRenderer()

    with pytest.warns(
        DeprecationWarning, match="BaseRenderer.update_camera is deprecated. Call update_sensor instead."
    ):
        renderer.update_camera("data", "pos", "quat", "K")

    assert renderer.received == ("data", "pos", "quat", "K")


def test_legacy_extension_can_delegate_to_migrated_parent():
    """Camera extensions can adjust a pose and pass it to the inherited backend."""

    class Backend(_RendererStub):
        def update_sensor(self, render_data, positions, orientations, intrinsics=None):
            self.received = (render_data, positions, orientations, intrinsics)

    class Extension(Backend):
        def update_camera(self, render_data, positions, orientations, intrinsics):
            super().update_camera(render_data, positions + 1, orientations, intrinsics)

    renderer = Extension()
    with pytest.warns(DeprecationWarning):
        renderer.update_sensor("camera", 2, "quat", "K")
    assert renderer.received == ("camera", 3, "quat", "K")
