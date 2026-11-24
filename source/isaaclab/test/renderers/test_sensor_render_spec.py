# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SensorRenderSpec construction and CameraRenderSpec aliases."""

import pytest

from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.renderers.sensor_render_spec import SensorRenderSpec

pytestmark = pytest.mark.kitless


def test_sensor_render_spec_init():
    """Callers pass prim_paths and view_count positionally or by keyword."""
    # Test positional arguments.
    spec = SensorRenderSpec(object(), "cuda:0", 2, ("/World/Camera",), 3)
    assert spec.device == "cuda:0"
    assert spec.num_instances == 2
    assert spec.prim_paths == ("/World/Camera",)
    assert spec.view_count == 3

    # Test keyword arguments.
    spec = SensorRenderSpec(
        cfg=object(),
        device="cpu",
        num_instances=1,
        prim_paths=("/World/Cam",),
        view_count=4,
    )
    assert spec.device == "cpu"
    assert spec.num_instances == 1
    assert spec.prim_paths == ("/World/Cam",)
    assert spec.view_count == 4

    with pytest.raises(
        TypeError, match=r"SensorRenderSpec\.__init__\(\) missing 1 required positional argument: 'prim_paths'"
    ):
        spec = SensorRenderSpec(object(), "cpu", 1, view_count=1)

    with pytest.raises(
        TypeError, match=r"SensorRenderSpec\.__init__\(\) missing 1 required positional argument: 'view_count'"
    ):
        spec = SensorRenderSpec(object(), "cpu", 1, ("/World/Camera",))


def test_camera_render_spec_backwards_compatibility():
    """CameraRenderSpec is a deprecated SensorRenderSpec; camera_prim_paths still initializes and reads prim_paths."""
    with pytest.warns(DeprecationWarning, match="CameraRenderSpec is deprecated. Use SensorRenderSpec instead."):
        spec = CameraRenderSpec(object(), device="cpu", num_instances=2, prim_paths=("/World/Camera",), view_count=2)
    assert isinstance(spec, SensorRenderSpec)
    assert spec.prim_paths == ("/World/Camera",)

    assert spec.camera_prim_paths == spec.prim_paths

    with pytest.warns(DeprecationWarning, match="CameraRenderSpec is deprecated. Use SensorRenderSpec instead."):
        spec = CameraRenderSpec(
            object(), device="cpu", num_instances=1, camera_prim_paths=("/World/Cam",), view_count=1
        )
    assert spec.prim_paths == ("/World/Cam",)

    with pytest.warns(DeprecationWarning, match="CameraRenderSpec is deprecated. Use SensorRenderSpec instead."):
        with pytest.raises(TypeError, match="Pass only one of 'prim_paths' and 'camera_prim_paths'."):
            spec = CameraRenderSpec(object(), "cpu", 1, ("/World/A",), 1, camera_prim_paths=("/World/B",))
