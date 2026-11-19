# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton's renderer requires whole native-grid tiles."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab_fovi.sensors.foveated_newton_camera import FoveatedNewtonCamera

from isaaclab.sensors.camera.camera import Camera

pytestmark = pytest.mark.kitless


def test_native_grid_render_size_must_fill_newton_tiles():
    camera = SimpleNamespace(
        cfg=SimpleNamespace(
            width=83,
            height=83,
            renderer_cfg=SimpleNamespace(render_order="tiled", tile_rendering_width=16, tile_rendering_height=8),
        )
    )
    with pytest.raises(ValueError, match="83x83.*16x8"):
        FoveatedNewtonCamera._initialize_impl(camera)


def test_runtime_intrinsics_leave_the_retinas_rays_alone(monkeypatch) -> None:
    """The retina sets a foveated camera's rays, so runtime intrinsics, such as a viewer's zoom sets, leave the camera
    as configured with a warning to the caller, rather than reaching the renderer, which would replace the rays with a
    pinhole's."""

    def replace_rays(*args, **kwargs):
        raise AssertionError("the stock setter replaces the renderer's rays")

    monkeypatch.setattr(Camera, "set_intrinsic_matrices", replace_rays)
    monkeypatch.setattr(Camera, "__del__", lambda self: None)
    camera = object.__new__(FoveatedNewtonCamera)
    with pytest.warns(UserWarning, match="leaves a foveated camera as configured") as caught:
        camera.set_intrinsic_matrices(torch.eye(3).repeat(2, 1, 1))
    # The warning points at the caller's line.
    assert caught[0].filename == __file__
