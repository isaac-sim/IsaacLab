# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched captures preserve corrected ranges while the camera is not due."""

from types import SimpleNamespace

import torch

from isaaclab_fovi.sensors.calibrated_camera import CalibratedTiledCamera


def test_held_frame_does_not_accumulate_near_distance() -> None:
    camera = SimpleNamespace()
    camera.cfg = SimpleNamespace(
        renderer_cfg=SimpleNamespace(renderer_type="ovrtx"),
        spawn=SimpleNamespace(clipping_range=(0.04, 20.0)),
    )
    camera._is_outdated = [True]
    camera._env_mask_has_any = lambda mask: mask[0]
    ranges = torch.tensor([[[[1.0], [2.0]]]])
    camera._data = SimpleNamespace(output={"distance_to_camera": SimpleNamespace(torch=ranges)})
    camera._invalid_pixels = torch.tensor([[[[False], [True]]]])
    CalibratedTiledCamera.process_rendered_frame(camera)
    torch.testing.assert_close(ranges, torch.tensor([[[[1.04], [20.0]]]]))
    camera._is_outdated[0] = False
    CalibratedTiledCamera.process_rendered_frame(camera)
    torch.testing.assert_close(ranges, torch.tensor([[[[1.04], [20.0]]]]))
