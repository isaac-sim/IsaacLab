# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Building blocks of active-gaze camera tasks: a foveated version of a stock camera, and the gaze observations."""

from __future__ import annotations

from dataclasses import MISSING, fields

from isaaclab_fovi import mdp
from isaaclab_fovi.sensors import FoveatedCameraCfg

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass


def foveated_camera(camera: CameraCfg, foveated_res: int | None = None) -> FoveatedCameraCfg:
    """A foveated camera at the camera's mount, with its optics, data types and renderer.

    The retina covers the camera's short side and is ``foveated_res`` samples across, by default the camera's width.
    Samples outside the camera's field of view are zero.
    The foveated camera keeps its own image size and per-frame pose updates, which render each new gaze.
    """
    # A copy, so the foveated camera shares no nested configuration (spawn, renderer, offset) with the preset's.
    camera = camera.copy()
    own = {"class_type", "width", "height", "update_latest_camera_pose"}
    values = {field.name: getattr(camera, field.name) for field in fields(CameraCfg) if field.name not in own}
    resolution = camera.width if foveated_res is None else foveated_res
    return FoveatedCameraCfg(**values, foveated_res=resolution, mask_to_sensor_fov=True)


@configclass
class GazeObservationsCfg(ObservationGroupCfg):
    """The gaze, in field radii, and its velocity, in field radii per second."""

    state = ObservationTermCfg(func=mdp.gaze_state)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class OracleObservationsCfg(ObservationGroupCfg):
    """The teacher's gaze labels. Training labels, never a policy input."""

    labels: ObservationTermCfg = MISSING

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
