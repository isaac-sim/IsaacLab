# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Building blocks of active-gaze camera tasks: a foveated version of a stock camera, and the gaze observations."""

from __future__ import annotations

from dataclasses import MISSING, fields

from isaaclab_fovi import mdp
from isaaclab_fovi.sensors import FoveatedCameraCfg

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg, preset


def foveated_camera(camera: CameraCfg, foveated_res: int | None = None) -> FoveatedCameraCfg:
    """A foveated camera at the camera's mount, with its optics, data types and renderer.

    The retina covers the camera's short side and is ``foveated_res`` samples across, by default the camera's width,
    so the number of rendered rays stays close to the camera's pixels. Samples the camera does not see are zero.
    The foveated camera keeps its own image size and per-frame pose updates, which render each new gaze.
    """
    # A copy, so the foveated camera shares no nested configuration (spawn, renderer, offset) with the preset's.
    camera = camera.copy()
    own = {"class_type", "width", "height", "update_latest_camera_pose"}
    values = {field.name: getattr(camera, field.name) for field in fields(CameraCfg) if field.name not in own}
    resolution = camera.width if foveated_res is None else foveated_res
    return FoveatedCameraCfg(**values, foveated_res=resolution, mask_to_sensor_fov=True)


def foveated_presets(cameras: PresetCfg, foveated_res: int | None = None) -> PresetCfg:
    """The foveated version of every camera preset, under the same names. Other sensors are left out.

    ``foveated_res`` applies to every preset; by default each keeps its camera's width.
    """
    options = {}
    for name in cameras.__dataclass_fields__:
        camera = getattr(cameras, name)
        if not name.startswith("_") and isinstance(camera, CameraCfg):
            options[name] = foveated_camera(camera, foveated_res)
    return preset(**options)


def select_presets(options: PresetCfg, names) -> PresetCfg:
    """The options of a preset whose names are in ``names``, plus its default."""
    return preset(
        **{name: getattr(options, name) for name in options.__dataclass_fields__ if name in {*names, "default"}}
    )


def oracle_observations(
    target_cfg: SceneEntityCfg, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> OracleObservationsCfg:
    """The teacher's gaze labels toward a point fixed to an asset."""
    return OracleObservationsCfg(
        labels=ObservationTermCfg(func=mdp.gaze_teacher_labels, params={"target_cfg": target_cfg, "offset": offset})
    )


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
