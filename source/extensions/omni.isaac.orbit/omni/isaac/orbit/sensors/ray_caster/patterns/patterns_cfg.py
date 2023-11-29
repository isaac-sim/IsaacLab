# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast sensor."""

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import Callable, Sequence

from omni.isaac.orbit.utils import configclass

from . import patterns


@configclass
class PatternBaseCfg:
    """Base configuration for a pattern."""

    func: Callable[[PatternBaseCfg, str], tuple[torch.Tensor, torch.Tensor]] = MISSING
    """Function to generate the pattern.

    The function should take in the configuration and the device name as arguments. It should return
    the pattern's starting positions and directions as a tuple of torch.Tensor.
    """


@configclass
class GridPatternCfg(PatternBaseCfg):
    """Configuration for the grid pattern for ray-casting.

    Defines a 2D grid of rays in the coordinates of the sensor.
    """

    func: Callable = patterns.grid_pattern

    resolution: float = MISSING
    """Grid resolution (in meters)."""
    size: tuple[float, float] = MISSING
    """Grid size (length, width) (in meters)."""
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    """Ray direction. Defaults to (0.0, 0.0, -1.0)."""


@configclass
class PinholeCameraPatternCfg(PatternBaseCfg):
    """Configuration for a pinhole camera depth image pattern for ray-casting."""

    func: Callable = patterns.pinhole_camera_pattern

    focal_length: float = 24.0
    """Perspective focal length (in cm). Defaults to 24.0cm.

    Longer lens lengths narrower FOV, shorter lens lengths wider FOV.
    """
    horizontal_aperture: float = 20.955
    """Horizontal aperture (in mm). Defaults to 20.955mm.

    Emulates sensor/film width on a camera.

    Note:
        The default value is the horizontal aperture of a 35 mm spherical projector.
    """
    horizontal_aperture_offset: float = 0.0
    """Offsets Resolution/Film gate horizontally. Defaults to 0.0."""
    vertical_aperture_offset: float = 0.0
    """Offsets Resolution/Film gate vertically. Defaults to 0.0."""
    width: int = MISSING
    """Width of the image (in pixels)."""
    height: int = MISSING
    """Height of the image (in pixels)."""


@configclass
class BpearlPatternCfg(PatternBaseCfg):
    """Configuration for the Bpearl pattern for ray-casting."""

    func: Callable = patterns.bpearl_pattern

    horizontal_fov: float = 360.0
    """Horizontal field of view (in degrees). Defaults to 360.0."""

    horizontal_res: float = 10.0
    """Horizontal resolution (in degrees). Defaults to 10.0."""

    # fmt: off
    vertical_ray_angles: Sequence[float] = [
        89.5, 86.6875, 83.875, 81.0625, 78.25, 75.4375, 72.625, 69.8125, 67.0, 64.1875, 61.375,
        58.5625, 55.75, 52.9375, 50.125, 47.3125, 44.5, 41.6875, 38.875, 36.0625, 33.25, 30.4375,
        27.625, 24.8125, 22, 19.1875, 16.375, 13.5625, 10.75, 7.9375, 5.125, 2.3125
    ]
    # fmt: on
    """Vertical ray angles (in degrees). Defaults to a list of 32 angles.

    Note:
        We manually set the vertical ray angles to match the Bpearl sensor. The ray-angles
        are not evenly spaced.
    """
