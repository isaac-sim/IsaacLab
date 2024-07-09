# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for randomizations that are applied at the time of spawning prims."""

from __future__ import annotations

from dataclasses import MISSING

from pxr import Usd

from omni.isaac.lab.utils import configclass

from .randomizations import randomize_joint_offset, randomize_scale


@configclass
class RandomizationCfg:
    """Base class for randomization configurations."""

    func: callable[..., Usd.Prim] = MISSING


@configclass
class RandomizeJointOffsetsCfg(RandomizationCfg):
    """Randomize the joint offsets of a joint in an articulation
    The joint offsets are randomized in the x, y, and z axis by adding random value within the specified range.
    """

    x_range: tuple[float, float] = (0, 0)
    """The range of the randomization in the x-axis"""

    y_range: tuple[float, float] = (0, 0)
    """The range of the randomization in the y-axis"""

    z_range: tuple[float, float] = (0, 0)
    """The range of the randomization in the z-axis"""

    joint_name: str = MISSING
    """Joint name to randomize. Note that the joint prim is assumed to be located
    at <prim_path>/<joint_name>"""

    func: callable[..., Usd.Prim] = randomize_joint_offset


@configclass
class RandomizeScaleCfg(RandomizationCfg):
    """Randomize the scale of a prim.
    The scale is randomized in the x, y, and z axis by multiplying the current scale by a random value within the specified range.
    """

    x_range: tuple[float, float] = (1, 1)
    """The range of the randomization in the x-axis."""
    y_range: tuple[float, float] = (1, 1)
    """The range of the randomization in the y-axis."""
    z_range: tuple[float, float] = (1, 1)
    """The range of the randomization in the z-axis."""

    equal_scale: bool = False
    """If True, the sampled x_range is used to scale all axes."""

    func: callable[..., Usd.Prim] = randomize_scale
