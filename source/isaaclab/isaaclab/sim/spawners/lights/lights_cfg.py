# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils import configclass

from . import lights


@configclass
class LightCfg(SpawnerCfg):
    """Configuration parameters for creating a light in the scene.

    Please refer to the documentation on `USD LuxLight <https://openusd.org/dev/api/class_usd_lux_light_a_p_i.html>`_
    for more information.

    .. note::
        The default values for the attributes are those specified in the their official documentation.
    """

    func: Callable = lights.spawn_light

    prim_type: str = MISSING
    """The prim type name for the light prim."""

    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """The color of emitted light, in energy-linear terms. Defaults to white."""

    enable_color_temperature: bool = False
    """Enables color temperature. Defaults to false."""

    color_temperature: float = 6500.0
    """Color temperature (in Kelvin) representing the white point. The valid range is [1000, 10000]. Defaults to 6500K.

    The `color temperature <https://en.wikipedia.org/wiki/Color_temperature>`_ corresponds to the warmth
    or coolness of light. Warmer light has a lower color temperature, while cooler light has a higher
    color temperature.

    Note:
        It only takes effect when :attr:`enable_color_temperature` is true.
    """

    normalize: bool = False
    """Normalizes power by the surface area of the light. Defaults to false.

    This makes it easier to independently adjust the power and shape of the light, by causing the power
    to not vary with the area or angular size of the light.
    """

    exposure: float = 0.0
    """Scales the power of the light exponentially as a power of 2. Defaults to 0.0.

    The result is multiplied against the intensity.
    """

    intensity: float = 1.0
    """Scales the power of the light linearly. Defaults to 1.0."""


@configclass
class DiskLightCfg(LightCfg):
    """Configuration parameters for creating a disk light in the scene.

    A disk light is a light source that emits light from a disk. It is useful for simulating
    fluorescent lights. For more information, please refer to the documentation on
    `USDLux DiskLight <https://openusd.org/dev/api/class_usd_lux_disk_light.html>`_.

    .. note::
        The default values for the attributes are those specified in the their official documentation.
    """

    prim_type = "DiskLight"

    radius: float = 0.5
    """Radius of the disk (in m). Defaults to 0.5m."""


@configclass
class DistantLightCfg(LightCfg):
    """Configuration parameters for creating a distant light in the scene.

    A distant light is a light source that is infinitely far away, and emits parallel rays of light.
    It is useful for simulating sun/moon light. For more information, please refer to the documentation on
    `USDLux DistantLight <https://openusd.org/dev/api/class_usd_lux_distant_light.html>`_.

    .. note::
        The default values for the attributes are those specified in the their official documentation.
    """

    prim_type = "DistantLight"

    angle: float = 0.53
    """Angular size of the light (in degrees). Defaults to 0.53 degrees.

    As an example, the Sun is approximately 0.53 degrees as seen from Earth.
    Higher values broaden the light and therefore soften shadow edges.
    """


@configclass
class DomeLightCfg(LightCfg):
    """Configuration parameters for creating a dome light in the scene.

    A dome light is a light source that emits light inwards from all directions. It is also possible to
    attach a texture to the dome light, which will be used to emit light. For more information, please refer
    to the documentation on `USDLux DomeLight <https://openusd.org/dev/api/class_usd_lux_dome_light.html>`_.

    .. note::
        The default values for the attributes are those specified in the their official documentation.
    """

    prim_type = "DomeLight"

    texture_file: str | None = None
    """A color texture to use on the dome, such as an HDR (high dynamic range) texture intended
    for IBL (image based lighting). Defaults to None.

    If None, the dome will emit a uniform color.
    """

    texture_format: Literal["automatic", "latlong", "mirroredBall", "angular", "cubeMapVerticalCross"] = "automatic"
    """The parametrization format of the color map file. Defaults to "automatic".

    Valid values are:

    * ``"automatic"``: Tries to determine the layout from the file itself. For example, Renderman texture files embed an explicit parameterization.
    * ``"latlong"``: Latitude as X, longitude as Y.
    * ``"mirroredBall"``: An image of the environment reflected in a sphere, using an implicitly orthogonal projection.
    * ``"angular"``: Similar to mirroredBall but the radial dimension is mapped linearly to the angle, providing better sampling at the edges.
    * ``"cubeMapVerticalCross"``: A cube map with faces laid out as a vertical cross.
    """

    visible_in_primary_ray: bool = True
    """Whether the dome light is visible in the primary ray. Defaults to True.

    If true, the texture in the sky is visible, otherwise the sky is black.
    """


@configclass
class CylinderLightCfg(LightCfg):
    """Configuration parameters for creating a cylinder light in the scene.

    A cylinder light is a light source that emits light from a cylinder. It is useful for simulating
    fluorescent lights. For more information, please refer to the documentation on
    `USDLux CylinderLight <https://openusd.org/dev/api/class_usd_lux_cylinder_light.html>`_.

    .. note::
        The default values for the attributes are those specified in the their official documentation.
    """

    prim_type = "CylinderLight"

    length: float = 1.0
    """Length of the cylinder (in m). Defaults to 1.0m."""

    radius: float = 0.5
    """Radius of the cylinder (in m). Defaults to 0.5m."""

    treat_as_line: bool = False
    """Treats the cylinder as a line source, i.e. a zero-radius cylinder. Defaults to false."""


@configclass
class SphereLightCfg(LightCfg):
    """Configuration parameters for creating a sphere light in the scene.

    A sphere light is a light source that emits light outward from a sphere. For more information,
    please refer to the documentation on
    `USDLux SphereLight <https://openusd.org/dev/api/class_usd_lux_sphere_light.html>`_.

    .. note::
        The default values for the attributes are those specified in the their official documentation.
    """

    prim_type = "SphereLight"

    radius: float = 0.5
    """Radius of the sphere. Defaults to 0.5m."""

    treat_as_point: bool = False
    """Treats the sphere as a point source, i.e. a zero-radius sphere. Defaults to false."""
