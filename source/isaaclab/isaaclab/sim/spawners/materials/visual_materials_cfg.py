# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.utils import configclass

from . import visual_materials


@configclass
class VisualMaterialCfg:
    """Configuration parameters for creating a visual material."""

    func: Callable = MISSING
    """The function to use for creating the material."""


@configclass
class PreviewSurfaceCfg(VisualMaterialCfg):
    """Configuration parameters for creating a preview surface.

    See :meth:`spawn_preview_surface` for more information.
    """

    func: Callable = visual_materials.spawn_preview_surface

    diffuse_color: tuple[float, float, float] = (0.18, 0.18, 0.18)
    """The RGB diffusion color. This is the base color of the surface. Defaults to a dark gray."""
    emissive_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """The RGB emission component of the surface. Defaults to black."""
    roughness: float = 0.5
    """The roughness for specular lobe. Ranges from 0 (smooth) to 1 (rough). Defaults to 0.5."""
    metallic: float = 0.0
    """The metallic component. Ranges from 0 (dielectric) to 1 (metal). Defaults to 0."""
    opacity: float = 1.0
    """The opacity of the surface. Ranges from 0 (transparent) to 1 (opaque). Defaults to 1.

    Note:
        Opacity only affects the surface's appearance during interactive rendering.
    """


@configclass
class MdlFileCfg(VisualMaterialCfg):
    """Configuration parameters for loading an MDL material from a file.

    See :meth:`spawn_from_mdl_file` for more information.
    """

    func: Callable = visual_materials.spawn_from_mdl_file

    mdl_path: str = MISSING
    """The path to the MDL material.

    NVIDIA Omniverse provides various MDL materials in the NVIDIA Nucleus.
    To use these materials, you can set the path of the material in the nucleus directory
    using the ``{NVIDIA_NUCLEUS_DIR}`` variable. This is internally resolved to the path of the
    NVIDIA Nucleus directory on the host machine through the attribute
    :attr:`isaaclab.utils.assets.NVIDIA_NUCLEUS_DIR`.

    For example, to use the "Aluminum_Anodized" material, you can set the path to:
    ``{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized.mdl``.
    """
    project_uvw: bool | None = None
    """Whether to project the UVW coordinates of the material. Defaults to None.

    If None, then the default setting in the MDL material will be used.
    """
    albedo_brightness: float | None = None
    """Multiplier for the diffuse color of the material. Defaults to None.

    If None, then the default setting in the MDL material will be used.
    """
    texture_scale: tuple[float, float] | None = None
    """The scale of the texture. Defaults to None.

    If None, then the default setting in the MDL material will be used.
    """


@configclass
class GlassMdlCfg(VisualMaterialCfg):
    """Configuration parameters for loading a glass MDL material.

    This is a convenience class for loading a glass MDL material. For more information on
    glass materials, see the `documentation <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/materials.html#omniglass>`__.

    .. note::
        The default values are taken from the glass material in the NVIDIA Nucleus.
    """

    func: Callable = visual_materials.spawn_from_mdl_file

    mdl_path: str = "OmniGlass.mdl"
    """The path to the MDL material. Defaults to the glass material in the NVIDIA Nucleus."""
    glass_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """The RGB color or tint of the glass. Defaults to white."""
    frosting_roughness: float = 0.0
    """The amount of reflectivity of the surface. Ranges from 0 (perfectly clear) to 1 (frosted).
    Defaults to 0."""
    thin_walled: bool = False
    """Whether to perform thin-walled refraction. Defaults to False."""
    glass_ior: float = 1.491
    """The incidence of refraction to control how much light is bent when passing through the glass.
    Defaults to 1.491, which is the IOR of glass.
    """
