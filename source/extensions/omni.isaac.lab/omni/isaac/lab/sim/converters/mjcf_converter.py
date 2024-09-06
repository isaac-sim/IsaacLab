# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import omni.kit.commands
import omni.usd
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.version import get_version
from pxr import Usd

from .asset_converter_base import AssetConverterBase
from .mjcf_converter_cfg import MjcfConverterCfg



class MjcfConverter(AssetConverterBase):
    """Converter for a MJCF description file to a USD file.

    This class wraps around the `omni.isaac.mjcf_importer`_ extension to provide a lazy implementation
    for MJCF to USD conversion. It stores the output USD file in an instanceable format since that is
    what is typically used in all learning related applications.

    .. caution::
        The current lazy conversion implementation does not automatically trigger USD generation if
        only the mesh files used by the MJCF are modified. To force generation, either set
        :obj:`AssetConverterBaseCfg.force_usd_conversion` to True or delete the output directory.

    .. note::
        From Isaac Sim 2023.1 onwards, the extension name changed from ``omni.isaac.mjcf`` to
        ``omni.importer.mjcf``. This converter class automatically detects the version of Isaac Sim
        and uses the appropriate extension.

    .. _omni.isaac.mjcf_importer: https://docs.omniverse.nvidia.com/isaacsim/latest/ext_omni_isaac_mjcf.html
    """

    cfg: MjcfConverterCfg
    """The configuration instance for MJCF to USD conversion."""

    def __init__(self, cfg: MjcfConverterCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for URDF to USD conversion.
        """
        super().__init__(cfg=cfg)

    """
    Implementation specific methods.
    """

    def _convert_asset(self, cfg: MjcfConverterCfg):
        """Calls underlying Omniverse command to convert MJCF to USD.

        Args:
            cfg: The configuration instance for MJCF to USD conversion.
        """
        pass
