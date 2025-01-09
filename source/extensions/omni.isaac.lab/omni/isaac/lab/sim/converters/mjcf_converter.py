# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import omni.kit.commands
import omni.usd
from omni.isaac.core.utils.extensions import enable_extension
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
        import_config = self._get_mjcf_import_config(cfg)
        omni.kit.commands.execute(
            "MJCFCreateAsset",
            mjcf_path=cfg.asset_path,
            import_config=import_config,
            dest_path=self.usd_path,
        )

        # fix the issue that material paths are not relative
        if self.cfg.make_instanceable:
            instanced_usd_path = os.path.join(self.usd_dir, self.usd_instanceable_meshes_path)
            stage = Usd.Stage.Open(instanced_usd_path)
            # resolve all paths relative to layer path
            source_layer = stage.GetRootLayer()
            omni.usd.resolve_paths(source_layer.identifier, source_layer.identifier)
            stage.Save()

        # fix the issue that material paths are not relative
        # note: This issue seems to have popped up in Isaac Sim 2023.1.1
        stage = Usd.Stage.Open(self.usd_path)
        # resolve all paths relative to layer path
        source_layer = stage.GetRootLayer()
        omni.usd.resolve_paths(source_layer.identifier, source_layer.identifier)
        stage.Save()

    def _get_mjcf_import_config(self, cfg: MjcfConverterCfg) -> omni.importer.mjcf.ImportConfig:
        """Returns the import configuration for MJCF to USD conversion.

        Args:
            cfg: The configuration instance for MJCF to USD conversion.

        Returns:
            The constructed ``ImportConfig`` object containing the desired settings.
        """

        # Enable MJCF Extensions
        enable_extension("omni.importer.mjcf")

        from omni.importer.mjcf import _mjcf as omni_mjcf

        import_config = omni_mjcf.ImportConfig()

        # set the unit scaling factor, 1.0 means meters, 100.0 means cm
        import_config.set_distance_scale(1.0)
        # set imported robot as default prim
        import_config.set_make_default_prim(True)
        # add a physics scene to the stage on import if none exists
        import_config.set_create_physics_scene(False)
        # set flag to parse <site> tag
        import_config.set_import_sites(True)

        # -- instancing settings
        # meshes will be placed in a separate usd file
        import_config.set_make_instanceable(cfg.make_instanceable)
        import_config.set_instanceable_usd_path(self.usd_instanceable_meshes_path)

        # -- asset settings
        # default density used for links, use 0 to auto-compute
        import_config.set_density(cfg.link_density)
        # import inertia tensor from urdf, if it is not specified in urdf it will import as identity
        import_config.set_import_inertia_tensor(cfg.import_inertia_tensor)

        # -- physics settings
        # create fix joint for base link
        import_config.set_fix_base(cfg.fix_base)
        # self collisions between links in the articulation
        import_config.set_self_collision(cfg.self_collision)

        return import_config
