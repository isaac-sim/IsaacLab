# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

from .asset_converter_base import AssetConverterBase
from .mjcf_converter_cfg import MjcfConverterCfg


class MjcfConverter(AssetConverterBase):
    """Converter for a MJCF description file to a USD file.

    This class wraps around the `isaacsim.asset.importer.mjcf`_ extension to provide a lazy implementation
    for MJCF to USD conversion. It uses the :class:`MJCFImporter` class and :class:`MJCFImporterConfig`
    dataclass from Isaac Sim to perform the conversion.

    .. caution::
        The current lazy conversion implementation does not automatically trigger USD generation if
        only the mesh files used by the MJCF are modified. To force generation, either set
        :obj:`AssetConverterBaseCfg.force_usd_conversion` to True or delete the output directory.

    .. note::
        From Isaac Sim 5.0 onwards, the MJCF importer uses the ``mujoco-usd-converter`` library
        and the :class:`MJCFImporter` / :class:`MJCFImporterConfig` API. The old command-based API
        (``MJCFCreateAsset`` / ``MJCFCreateImportConfig``) is deprecated.

    .. note::
        The :attr:`~AssetConverterBaseCfg.make_instanceable` setting from the base class is not
        supported by the new MJCF importer and will be ignored.

    .. _isaacsim.asset.importer.mjcf: https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_mjcf.html
    """

    cfg: MjcfConverterCfg
    """The configuration instance for MJCF to USD conversion."""

    def __init__(self, cfg: MjcfConverterCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for MJCF to USD conversion.
        """
        # The new MJCF importer outputs to: {usd_path}/{robot_name}/{robot_name}.usda
        # Pre-adjust usd_file_name to match this output structure so that lazy conversion works correctly.
        file_basename = os.path.splitext(os.path.basename(cfg.asset_path))[0]
        cfg.usd_file_name = os.path.join(file_basename, f"{file_basename}.usda")
        super().__init__(cfg=cfg)

    """
    Implementation specific methods.
    """

    def _convert_asset(self, cfg: MjcfConverterCfg):
        """Calls underlying Isaac Sim MJCFImporter to convert MJCF to USD.

        Args:
            cfg: The configuration instance for MJCF to USD conversion.
        """
        import shutil

        from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

        # Clean up existing output subdirectory so the importer writes fresh files.
        # The MJCFImporter outputs to {usd_dir}/{robot_name}/{robot_name}.usda and may
        # skip writing if the output already exists from a previous conversion.
        file_basename = os.path.splitext(os.path.basename(cfg.asset_path))[0]
        output_subdir = os.path.join(self.usd_dir, file_basename)
        if os.path.exists(output_subdir):
            shutil.rmtree(output_subdir)

        import_config = MJCFImporterConfig(
            mjcf_path=cfg.asset_path,
            usd_path=self.usd_dir,
            merge_mesh=cfg.merge_mesh,
            collision_from_visuals=cfg.collision_from_visuals,
            collision_type=cfg.collision_type,
            allow_self_collision=cfg.self_collision,
        )

        importer = MJCFImporter(import_config)
        importer.import_mjcf()
