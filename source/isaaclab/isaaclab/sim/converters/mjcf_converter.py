# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import omni.kit.app

from .asset_converter_base import AssetConverterBase
from .mjcf_converter_cfg import MjcfConverterCfg


class MjcfConverter(AssetConverterBase):
    """Converter for a MJCF description file to a USD file.

    This class wraps around the `isaacsim.asset.importer.mjcf`_ extension to provide a lazy
    implementation for MJCF to USD conversion. All conversion logic (USD schema application,
    fix-base, density, actuator gains, self-collision, mesh merging, asset transformer
    profile) is performed by :class:`~isaacsim.asset.importer.mjcf.MJCFImporter` — this class
    only translates :class:`MjcfConverterCfg` into a flat
    :class:`~isaacsim.asset.importer.mjcf.MJCFImporterConfig`.

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
        # enable the MJCF importer extension
        manager = omni.kit.app.get_app().get_extension_manager()
        if not manager.is_extension_enabled("isaacsim.asset.importer.mjcf"):
            manager.set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)

        # The MJCF importer outputs to: {usd_path}/{robot_name}/{robot_name}.usda
        # Pre-adjust `usd_file_name` to match this output structure so that lazy conversion works correctly.
        file_basename = os.path.splitext(os.path.basename(cfg.asset_path))[0]
        cfg.usd_file_name = os.path.join(file_basename, f"{file_basename}.usda")
        super().__init__(cfg=cfg)

    def _convert_asset(self, cfg: MjcfConverterCfg):
        """Run the Isaac Sim MJCF importer pipeline.

        Args:
            cfg: The configuration instance for MJCF to USD conversion.
        """
        from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

        import_config = MJCFImporterConfig(
            mjcf_path=cfg.asset_path,
            usd_path=self.usd_dir,
            import_scene=cfg.import_physics_scene,
            merge_mesh=cfg.merge_mesh,
            collision_from_visuals=cfg.collision_from_visuals,
            collision_type=cfg.collision_type,
            allow_self_collision=cfg.self_collision,
            robot_type=cfg.robot_type,
            fix_base=cfg.fix_base,
            link_density=cfg.link_density if cfg.link_density > 0.0 else None,
            override_gain_type=cfg.override_gain_type,
            override_bias_type=cfg.override_bias_type,
            override_gain_prm=cfg.override_gain_prm,
            override_bias_prm=cfg.override_bias_prm,
            run_asset_transformer=cfg.run_asset_transformer,
            run_multi_physics_conversion=cfg.run_multi_physics_conversion,
            debug_mode=cfg.debug_mode,
        )

        generated_usd_path = MJCFImporter(import_config).import_mjcf()
        if generated_usd_path:
            generated_usd_path = os.path.normpath(generated_usd_path)
            self._usd_file_name = os.path.relpath(generated_usd_path, self.usd_dir)
