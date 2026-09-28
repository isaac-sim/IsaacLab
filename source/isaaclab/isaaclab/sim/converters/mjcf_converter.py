# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import os
from collections import defaultdict

from ...utils.version import has_kit
from .asset_converter_base import AssetConverterBase
from .mjcf_converter_cfg import MjcfConverterCfg

logger = logging.getLogger(__name__)


class MjcfConverter(AssetConverterBase):
    """Converter for a MJCF description file to a USD file.

    This class wraps around the `isaacsim.asset.importer.mjcf`_ API to provide a lazy
    implementation for MJCF to USD conversion. When the full Isaac Sim runtime is available,
    the Isaac Sim MJCF importer extension is enabled and used; otherwise, the API is loaded
    from the standalone ``isaacsim-asset-isolated`` package. All conversion logic (USD schema
    application, fix-base, density, actuator gains, self-collision, mesh merging, asset
    transformer profile) is performed by :class:`~isaacsim.asset.importer.mjcf.MJCFImporter` —
    this class translates :class:`MjcfConverterCfg` into a flat
    :class:`~isaacsim.asset.importer.mjcf.MJCFImporterConfig`. After the import, it only selects the
    configured physics variant and corrects the PhysX description of the joint friction loss, damping,
    and armature (see :attr:`MjcfConverterCfg.run_multi_physics_conversion`).

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
        # Inside Kit the importer ships as an extension and must be enabled before it can be
        # imported; kitlessly the same module resolves from the standalone importer wheel.
        if has_kit():
            from ..utils import enable_extension  # noqa: PLC0415

            enable_extension("isaacsim.asset.importer.mjcf")
        from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig  # noqa: PLC0415

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
            if cfg.run_multi_physics_conversion:
                self._convert_joint_dynamics_to_physx(layered=cfg.run_asset_transformer)
                # PhysX reads the PhysX description of a layered asset only in its "physx" variant
                if not cfg.run_asset_transformer or cfg.physics_variant == cfg.PhysicsVariant.PHYSX:
                    self._warn_joint_springs(layered=cfg.run_asset_transformer)

    def _warn_joint_springs(self, layered: bool):
        """Warn about the MJCF joint springs, which the PhysX description of the asset omits.

        PhysX joints have no passive spring, so neither the joint ``stiffness`` nor a ``springdamper``, which
        also replaces the joint damping, reaches PhysX.

        Args:
            layered: Whether the asset transformer of the importer split the asset into layers.
        """
        from pxr import Usd, UsdPhysics  # noqa: PLC0415

        if layered:
            usd_path = os.path.join(os.path.dirname(self.usd_path), "payloads", "Physics", "mujoco.usda")
            if not os.path.isfile(usd_path):
                return
        else:
            usd_path = self.usd_path
        # the prims do not keep the stage alive, so it is held for the whole traversal
        stage = Usd.Stage.Open(usd_path)
        springs = defaultdict(list)
        # the physics layers only add opinions under the asset's prims, so they have to be traversed with the
        # all-prims predicate to reach the joints
        for joint in stage.TraverseAll():
            if not joint.IsA(UsdPhysics.Joint):
                continue
            if joint.GetAttribute("mjc:stiffness").Get():
                springs["stiffness"].append(joint.GetName())
            if any(joint.GetAttribute("mjc:springdamper").Get() or ()):
                springs["springdamper stiffness and damping"].append(joint.GetName())
        for quantity, joint_names in springs.items():
            logger.warning(
                "MjcfConverter: PhysX joints have no passive spring, so the PhysX description of the asset omits"
                f" the MJCF joint {quantity} of the joints: {', '.join(joint_names)}."
            )
