# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import pathlib

import carb
import omni.kit.app

from .asset_converter_base import AssetConverterBase
from .urdf_converter_cfg import UrdfConverterCfg


class UrdfConverter(AssetConverterBase):
    """Converter for a URDF description file to a USD file.

    This class wraps around the `isaacsim.asset.importer.urdf`_ extension to provide a lazy
    implementation for URDF to USD conversion. It stores the output USD file in an instanceable
    format since that is what is typically used in all learning related applications.

    The heavy lifting (URDF parsing, fixed-joint merging, fix-base insertion, joint-drive
    configuration, density override, asset transformer profile) is delegated to Isaac Sim's
    :class:`~isaacsim.asset.importer.urdf.URDFImporter` together with
    :class:`~isaacsim.asset.importer.urdf.URDFImporterConfig`. IsaacLab only translates its
    user-friendly :class:`UrdfConverterCfg` into the flat importer config.

    .. caution::
        The current lazy conversion implementation does not automatically trigger USD generation if
        only the mesh files used by the URDF are modified. To force generation, either set
        :obj:`AssetConverterBaseCfg.force_usd_conversion` to True or delete the output directory.

    .. note::
        From Isaac Sim 4.5 onwards, the extension name changed from ``omni.importer.urdf`` to
        ``isaacsim.asset.importer.urdf``.

    .. note::
        In the URDF importer 3.0, the conversion pipeline uses the ``urdf-usd-converter`` library
        and the ``isaacsim.asset.transformer.rules`` extension to produce structured USD output.
        Features such as ``convert_mimic_joints_to_normal_joints`` and
        ``replace_cylinders_with_capsules`` are no longer natively supported by the importer and
        will emit warnings if enabled.

    .. _isaacsim.asset.importer.urdf: https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_urdf.html
    """

    cfg: UrdfConverterCfg
    """The configuration instance for URDF to USD conversion."""

    def __init__(self, cfg: UrdfConverterCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for URDF to USD conversion.
        """
        # enable the URDF importer extension
        manager = omni.kit.app.get_app().get_extension_manager()
        if not manager.is_extension_enabled("isaacsim.asset.importer.urdf"):
            manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)

        # set `usd_file_name` to match the importer's output path structure:
        # the importer generates `{usd_path}/{robot_name}/{robot_name}.usda`
        robot_name = pathlib.PurePath(cfg.asset_path).stem
        cfg.usd_file_name = os.path.join(robot_name, f"{robot_name}.usda")

        super().__init__(cfg=cfg)

    def _convert_asset(self, cfg: UrdfConverterCfg):
        """Run the Isaac Sim URDF importer pipeline.

        Translates :class:`UrdfConverterCfg` into a flat
        :class:`~isaacsim.asset.importer.urdf.URDFImporterConfig` and invokes
        :meth:`~isaacsim.asset.importer.urdf.URDFImporter.import_urdf`. The importer handles
        fixed-joint merging, fix-base insertion, joint-drive configuration, link density
        overrides, and the asset transformer profile internally.

        Args:
            cfg: The URDF conversion configuration.
        """
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

        # log warnings for features no longer supported by the URDF importer 3.0
        self._warn_unsupported_features(cfg)

        # translate nested `JointDriveCfg` into flat importer fields
        drive_type, target_type, stiffness, damping = self._unpack_joint_drive(cfg.joint_drive)

        import_config = URDFImporterConfig(
            urdf_path=os.path.normpath(cfg.asset_path),
            usd_path=os.path.normpath(self.usd_dir),
            merge_fixed_joints=cfg.merge_fixed_joints,
            merge_mesh=cfg.merge_mesh,
            collision_from_visuals=cfg.collision_from_visuals,
            collision_type=cfg.collision_type,
            allow_self_collision=cfg.self_collision,
            ros_package_paths=list(cfg.ros_package_paths),
            robot_type=cfg.robot_type,
            fix_base=cfg.fix_base,
            link_density=cfg.link_density if cfg.link_density > 0.0 else None,
            joint_drive_type=drive_type,
            joint_target_type=target_type,
            override_joint_stiffness=stiffness,
            override_joint_damping=damping,
            run_asset_transformer=cfg.run_asset_transformer,
            run_multi_physics_conversion=cfg.run_multi_physics_conversion,
            debug_mode=cfg.debug_mode,
        )

        generated_usd_path = URDFImporter(import_config).import_urdf()
        if generated_usd_path:
            generated_usd_path = os.path.normpath(generated_usd_path)
            self._usd_file_name = os.path.relpath(generated_usd_path, self.usd_dir)

    @staticmethod
    def _warn_unsupported_features(cfg: UrdfConverterCfg):
        """Log warnings for configuration options no longer supported by the URDF importer 3.0.

        Args:
            cfg: The URDF conversion configuration.
        """
        if cfg.convert_mimic_joints_to_normal_joints:
            carb.log_warn(
                "UrdfConverter: 'convert_mimic_joints_to_normal_joints' is no longer supported"
                " by the URDF importer 3.0."
            )
        if cfg.replace_cylinders_with_capsules:
            carb.log_warn(
                "UrdfConverter: 'replace_cylinders_with_capsules' is no longer supported by the URDF importer 3.0."
            )
        if cfg.root_link_name:
            carb.log_warn("UrdfConverter: 'root_link_name' is no longer supported by the URDF importer 3.0.")
        if cfg.joint_drive and isinstance(
            cfg.joint_drive.gains,
            UrdfConverterCfg.JointDriveCfg.NaturalFrequencyGainsCfg,
        ):
            import warnings

            warnings.warn(
                "UrdfConverter: 'NaturalFrequencyGainsCfg' is deprecated and no longer supported by the"
                " URDF importer 3.0. The `compute_natural_stiffness` function has been removed."
                " Joint drive gains will be left at the values produced by the URDF importer."
                " Please use 'PDGainsCfg' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    @staticmethod
    def _unpack_joint_drive(joint_drive: UrdfConverterCfg.JointDriveCfg | None) -> tuple:
        """Translate an IsaacLab :class:`UrdfConverterCfg.JointDriveCfg` into flat importer fields.

        Args:
            joint_drive: The nested IsaacLab joint-drive configuration, or ``None``.

        Returns:
            Tuple ``(drive_type, target_type, stiffness, damping)`` suitable for
            :class:`~isaacsim.asset.importer.urdf.URDFImporterConfig`. Entries are ``None`` when
            the user did not request an override.
        """
        if joint_drive is None:
            return None, None, None, None

        gains = joint_drive.gains
        if isinstance(gains, UrdfConverterCfg.JointDriveCfg.PDGainsCfg):
            stiffness = gains.stiffness
            damping = gains.damping
        else:
            # `NaturalFrequencyGainsCfg` is deprecated; leave gains unchanged.
            stiffness = None
            damping = None

        return joint_drive.drive_type, joint_drive.target_type, stiffness, damping
