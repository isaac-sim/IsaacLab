# Copyright (c) 2022-2024, The ORBIT Project Developers.
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
from .urdf_converter_cfg import UrdfConverterCfg

_DRIVE_TYPE = {
    "none": 0,
    "position": 1,
    "velocity": 2,
}
"""Mapping from drive type name to URDF importer drive number."""

_NORMALS_DIVISION = {
    "catmullClark": 0,
    "loop": 1,
    "bilinear": 2,
    "none": 3,
}
"""Mapping from normals division name to urdf importer normals division number."""


class UrdfConverter(AssetConverterBase):
    """Converter for a URDF description file to a USD file.

    This class wraps around the `omni.isaac.urdf_importer`_ extension to provide a lazy implementation
    for URDF to USD conversion. It stores the output USD file in an instanceable format since that is
    what is typically used in all learning related applications.

    .. caution::
        The current lazy conversion implementation does not automatically trigger USD generation if
        only the mesh files used by the URDF are modified. To force generation, either set
        :obj:`AssetConverterBaseCfg.force_usd_conversion` to True or delete the output directory.

    .. note::
        From Isaac Sim 2023.1 onwards, the extension name changed from ``omni.isaac.urdf`` to
        ``omni.importer.urdf``. This converter class automatically detects the version of Isaac Sim
        and uses the appropriate extension.

        The new extension supports a custom XML tag``"dont_collapse"`` for joints. Setting this parameter
        to true in the URDF joint tag prevents the child link from collapsing when the associated joint type
        is "fixed".

    .. _omni.isaac.urdf_importer: https://docs.omniverse.nvidia.com/isaacsim/latest/ext_omni_isaac_urdf.html
    """

    cfg: UrdfConverterCfg
    """The configuration instance for URDF to USD conversion."""

    def __init__(self, cfg: UrdfConverterCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for URDF to USD conversion.
        """
        super().__init__(cfg=cfg)

    """
    Implementation specific methods.
    """

    def _convert_asset(self, cfg: UrdfConverterCfg):
        """Calls underlying Omniverse command to convert URDF to USD.

        Args:
            cfg: The URDF conversion configuration.
        """
        import_config = self._get_urdf_import_config(cfg)
        omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=cfg.asset_path,
            import_config=import_config,
            dest_path=self.usd_path,
        )
        # fix the issue that material paths are not relative
        if self.cfg.make_instanceable:
            usd_path = os.path.join(self.usd_dir, self.usd_instanceable_meshes_path)
            stage = Usd.Stage.Open(usd_path)
            # resolve all paths relative to layer path
            source_layer = stage.GetRootLayer()
            omni.usd.resolve_paths(source_layer.identifier, source_layer.identifier)
            stage.Save()

    """
    Helper methods.
    """

    def _get_urdf_import_config(self, cfg: UrdfConverterCfg) -> omni.importer.urdf.ImportConfig:
        """Create and fill URDF ImportConfig with desired settings

        Args:
            cfg: The URDF conversion configuration.

        Returns:
            The constructed ``ImportConfig`` object containing the desired settings.
        """
        # Enable urdf extension
        enable_extension("omni.importer.urdf")

        from omni.importer.urdf import _urdf as omni_urdf

        import_config = omni_urdf.ImportConfig()

        # set the unit scaling factor, 1.0 means meters, 100.0 means cm
        import_config.set_distance_scale(1.0)
        # set imported robot as default prim
        import_config.set_make_default_prim(True)
        # add a physics scene to the stage on import if none exists
        import_config.set_create_physics_scene(False)

        # -- instancing settings
        # meshes will be placed in a separate usd file
        import_config.set_make_instanceable(cfg.make_instanceable)
        import_config.set_instanceable_usd_path(self.usd_instanceable_meshes_path)

        # -- asset settings
        # default density used for links, use 0 to auto-compute
        import_config.set_density(cfg.link_density)
        # import inertia tensor from urdf, if it is not specified in urdf it will import as identity
        import_config.set_import_inertia_tensor(cfg.import_inertia_tensor)
        # decompose a convex mesh into smaller pieces for a closer fit
        import_config.set_convex_decomp(cfg.convex_decompose_mesh)
        import_config.set_subdivision_scheme(_NORMALS_DIVISION["bilinear"])

        # -- physics settings
        # create fix joint for base link
        import_config.set_fix_base(cfg.fix_base)
        # consolidating links that are connected by fixed joints
        import_config.set_merge_fixed_joints(cfg.merge_fixed_joints)
        # self collisions between links in the articulation
        import_config.set_self_collision(cfg.self_collision)

        # default drive type used for joints
        import_config.set_default_drive_type(_DRIVE_TYPE[cfg.default_drive_type])
        # default proportional gains
        import_config.set_default_drive_strength(cfg.default_drive_stiffness)
        # default derivative gains
        import_config.set_default_position_drive_damping(cfg.default_drive_damping)

        return import_config
