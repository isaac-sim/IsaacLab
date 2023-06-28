# Copyright [2023] Boston Dynamics AI Institute, Inc.
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import json
import os
import pathlib
import random
from dataclasses import MISSING
from datetime import datetime
from typing import Optional

import omni.kit.commands
from omni.isaac.urdf import _urdf as omni_urdf

from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.io import dump_yaml

__all__ = ["UrdfLoaderCfg", "UrdfLoader"]


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


@configclass
class UrdfLoaderCfg:
    """The configuration class for UrdfLoader."""

    urdf_path: str = MISSING
    """The path to the urdf file (e.g. path/to/urdf/robot.urdf)."""

    usd_dir: Optional[str] = None
    """The output directory path to store the generated USD file. Defaults to :obj:`None`.

    If set to :obj:`None`, it is resolved as ``/tmp/Orbit/usd_{date}_{time}_{random}``, where
    the parameters in braces are runtime generated.
    """

    usd_file_name: Optional[str] = None
    """The name of the generated usd file. Defaults to :obj:`None`.

    If set to :obj:`None`, it is resolved from the urdf file name.
    """

    force_usd_conversion: bool = False
    """Force the conversion of the urdf file to usd. Defaults to False."""

    link_density = 0.0
    """Default density used for links. Defaults to 0.

    This setting is only effective if ``"inertial"`` properties are missing in the URDF.
    """

    import_inertia_tensor: bool = True
    """Import the inertia tensor from urdf. Defaults to True.

    If the ``"inertial"`` tag is missing, then it is imported as an identity.
    """

    convex_decompose_mesh = False
    """Decompose a convex mesh into smaller pieces for a closer fit. Defaults to False."""

    fix_base: bool = MISSING
    """Create a fix joint to the root/base link. Defaults to True."""

    merge_fixed_joints: bool = False
    """Consolidate links that are connected by fixed joints. Defaults to False."""

    self_collision: bool = False
    """Activate self-collisions between links of the articulation. Defaults to False."""

    default_drive_type: str = "none"
    """The drive type used for joints. Defaults to ``"none"``.

    The drive type dictates the loaded joint PD gains and USD attributes for joint control:

    * ``"none"``: The joint stiffness and damping are set to 0.0.
    * ``"position"``: The joint stiff and damping are set based on the URDF file or provided configuration.
    * ``"velocity"``: The joint stiff is set to zero and damping is based on the URDF file or provided configuration.
    """

    default_drive_stiffness: float = 0.0
    """The default stiffness of the joint drive. Defaults to 0.0."""

    default_drive_damping: float = 0.0
    """The default damping of the joint drive. Defaults to 0.0.

    Note:
        If set to zero, the values parsed from the URDF joint tag ``"<dynamics><damping>"`` are used.
        Otherwise, it is overridden by the configured value.
    """


class UrdfLoader:
    """Loader for a URDF description file as an instanceable USD file.

    This class wraps around the ``omni.isaac.urdf_importer`` extension to provide a lazy implementation
    for URDF to USD conversion. It stores the output USD file in an instanceable format since that is
    what is typically used in all learning related applications.

    The file conversion is lazy if the output usd directory, :obj:`UrdfLoaderCfg.usd_dir`, is provided.
    In the lazy conversion, the USD file is only re-generated if the usd files do not exist or if they exist,
    the provided configuration or the main urdf file is modified. To override this behavior, set
    :obj:`UrdfLoaderCfg.force_usd_conversion` as True.

    In the case that no USD directory is defined, lazy conversion is deactivated and the generated USD file is
    stored in folder ``/tmp/Orbit/usd_{date}_{time}_{random}``, where the parameters in braces are generated
    at runtime. The random identifiers help avoid a race condition where two simultaneously triggered conversions
    try to use the same directory for reading/writing the generated files.

    .. caution::
        The current lazy conversion implementation does not automatically trigger USD generation if only the
        mesh files used by the URDF are modified. To force generation, either set
        :obj:`UrdfLoaderCfg.force_usd_conversion` to True or remove the USD folder.

    .. note::
        Additionally, changes to the parameters :obj:`UrdfLoaderCfg.urdf_path`, :obj:`UrdfLoaderCfg.usd_dir`, and
        :obj:`UrdfLoaderCfg.usd_file_name` are not considered as modifications in the configuration instance that
        trigger USD file re-generation.

    """

    def __init__(self, cfg: UrdfLoaderCfg):
        """Initializes the class.

        Args:
            cfg (UrdfLoaderCfg): The configuration instance for URDF to USD conversion.

        Raises:
            ValueError: When provided URDF file does not exist.
        """
        # check if the urdf file exists
        if not os.path.isfile(cfg.urdf_path):
            raise ValueError(f"The URDF path does not exist: ({cfg.urdf_path})!")

        # resolve USD directory name
        if cfg.usd_dir is None:
            # a folder in "/tmp/Orbit" by the name: usd_{date}_{time}_{random}
            time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._usd_dir = f"/tmp/Orbit/usd_{time_tag}_{random.randrange(10000)}"
        else:
            self._usd_dir = cfg.usd_dir

        # resolve the file name from urdf file name if not provided
        if cfg.usd_file_name is None:
            usd_file_name = pathlib.PurePath(cfg.urdf_path).stem
        else:
            usd_file_name = cfg.usd_file_name
        # add USD extension if not provided
        if not (usd_file_name.endswith(".usd") or usd_file_name.endswith(".usda")):
            self._usd_file_name = usd_file_name + ".usd"
        else:
            self._usd_file_name = usd_file_name

        # create the USD directory
        os.makedirs(self.usd_dir, exist_ok=True)
        # check if usd files exist
        self._usd_file_exists = os.path.isfile(self.usd_path)
        # path to read/write urdf hash file
        dest_hash_path = f"{self.usd_dir}/.urdf_hash"
        # convert urdf to hash
        urdf_hash = UrdfLoader._config_to_hash(cfg)
        # read the saved hash
        try:
            with open(dest_hash_path) as f:
                existing_urdf_hash = f.readline()
                self._is_same_urdf = existing_urdf_hash == urdf_hash
        except FileNotFoundError:
            self._is_same_urdf = False

        # generate usd files
        if cfg.force_usd_conversion or not self._usd_file_exists or not self._is_same_urdf:
            # write the updated hash
            with open(dest_hash_path, "w") as f:
                f.write(urdf_hash)
            # Convert urdf to an instantiable usd
            import_config = self._get_urdf_import_config(cfg)
            omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=cfg.urdf_path,
                import_config=import_config,
                dest_path=self.usd_path,
            )
            # Dump the configuration to a file
            dump_yaml(os.path.join(self.usd_dir, "config.yaml"), cfg.to_dict())

    """
    Properties.
    """

    @property
    def usd_dir(self) -> str:
        """The path to the directory where the generated USD files are stored."""
        return self._usd_dir

    @property
    def usd_file_name(self) -> str:
        """The file name of the generated USD file."""
        return self._usd_file_name

    @property
    def usd_path(self) -> str:
        """The path to the generated USD file."""
        return os.path.join(self.usd_dir, self.usd_file_name)

    @property
    def usd_instanceable_meshes_path(self) -> str:
        """The path to the USD mesh file."""
        return os.path.join(self.usd_dir, "Props", "instanceable_meshes.usd")

    """
    Private helpers.
    """

    def _get_urdf_import_config(self, cfg: UrdfLoaderCfg) -> omni_urdf.ImportConfig:
        """Set the settings into the import config."""

        import_config = omni_urdf.ImportConfig()

        # set the unit scaling factor, 1.0 means meters, 100.0 means cm
        import_config.set_distance_scale(1.0)
        # set imported robot as default prim
        import_config.set_make_default_prim(True)
        # add a physics scene to the stage on import if none exists
        import_config.set_create_physics_scene(False)

        # -- instancing settings
        # meshes will be placed in a separate usd file
        import_config.set_make_instanceable(True)
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

    @staticmethod
    def _config_to_hash(cfg: UrdfLoaderCfg) -> str:
        """Converts the configuration object and urdf file to an MD5 hash of a string.

        .. warning::
            It only checks the main urdf file not the mesh files.

        Args:
            config (UrdfLoaderCfg): The urdf loader configuration object.

        Returns:
            An MD5 hash of a string.
        """

        # convert ro dict and remove path related info
        config_dic = cfg.to_dict()
        _ = config_dic.pop("urdf_path")
        _ = config_dic.pop("usd_dir")
        _ = config_dic.pop("usd_file_name")
        # convert config dic to bytes
        config_bytes = json.dumps(config_dic).encode()
        # hash config
        md5 = hashlib.md5()
        md5.update(config_bytes)

        # read the urdf file to observe changes
        with open(cfg.urdf_path, "rb") as f:
            while True:
                # read 64kb chunks to avoid memory issues for the large files!
                data = f.read(65536)
                if not data:
                    break
                md5.update(data)
        return md5.hexdigest()
