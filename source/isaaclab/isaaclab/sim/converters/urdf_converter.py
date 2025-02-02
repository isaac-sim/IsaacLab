# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import re

import isaacsim
import omni.kit.app
import omni.kit.commands
import omni.usd
from isaacsim.core.utils.extensions import enable_extension

from .asset_converter_base import AssetConverterBase
from .urdf_converter_cfg import UrdfConverterCfg


class UrdfConverter(AssetConverterBase):
    """Converter for a URDF description file to a USD file.

    This class wraps around the `isaacsim.asset.importer.urdf`_ extension to provide a lazy implementation
    for URDF to USD conversion. It stores the output USD file in an instanceable format since that is
    what is typically used in all learning related applications.

    .. caution::
        The current lazy conversion implementation does not automatically trigger USD generation if
        only the mesh files used by the URDF are modified. To force generation, either set
        :obj:`AssetConverterBaseCfg.force_usd_conversion` to True or delete the output directory.

    .. note::
        From Isaac Sim 4.5 onwards, the extension name changed from ``omni.importer.urdf`` to
        ``isaacsim.asset.importer.urdf``. This converter class now uses the latest extension from Isaac Sim.

    .. _isaacsim.asset.importer.urdf: https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/ext_isaacsim_asset_importer_urdf.html
    """

    cfg: UrdfConverterCfg
    """The configuration instance for URDF to USD conversion."""

    def __init__(self, cfg: UrdfConverterCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for URDF to USD conversion.
        """
        manager = omni.kit.app.get_app().get_extension_manager()
        if not manager.is_extension_enabled("isaacsim.asset.importer.urdf"):
            enable_extension("isaacsim.asset.importer.urdf")
        from isaacsim.asset.importer.urdf._urdf import acquire_urdf_interface

        self._urdf_interface = acquire_urdf_interface()
        super().__init__(cfg=cfg)

    """
    Implementation specific methods.
    """

    def _convert_asset(self, cfg: UrdfConverterCfg):
        """Calls underlying Omniverse command to convert URDF to USD.

        Args:
            cfg: The URDF conversion configuration.
        """

        import_config = self._get_urdf_import_config()
        # parse URDF file
        result, self._robot_model = omni.kit.commands.execute(
            "URDFParseFile", urdf_path=cfg.asset_path, import_config=import_config
        )

        if result:
            if cfg.joint_drive:
                # modify joint parameters
                self._update_joint_parameters()

            # set root link name
            if cfg.root_link_name:
                self._robot_model.root_link = cfg.root_link_name

            # convert the model to USD
            omni.kit.commands.execute(
                "URDFImportRobot",
                urdf_path=cfg.asset_path,
                urdf_robot=self._robot_model,
                import_config=import_config,
                dest_path=self.usd_path,
            )
        else:
            raise ValueError(f"Failed to parse URDF file: {cfg.asset_path}")

    """
    Helper methods.
    """

    def _get_urdf_import_config(self) -> isaacsim.asset.importer.urdf.ImportConfig:
        """Create and fill URDF ImportConfig with desired settings

        Returns:
            The constructed ``ImportConfig`` object containing the desired settings.
        """
        # create a new import config
        _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")

        # set the unit scaling factor, 1.0 means meters, 100.0 means cm
        import_config.set_distance_scale(1.0)
        # set imported robot as default prim
        import_config.set_make_default_prim(True)
        # add a physics scene to the stage on import if none exists
        import_config.set_create_physics_scene(False)

        # -- asset settings
        # default density used for links, use 0 to auto-compute
        import_config.set_density(self.cfg.link_density)
        # mesh simplification settings
        convex_decomp = self.cfg.collider_type == "convex_decomposition"
        import_config.set_convex_decomp(convex_decomp)
        # create collision geometry from visual geometry
        import_config.set_collision_from_visuals(self.cfg.collision_from_visuals)
        # consolidating links that are connected by fixed joints
        import_config.set_merge_fixed_joints(self.cfg.merge_fixed_joints)
        # -- physics settings
        # create fix joint for base link
        import_config.set_fix_base(self.cfg.fix_base)
        # self collisions between links in the articulation
        import_config.set_self_collision(self.cfg.self_collision)
        # convert mimic joints to normal joints
        import_config.set_parse_mimic(self.cfg.convert_mimic_joints_to_normal_joints)
        # replace cylinder shapes with capsule shapes
        import_config.set_replace_cylinders_with_capsules(self.cfg.replace_cylinders_with_capsules)

        return import_config

    def _update_joint_parameters(self):
        """Update the joint parameters based on the configuration."""
        # set the drive type
        self._set_joints_drive_type()
        # set the drive target type
        self._set_joints_drive_target_type()
        # set the drive gains
        self._set_joint_drive_gains()

    def _set_joints_drive_type(self):
        """Set the joint drive type for all joints in the URDF model."""
        from isaacsim.asset.importer.urdf._urdf import UrdfJointDriveType

        drive_type_mapping = {
            "force": UrdfJointDriveType.JOINT_DRIVE_FORCE,
            "acceleration": UrdfJointDriveType.JOINT_DRIVE_ACCELERATION,
        }

        if isinstance(self.cfg.joint_drive.drive_type, str):
            for joint in self._robot_model.joints.values():
                joint.drive.set_drive_type(drive_type_mapping[self.cfg.joint_drive.drive_type])
        elif isinstance(self.cfg.joint_drive.drive_type, dict):
            for joint_name, drive_type in self.cfg.joint_drive.drive_type.items():
                # handle joint name being a regex
                matches = [s for s in self._robot_model.joints.keys() if re.search(joint_name, s)]
                if not matches:
                    raise ValueError(
                        f"The joint name {joint_name} in the drive type config was not found in the URDF file. The"
                        f" joint names in the URDF are {list(self._robot_model.joints.keys())}"
                    )
                for match in matches:
                    joint = self._robot_model.joints[match]
                    joint.drive.set_drive_type(drive_type_mapping[drive_type])

    def _set_joints_drive_target_type(self):
        """Set the joint drive target type for all joints in the URDF model."""
        from isaacsim.asset.importer.urdf._urdf import UrdfJointTargetType

        target_type_mapping = {
            "none": UrdfJointTargetType.JOINT_DRIVE_NONE,
            "position": UrdfJointTargetType.JOINT_DRIVE_POSITION,
            "velocity": UrdfJointTargetType.JOINT_DRIVE_VELOCITY,
        }

        if isinstance(self.cfg.joint_drive.target_type, str):
            for joint in self._robot_model.joints.values():
                joint.drive.set_target_type(target_type_mapping[self.cfg.joint_drive.target_type])
        elif isinstance(self.cfg.joint_drive.target_type, dict):
            for joint_name, target_type in self.cfg.joint_drive.target_type.items():
                # handle joint name being a regex
                matches = [s for s in self._robot_model.joints.keys() if re.search(joint_name, s)]
                if not matches:
                    raise ValueError(
                        f"The joint name {joint_name} in the target type config was not found in the URDF file. The"
                        f" joint names in the URDF are {list(self._robot_model.joints.keys())}"
                    )
                for match in matches:
                    joint = self._robot_model.joints[match]
                    joint.drive.set_target_type(target_type_mapping[target_type])

    def _set_joint_drive_gains(self):
        """Set the joint drive gains for all joints in the URDF model."""

        # set the gains directly from stiffness and damping values
        if isinstance(self.cfg.joint_drive.gains, UrdfConverterCfg.JointDriveCfg.PDGainsCfg):
            # stiffness
            if isinstance(self.cfg.joint_drive.gains.stiffness, (float, int)):
                for joint in self._robot_model.joints.values():
                    self._set_joint_drive_stiffness(joint, self.cfg.joint_drive.gains.stiffness)
            elif isinstance(self.cfg.joint_drive.gains.stiffness, dict):
                for joint_name, stiffness in self.cfg.joint_drive.gains.stiffness.items():
                    # handle joint name being a regex
                    matches = [s for s in self._robot_model.joints.keys() if re.search(joint_name, s)]
                    if not matches:
                        raise ValueError(
                            f"The joint name {joint_name} in the drive stiffness config was not found in the URDF file."
                            f" The joint names in the URDF are {list(self._robot_model.joints.keys())}"
                        )
                    for match in matches:
                        joint = self._robot_model.joints[match]
                        self._set_joint_drive_stiffness(joint, stiffness)
            # damping
            if isinstance(self.cfg.joint_drive.gains.damping, (float, int)):
                for joint in self._robot_model.joints.values():
                    self._set_joint_drive_damping(joint, self.cfg.joint_drive.gains.damping)
            elif isinstance(self.cfg.joint_drive.gains.damping, dict):
                for joint_name, damping in self.cfg.joint_drive.gains.damping.items():
                    # handle joint name being a regex
                    matches = [s for s in self._robot_model.joints.keys() if re.search(joint_name, s)]
                    if not matches:
                        raise ValueError(
                            f"The joint name {joint_name} in the drive damping config was not found in the URDF file."
                            f" The joint names in the URDF are {list(self._robot_model.joints.keys())}"
                        )
                    for match in matches:
                        joint = self._robot_model.joints[match]
                        self._set_joint_drive_damping(joint, damping)

        # set the gains from natural frequency and damping ratio
        elif isinstance(self.cfg.joint_drive.gains, UrdfConverterCfg.JointDriveCfg.NaturalFrequencyGainsCfg):
            # damping ratio
            if isinstance(self.cfg.joint_drive.gains.damping_ratio, (float, int)):
                for joint in self._robot_model.joints.values():
                    joint.drive.damping_ratio = self.cfg.joint_drive.gains.damping_ratio
            elif isinstance(self.cfg.joint_drive.gains.damping_ratio, dict):
                for joint_name, damping_ratio in self.cfg.joint_drive.gains.damping_ratio.items():
                    # handle joint name being a regex
                    matches = [s for s in self._robot_model.joints.keys() if re.search(joint_name, s)]
                    if not matches:
                        raise ValueError(
                            f"The joint name {joint_name} in the damping ratio config was not found in the URDF file."
                            f" The joint names in the URDF are {list(self._robot_model.joints.keys())}"
                        )
                    for match in matches:
                        joint = self._robot_model.joints[match]
                        joint.drive.damping_ratio = damping_ratio

            # natural frequency (this has to be done after damping ratio is set)
            if isinstance(self.cfg.joint_drive.gains.natural_frequency, (float, int)):
                for joint in self._robot_model.joints.values():
                    joint.drive.natural_frequency = self.cfg.joint_drive.gains.natural_frequency
                    self._set_joint_drive_gains_from_natural_frequency(joint)
            elif isinstance(self.cfg.joint_drive.gains.natural_frequency, dict):
                for joint_name, natural_frequency in self.cfg.joint_drive.gains.natural_frequency.items():
                    # handle joint name being a regex
                    matches = [s for s in self._robot_model.joints.keys() if re.search(joint_name, s)]
                    if not matches:
                        raise ValueError(
                            f"The joint name {joint_name} in the natural frequency config was not found in the URDF"
                            f" file. The joint names in the URDF are {list(self._robot_model.joints.keys())}"
                        )
                    for match in matches:
                        joint = self._robot_model.joints[match]
                        joint.drive.natural_frequency = natural_frequency
                        self._set_joint_drive_gains_from_natural_frequency(joint)

    def _set_joint_drive_stiffness(self, joint, stiffness: float):
        """Set the joint drive stiffness.

        Args:
            joint: The joint from the URDF robot model.
            stiffness: The stiffness value.
        """
        from isaacsim.asset.importer.urdf._urdf import UrdfJointType

        if joint.type == UrdfJointType.JOINT_PRISMATIC:
            joint.drive.set_strength(stiffness)
        else:
            # we need to convert the stiffness from radians to degrees
            joint.drive.set_strength(math.pi / 180 * stiffness)

    def _set_joint_drive_damping(self, joint, damping: float):
        """Set the joint drive damping.

        Args:
            joint: The joint from the URDF robot model.
            damping: The damping value.
        """
        from isaacsim.asset.importer.urdf._urdf import UrdfJointType

        if joint.type == UrdfJointType.JOINT_PRISMATIC:
            joint.drive.set_damping(damping)
        else:
            # we need to convert the damping from radians to degrees
            joint.drive.set_damping(math.pi / 180 * damping)

    def _set_joint_drive_gains_from_natural_frequency(self, joint):
        """Compute the joint drive gains from the natural frequency and damping ratio.

        Args:
            joint: The joint from the URDF robot model.
        """
        from isaacsim.asset.importer.urdf._urdf import UrdfJointDriveType, UrdfJointTargetType

        strength = self._urdf_interface.compute_natural_stiffness(
            self._robot_model,
            joint.name,
            joint.drive.natural_frequency,
        )
        self._set_joint_drive_stiffness(joint, strength)

        if joint.drive.target_type == UrdfJointTargetType.JOINT_DRIVE_POSITION:
            m_eq = 1.0
            if joint.drive.drive_type == UrdfJointDriveType.JOINT_DRIVE_FORCE:
                m_eq = joint.inertia
            damping = 2 * m_eq * joint.drive.natural_frequency * joint.drive.damping_ratio
            self._set_joint_drive_damping(joint, damping)
