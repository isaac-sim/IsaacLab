# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import gc
import importlib
import math
import os
import pathlib
import re
import shutil
import tempfile

import carb
import omni.kit.app

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
        ``isaacsim.asset.importer.urdf``.

    .. note::
        In the URDF importer 3.0, the conversion pipeline uses the ``urdf-usd-converter`` library
        and the ``isaacsim.asset.transformer.rules`` extension to produce structured USD output.
        Features such as ``convert_mimic_joints_to_normal_joints`` and
        ``replace_cylinders_with_capsules`` are no longer natively supported by the importer and
        will emit warnings if enabled.

    .. note::
        The ``merge_fixed_joints`` feature is implemented as a URDF XML pre-processing step that
        runs *before* the USD conversion. It removes fixed joints from the URDF and merges the
        child link's visual, collision, and inertial elements into the parent link.

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

        # set `usd_file_name` to match the new importer's output path structure:
        # the importer generates `{usd_path}/{robot_name}/{robot_name}.usda`
        robot_name = pathlib.PurePath(cfg.asset_path).stem
        cfg.usd_file_name = os.path.join(robot_name, f"{robot_name}.usda")

        super().__init__(cfg=cfg)

    """
    Implementation specific methods.
    """

    def _convert_asset(self, cfg: UrdfConverterCfg):
        """Calls the URDF importer 3.0 pipeline to convert URDF to USD.

        This method replicates the ``URDFImporter.import_urdf()`` pipeline from the
        ``isaacsim.asset.importer.urdf`` extension, inserting IsaacLab-specific post-processing
        (fix base, joint drives, link density) on the intermediate stage before the asset
        transformer restructures the output.

        Args:
            cfg: The URDF conversion configuration.
        """
        from isaacsim.asset.importer.utils.impl import importer_utils, stage_utils
        from pxr import Sdf

        from .urdf_utils import merge_fixed_joints

        # log warnings for features no longer supported by the URDF importer 3.0
        self._warn_unsupported_features(cfg)

        urdf_path = os.path.normpath(cfg.asset_path)
        robot_name = os.path.basename(urdf_path).split(".")[0]
        usd_path = os.path.normpath(self.usd_dir)

        # step 0: optionally pre-process the URDF to merge fixed joints
        # The merged file is written next to the original so that relative mesh paths
        # (e.g. ``meshes/link.stl``) continue to resolve correctly.  If the source
        # directory is read-only, a temp directory is used as a fallback (relative mesh
        # paths may not resolve in that case).
        merged_urdf_path: str | None = None
        if cfg.merge_fixed_joints:
            urdf_dir = os.path.dirname(urdf_path)
            try:
                fd, merged_urdf_path = tempfile.mkstemp(suffix=".urdf", prefix=".merged_", dir=urdf_dir)
                os.close(fd)
            except OSError:
                carb.log_warn(
                    "UrdfConverter: Cannot write merged URDF next to the original (read-only directory)."
                    " Falling back to a temp directory — relative mesh paths may not resolve."
                )
                merged_urdf_dir = tempfile.mkdtemp(prefix="isaaclab_urdf_merge_")
                merged_urdf_path = os.path.join(merged_urdf_dir, os.path.basename(urdf_path))
            merge_fixed_joints(urdf_path, merged_urdf_path)
            urdf_path = merged_urdf_path

        usdex_path = os.path.normpath(os.path.join(usd_path, "usdex"))
        intermediate_path = os.path.normpath(os.path.join(usd_path, "temp", f"{robot_name}.usd"))

        # step 1: convert URDF to intermediate USD using urdf-usd-converter
        urdf_usd_converter = importlib.import_module("urdf_usd_converter")
        converter = urdf_usd_converter.Converter(layer_structure=False, scene=False)
        asset: Sdf.AssetPath = converter.convert(urdf_path, usdex_path)

        # step 2: open the intermediate stage and run standard post-processing
        stage = stage_utils.open_stage(asset.path)
        if not stage:
            raise ValueError(f"Failed to open intermediate stage at path: {asset.path}")

        importer_utils.remove_custom_scopes(stage)
        importer_utils.add_rigid_body_schemas(stage)
        importer_utils.add_joint_schemas(stage)

        # step 3: apply optional importer features
        if cfg.collision_from_visuals:
            collision_type_map = {
                "convex_hull": "Convex Hull",
                "convex_decomposition": "Convex Decomposition",
            }
            collision_type = collision_type_map.get(cfg.collision_type, "Convex Hull")
            importer_utils.collision_from_visuals(stage, collision_type)

        importer_utils.enable_self_collision(stage, cfg.self_collision)

        # step 4: IsaacLab-specific post-processing on the intermediate stage
        if cfg.fix_base:
            self._apply_fix_base(stage)

        if cfg.link_density > 0:
            self._apply_link_density(stage, cfg.link_density)

        if cfg.joint_drive:
            self._apply_joint_drives(stage, cfg)

        # step 5: save the intermediate stage
        stage_utils.save_stage(stage, intermediate_path)
        stage = None
        gc.collect()

        # step 6: run the asset transformer to produce the final structured output
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = ext_manager.get_enabled_extension_id("isaacsim.asset.transformer.rules")
        extension_path = ext_manager.get_extension_path(ext_id)
        asset_structure_profile_json_path = os.path.normpath(
            os.path.abspath(os.path.join(extension_path, "data", "isaacsim_structure.json"))
        )

        importer_utils.run_asset_transformer_profile(
            input_stage_path=intermediate_path,
            output_package_root=os.path.normpath(os.path.join(usd_path, robot_name)),
            profile_json_path=asset_structure_profile_json_path,
        )

        # step 6b: fix ArticulationRootAPI placement for fixed-base articulations.
        # After the asset transformer, ArticulationRootAPI ends up on the root rigid body.
        # Having a FixedJoint on the same rigid body that has ArticulationRootAPI causes
        # PhysX to treat the articulation as a floating-base + constraint (maximal coordinate
        # tree) rather than a fixed-base reduced-coordinate articulation.
        # Moving ArticulationRootAPI to the parent of the root rigid body resolves this.
        if cfg.fix_base:
            final_usd_path = os.path.join(usd_path, robot_name, f"{robot_name}.usda")
            self._fix_articulation_root_for_fixed_base(final_usd_path)

        # step 7: clean up intermediate files
        if os.path.exists(usdex_path):
            shutil.rmtree(usdex_path)
        temp_dir = os.path.dirname(intermediate_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if merged_urdf_path is not None:
            with contextlib.suppress(OSError):
                os.remove(merged_urdf_path)
            # if we used a fallback temp directory, clean that up too
            merged_parent = os.path.dirname(merged_urdf_path)
            if merged_parent.startswith(tempfile.gettempdir()) and os.path.isdir(merged_parent):
                shutil.rmtree(merged_parent, ignore_errors=True)

    """
    Helper methods.
    """

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
    def _apply_fix_base(stage):
        """Add a fixed joint from the world to the root link of the robot.

        Args:
            stage: The USD stage to modify.
        """
        from pxr import UsdPhysics

        default_prim = stage.GetDefaultPrim()
        if not default_prim or not default_prim.IsValid():
            carb.log_warn("UrdfConverter: Cannot apply fix_base - no default prim found.")
            return

        # find the root link: first child with `RigidBodyAPI` under the prim hierarchy
        root_link = None
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                root_link = prim
                break

        if root_link is None:
            carb.log_warn("UrdfConverter: Cannot apply fix_base - no rigid body link found.")
            return

        # create a fixed joint connecting the world to the root link
        default_prim_path = default_prim.GetPath()
        joint_path = default_prim_path.AppendChild("fix_base_joint")

        fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        # `body0` left empty => connected to the world frame
        fixed_joint.CreateBody1Rel().SetTargets([root_link.GetPath()])

    @staticmethod
    def _fix_articulation_root_for_fixed_base(usd_path: str):
        """Move ArticulationRootAPI from the root rigid body to its parent prim.

        After the asset transformer, ArticulationRootAPI ends up on the root rigid body.
        When combined with a FixedJoint on that same body (``fix_base_joint``), PhysX treats
        the articulation as a floating-base + external constraint (maximal coordinate tree)
        rather than a proper fixed-base reduced-coordinate articulation.

        Moving ArticulationRootAPI to the parent of the root rigid body (a non-rigid Xform /
        Scope ancestor) resolves this, matching the pattern used by ``schemas.py``'s
        ``fix_root_link``.

        Changes are authored as **local opinions in the root layer** of the stage, which are
        stronger than the variant-payload-sublayer opinions written by the asset transformer.
        This means the root layer's ``delete apiSchemas`` overrides the ``prepend apiSchemas``
        in the deeper sublayers without modifying those files.

        Args:
            usd_path: Absolute path to the final ``.usda`` file produced by the asset transformer.
        """
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(usd_path)
        if not stage:
            carb.log_warn(
                f"UrdfConverter: Cannot open final stage at '{usd_path}'"
                " for fix_base ArticulationRootAPI post-processing."
            )
            return

        # Find the root rigid body that incorrectly has ArticulationRootAPI applied.
        root_body_prim = None
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                root_body_prim = prim
                break

        if root_body_prim is None:
            # ArticulationRootAPI is already on a non-rigid ancestor (correct) or not present.
            return

        parent_prim = root_body_prim.GetParent()
        if not parent_prim or not parent_prim.IsValid():
            carb.log_warn("UrdfConverter: Root rigid body has no valid parent prim — skipping ArticulationRootAPI fix.")
            return

        # Collect all articulation-related schema names applied to the root rigid body.
        articulation_api_names = [
            name
            for name in root_body_prim.GetAppliedSchemas()
            if "ArticulationRoot" in name or name == "PhysxArticulationAPI"
        ]

        # --- Apply ArticulationRootAPI schemas to the parent prim ---
        # (edit target is the root layer by default; writes local opinions)
        UsdPhysics.ArticulationRootAPI.Apply(parent_prim)
        already_on_parent = set(parent_prim.GetAppliedSchemas())
        for name in articulation_api_names:
            if name != "PhysicsArticulationRootAPI" and name not in already_on_parent:
                parent_prim.AddAppliedSchema(name)

        # --- Copy USD articulation attributes to the parent prim ---
        usd_art_api = UsdPhysics.ArticulationRootAPI(root_body_prim)
        for attr_name in usd_art_api.GetSchemaAttributeNames():
            attr = root_body_prim.GetAttribute(attr_name)
            val = attr.Get() if attr else None
            if val is not None:
                parent_attr = parent_prim.GetAttribute(attr_name)
                if not parent_attr:
                    parent_attr = parent_prim.CreateAttribute(attr_name, attr.GetTypeName())
                parent_attr.Set(val)

        # --- Copy physxArticulation:* attributes to the parent prim ---
        for attr in root_body_prim.GetAttributes():
            aname = attr.GetName()
            if aname.startswith("physxArticulation:"):
                val = attr.Get()
                if val is not None:
                    parent_attr = parent_prim.GetAttribute(aname)
                    if not parent_attr:
                        parent_attr = parent_prim.CreateAttribute(aname, attr.GetTypeName())
                    parent_attr.Set(val)

        # --- Remove ArticulationRootAPI schemas from the root rigid body ---
        # Writing "delete" list-ops in the root layer overrides "prepend" in sublayers.
        root_body_prim.RemoveAppliedSchema("PhysxArticulationAPI")
        root_body_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        for name in articulation_api_names:
            if name not in ("PhysicsArticulationRootAPI", "PhysxArticulationAPI"):
                root_body_prim.RemoveAppliedSchema(name)

        # Save only the root layer (sublayers produced by the asset transformer are untouched).
        stage.GetRootLayer().Save()

    @staticmethod
    def _apply_link_density(stage, density: float):
        """Set default density on rigid body links that have no explicit mass.

        Args:
            stage: The USD stage to modify.
            density: The density value in kg/m^3.
        """
        from pxr import UsdPhysics

        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.MassAPI):
                continue
            mass_api = UsdPhysics.MassAPI(prim)
            # only set density if mass is not explicitly specified (0.0 means auto-compute)
            mass_attr = mass_api.GetMassAttr()
            if mass_attr and mass_attr.HasValue() and mass_attr.Get() > 0.0:
                continue
            density_attr = mass_api.GetDensityAttr()
            if not density_attr:
                density_attr = mass_api.CreateDensityAttr()
            density_attr.Set(density)

    def _apply_joint_drives(self, stage, cfg: UrdfConverterCfg):
        """Set joint drive properties (type, target, gains) on USD joints.

        Args:
            stage: The USD stage to modify.
            cfg: The URDF converter configuration containing joint drive settings.
        """
        from pxr import UsdPhysics

        # collect all joints with their metadata
        joints: dict[str, tuple] = {}
        for prim in stage.Traverse():
            if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
                continue
            joint_name = prim.GetName()
            is_revolute = prim.IsA(UsdPhysics.RevoluteJoint)
            instance_name = "angular" if is_revolute else "linear"
            joints[joint_name] = (prim, is_revolute, instance_name)

        if not joints:
            return

        drive_cfg = cfg.joint_drive

        # apply drive type (force / acceleration)
        self._set_drive_type_on_joints(joints, drive_cfg)
        # apply target type (none / position / velocity)
        self._set_target_type_on_joints(joints, drive_cfg)
        # apply gains (stiffness / damping)
        self._set_drive_gains_on_joints(joints, drive_cfg)

    # ------------------------------------------------------------------
    # Joint drive helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_drive_type_on_joints(joints: dict, drive_cfg: UrdfConverterCfg.JointDriveCfg):
        """Set the drive type (force or acceleration) on joint prims.

        Args:
            joints: Mapping of joint name → (prim, is_revolute, instance_name).
            drive_cfg: The joint drive configuration.
        """
        from pxr import UsdPhysics

        def _apply(prim, instance_name: str, drive_type: str):
            drive = UsdPhysics.DriveAPI.Get(prim, instance_name)
            type_attr = drive.GetTypeAttr()
            if not type_attr:
                type_attr = drive.CreateTypeAttr()
            type_attr.Set(drive_type)

        if isinstance(drive_cfg.drive_type, str):
            for _name, (prim, _is_rev, inst) in joints.items():
                _apply(prim, inst, drive_cfg.drive_type)
        elif isinstance(drive_cfg.drive_type, dict):
            for pattern, drive_type in drive_cfg.drive_type.items():
                matches = [n for n in joints if re.search(pattern, n)]
                if not matches:
                    raise ValueError(
                        f"Joint name pattern '{pattern}' in drive_type config matched no joints."
                        f" Available joints: {list(joints.keys())}"
                    )
                for name in matches:
                    prim, _, inst = joints[name]
                    _apply(prim, inst, drive_type)

    @staticmethod
    def _set_target_type_on_joints(joints: dict, drive_cfg: UrdfConverterCfg.JointDriveCfg):
        """Set the target type (none, position, velocity) on joint prims.

        For ``"none"``, both stiffness and damping are zeroed out.

        Args:
            joints: Mapping of joint name → (prim, is_revolute, instance_name).
            drive_cfg: The joint drive configuration.
        """
        from pxr import UsdPhysics

        def _apply(prim, instance_name: str, target_type: str):
            drive = UsdPhysics.DriveAPI.Get(prim, instance_name)
            if target_type == "none":
                drive.GetStiffnessAttr().Set(0.0)
                drive.GetDampingAttr().Set(0.0)

        if isinstance(drive_cfg.target_type, str):
            for _name, (prim, _is_rev, inst) in joints.items():
                _apply(prim, inst, drive_cfg.target_type)
        elif isinstance(drive_cfg.target_type, dict):
            for pattern, target_type in drive_cfg.target_type.items():
                matches = [n for n in joints if re.search(pattern, n)]
                if not matches:
                    raise ValueError(
                        f"Joint name pattern '{pattern}' in target_type config matched no joints."
                        f" Available joints: {list(joints.keys())}"
                    )
                for name in matches:
                    prim, _, inst = joints[name]
                    _apply(prim, inst, target_type)

    @staticmethod
    def _set_drive_gains_on_joints(joints: dict, drive_cfg: UrdfConverterCfg.JointDriveCfg):
        """Set stiffness and damping on joint drive APIs.

        For revolute joints the user-facing values (Nm/rad) are converted to the USD
        convention (Nm/deg) by multiplying by ``pi / 180``.

        Args:
            joints: Mapping of joint name → (prim, is_revolute, instance_name).
            drive_cfg: The joint drive configuration.
        """
        from pxr import UsdPhysics

        gains = drive_cfg.gains
        if not isinstance(gains, UrdfConverterCfg.JointDriveCfg.PDGainsCfg):
            return

        def _set_stiffness(prim, instance_name: str, is_revolute: bool, value: float):
            drive = UsdPhysics.DriveAPI.Get(prim, instance_name)
            usd_value = value * math.pi / 180.0 if is_revolute else value
            stiffness_attr = drive.GetStiffnessAttr()
            if not stiffness_attr:
                stiffness_attr = drive.CreateStiffnessAttr()
            stiffness_attr.Set(usd_value)

        def _set_damping(prim, instance_name: str, is_revolute: bool, value: float):
            drive = UsdPhysics.DriveAPI.Get(prim, instance_name)
            usd_value = value * math.pi / 180.0 if is_revolute else value
            damping_attr = drive.GetDampingAttr()
            if not damping_attr:
                damping_attr = drive.CreateDampingAttr()
            damping_attr.Set(usd_value)

        # --- stiffness ---
        if isinstance(gains.stiffness, (float, int)):
            for _name, (prim, is_rev, inst) in joints.items():
                _set_stiffness(prim, inst, is_rev, gains.stiffness)
        elif isinstance(gains.stiffness, dict):
            for pattern, stiffness in gains.stiffness.items():
                matches = [n for n in joints if re.search(pattern, n)]
                if not matches:
                    raise ValueError(
                        f"Joint name pattern '{pattern}' in stiffness config matched no joints."
                        f" Available joints: {list(joints.keys())}"
                    )
                for name in matches:
                    prim, is_rev, inst = joints[name]
                    _set_stiffness(prim, inst, is_rev, stiffness)

        # --- damping ---
        if gains.damping is None:
            return
        if isinstance(gains.damping, (float, int)):
            for _name, (prim, is_rev, inst) in joints.items():
                _set_damping(prim, inst, is_rev, gains.damping)
        elif isinstance(gains.damping, dict):
            for pattern, damping in gains.damping.items():
                matches = [n for n in joints if re.search(pattern, n)]
                if not matches:
                    raise ValueError(
                        f"Joint name pattern '{pattern}' in damping config matched no joints."
                        f" Available joints: {list(joints.keys())}"
                    )
                for name in matches:
                    prim, is_rev, inst = joints[name]
                    _set_damping(prim, inst, is_rev, damping)
