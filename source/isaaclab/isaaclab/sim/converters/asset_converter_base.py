# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import hashlib
import json
import logging
import os
import pathlib
import random
import tempfile
from datetime import datetime
from typing import TYPE_CHECKING

from ...utils import to_dict, validate
from ...utils.assets import check_file_path
from ...utils.io import dump_yaml
from .asset_converter_base_cfg import AssetConverterBaseCfg

if TYPE_CHECKING:
    from pxr import Usd

logger = logging.getLogger(__name__)

_ARTICULATION_ROOT_ATTRIBUTES = ("newton:selfCollisionEnabled", "newton:jointsAddMobility")
"""Attributes of the Newton articulation-root schema, which move with the articulation root."""


class AssetConverterBase(abc.ABC):
    """Base class for converting an asset file from different formats into USD format.

    This class provides a common interface for converting an asset file into USD. It does not
    provide any implementation for the conversion. The derived classes must implement the
    :meth:`_convert_asset` method to provide the actual conversion.

    The file conversion is lazy if the output directory (:obj:`AssetConverterBaseCfg.usd_dir`) is provided.
    In the lazy conversion, the USD file is re-generated only if:

    * The asset file is modified.
    * The configuration parameters are modified.
    * The USD file does not exist.

    To override this behavior to force conversion, the flag :obj:`AssetConverterBaseCfg.force_usd_conversion`
    can be set to True.

    When no output directory is defined, lazy conversion is deactivated and the generated USD file is
    stored in folder ``<tempdir>/IsaacLab/usd_{date}_{time}_{random}``, where ``<tempdir>`` is the system
    temporary directory (e.g. ``/tmp`` on POSIX, ``%TEMP%`` on Windows) and the parameters in braces are
    generated at runtime. The random identifiers help avoid a race condition where two simultaneously
    triggered conversions try to use the same directory for reading/writing the generated files.

    .. note::
        Changes to the parameters :obj:`AssetConverterBaseCfg.asset_path`, :obj:`AssetConverterBaseCfg.usd_dir`, and
        :obj:`AssetConverterBaseCfg.usd_file_name` are not considered as modifications in the configuration instance
        that trigger the USD file re-generation.

    """

    def __init__(self, cfg: AssetConverterBaseCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for converting an asset file to USD format.

        Raises:
            ValueError: When provided asset file does not exist.
        """
        # check that the config is valid
        validate(cfg)
        # check if the asset file exists
        if not check_file_path(cfg.asset_path):
            raise ValueError(f"The asset path does not exist: {cfg.asset_path}")
        # save the inputs
        self.cfg = cfg

        # resolve USD directory name
        if cfg.usd_dir is None:
            # a folder in the system temp dir by the name: IsaacLab/usd_{date}_{time}_{random}
            time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._usd_dir = os.path.join(tempfile.gettempdir(), "IsaacLab", f"usd_{time_tag}_{random.randrange(10000)}")
        else:
            self._usd_dir = cfg.usd_dir

        # resolve the file name from asset file name if not provided
        if cfg.usd_file_name is None:
            usd_file_name = pathlib.PurePath(cfg.asset_path).stem
        else:
            usd_file_name = cfg.usd_file_name
        # add USD extension if not provided
        if not usd_file_name.endswith((".usd", ".usda")):
            usd_file_name += ".usd"
        self._usd_file_name = usd_file_name

        os.makedirs(self.usd_dir, exist_ok=True)
        self._usd_file_exists = os.path.isfile(self.usd_path)
        # the recorded hash tells whether the cached USD was generated from this asset and config
        self._dest_hash_path = os.path.join(self.usd_dir, ".asset_hash")
        self._asset_hash = self._config_to_hash(cfg)
        try:
            with open(self._dest_hash_path) as f:
                self._is_same_asset = f.readline() == self._asset_hash
        except FileNotFoundError:
            self._is_same_asset = False

        # convert the asset to USD if the hash is different or USD file does not exist
        if cfg.force_usd_conversion or not self._usd_file_exists or not self._is_same_asset:
            # convert the asset to USD
            self._convert_asset(cfg)
            # importers put the physics payloads behind a "Physics" variant set and disagree on
            # which variant to select, so settle it here
            self._select_physics_variant(cfg.physics_variant)
            # record the hash only now: writing it earlier would let a conversion that raised
            # still count as cached, so an identical retry would skip it and return the asset
            with open(self._dest_hash_path, "w") as f:
                f.write(self._asset_hash)
            # dump the configuration next to the asset, stamped with the converter that produced it
            config_path = os.path.join(self.usd_dir, "config.yaml")
            dump_yaml(config_path, to_dict(cfg))
            stamp = datetime.now().strftime("%Y-%m-%d at %H:%M:%S")
            with open(config_path, "a") as f:
                f.write(f"##\n# Generated by {self.__class__.__name__} on {stamp}.\n##\n")

    """
    Properties.
    """

    @property
    def usd_dir(self) -> str:
        """The absolute path to the directory where the generated USD files are stored."""
        return self._usd_dir

    @property
    def usd_file_name(self) -> str:
        """The file name of the generated USD file."""
        return self._usd_file_name

    @property
    def usd_path(self) -> str:
        """The absolute path to the generated USD file."""
        return os.path.join(self.usd_dir, self.usd_file_name)

    @property
    def usd_instanceable_meshes_path(self) -> str:
        """The relative path to the USD file with meshes.

        The path is with respect to the USD directory :attr:`usd_dir`. This is to ensure that the
        mesh references in the generated USD file are resolved relatively. Otherwise, it becomes
        difficult to move the USD asset to a different location.
        """
        return os.path.join(".", "Props", "instanceable_meshes.usd")

    """
    Implementation specifics.
    """

    @abc.abstractmethod
    def _convert_asset(self, cfg: AssetConverterBaseCfg):
        """Converts the asset file to USD.

        Args:
            cfg: The configuration instance for the input asset to USD conversion.
        """
        raise NotImplementedError()

    """
    Private helpers.
    """

    def _select_physics_variant(self, variant: str):
        """Author a selection for the ``"Physics"`` variant set on the converted asset.

        Importers put the physics description behind a ``"Physics"`` variant set, and which variant
        they select is not consistent: the Isaac Sim importer extensions leave the set unselected,
        which composes the asset without joints, articulation roots, or mass properties, while the
        standalone importer wheel selects one of its own. Authoring the configured variant here makes
        the outcome the same either way. The selection is authored on the asset, so a spawner that
        selects a different variant on the referencing prim still wins.

        Does nothing when the asset has no such variant set.

        Args:
            variant: The variant to select.

        Raises:
            ValueError: When the asset offers a ``"Physics"`` variant set without the requested
                variant. Substituting another one would silently hand back an asset configured for a
                different backend than the caller asked for.
        """
        from pxr import Usd

        stage = Usd.Stage.Open(self.usd_path)
        prim = stage.GetDefaultPrim()
        if not prim or "Physics" not in prim.GetVariantSets().GetNames():
            return
        variant_set = prim.GetVariantSets().GetVariantSet("Physics")
        available = variant_set.GetVariantNames()
        if variant not in available:
            raise ValueError(
                f"The converted asset has no '{variant}' physics variant. Set"
                f" {type(self.cfg).__name__}.physics_variant to one of: {available}."
            )
        if variant == variant_set.GetVariantSelection():
            return
        variant_set.SetVariantSelection(variant)
        stage.GetRootLayer().Save()

    def _root_articulations_at_world_joints(self, layered: bool):
        """Root each fixed-base articulation of the converted asset at the fixed joint that attaches it to the world.

        With ``fix_base``, the Isaac Sim importers keep a fixed world joint that the asset already has, such as the
        root joint that the URDF importer adds or the weld of an MJCF body without joints, but leave the articulation
        root on the root body. UsdPhysics roots a fixed-base articulation at its world joint or at an ancestor of
        that joint, while an articulation rooted at a rigid body is floating: PhysX simulates such an asset as a
        floating articulation that the world joint only holds as an additional constraint. The importers root the
        fixed joint that they create themselves in the same way. A root body on a revolute, prismatic, or D6 world
        joint keeps the root, since PhysX would weld an articulation rooted at that joint.

        The importers author the articulation roots in the physics layer, which the other physics layers include.
        Isaac Sim's ``fix_articulation_root_for_fixed_base`` does not apply here: it skips the prims that the
        physics layer only overrides, and it also roots non-fixed world joints.

        Remove this correction when the pinned Isaac Sim importers root an existing world joint as well
        (isaac-sim/IsaacSim#859).

        Args:
            layered: Whether the asset transformer of the importer split the asset into layers.
        """
        from pxr import Usd, UsdPhysics

        if layered:
            usd_path = os.path.join(os.path.dirname(self.usd_path), "payloads", "Physics", "physics.usda")
        else:
            usd_path = self.usd_path
        if not os.path.isfile(usd_path):
            logger.warning(f"Cannot root the articulations at their world joints: '{usd_path}' does not exist.")
            return
        stage = Usd.Stage.Open(usd_path)
        # the physics layer only adds opinions under the asset's prims, so it has to be traversed with the
        # all-prims predicate to reach them
        prims = list(stage.TraverseAll())
        joints = [prim for prim in prims if prim.IsA(UsdPhysics.FixedJoint)]
        modified = False
        for body in prims:
            if not (body.HasAPI(UsdPhysics.ArticulationRootAPI) and body.HasAPI(UsdPhysics.RigidBodyAPI)):
                continue
            world_joint = next((joint for joint in joints if _attaches_to_world(joint, body)), None)
            if world_joint is not None:
                _move_articulation_root(body, world_joint)
                modified = True
        if modified:
            stage.GetRootLayer().Save()

    @staticmethod
    def _config_to_hash(cfg: AssetConverterBaseCfg) -> str:
        """Converts the configuration object and asset file to an MD5 hash of a string.

        .. warning::
            It only checks the main asset file (:attr:`cfg.asset_path`).

        Args:
            config : The asset converter configuration object.

        Returns:
            An MD5 hash of a string.
        """

        # convert to dict and remove path related info
        config_dic = to_dict(cfg)
        _ = config_dic.pop("asset_path")
        _ = config_dic.pop("usd_dir")
        _ = config_dic.pop("usd_file_name")
        # convert config dic to bytes
        config_bytes = json.dumps(config_dic).encode()
        # hash config
        md5 = hashlib.md5()
        md5.update(config_bytes)

        # read the asset file to observe changes
        with open(cfg.asset_path, "rb") as f:
            while True:
                # read 64kb chunks to avoid memory issues for the large files!
                data = f.read(65536)
                if not data:
                    break
                md5.update(data)
        # return the hash
        return md5.hexdigest()


def _attaches_to_world(joint: Usd.Prim, body: Usd.Prim) -> bool:
    """Return whether an articulation joint attaches a body, as its second body, to the world.

    UsdPhysics attaches a joint to the closest rigid-body ancestor of its target, and to the world without one.
    """
    from pxr import UsdPhysics

    joint_api = UsdPhysics.Joint(joint)
    if not joint.IsActive() or not joint_api.GetJointEnabledAttr().Get():
        return False
    if joint_api.GetExcludeFromArticulationAttr().Get():
        return False
    if joint_api.GetBody1Rel().GetTargets() != [body.GetPath()]:
        return False
    targets = joint_api.GetBody0Rel().GetTargets()
    prim = joint.GetStage().GetPrimAtPath(targets[0]) if targets else None
    while prim and not prim.IsPseudoRoot():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return False
        prim = prim.GetParent()
    return True


def _move_articulation_root(source: Usd.Prim, target: Usd.Prim):
    """Move the articulation-root schemas and their authored attributes from one prim to another.

    These are the schemas and attributes that the Isaac Sim importers move to a fixed joint that they create. The
    schema names are read from the authored ``apiSchemas``, since USD leaves schemas without a registered
    definition out of :meth:`~pxr.Usd.Prim.GetAppliedSchemas`, which happens without Kit.

    Raises:
        RuntimeError: If a schema or an attribute cannot be moved.
    """
    from pxr import Sdf

    api_schemas = source.GetMetadata("apiSchemas") or Sdf.TokenListOp()
    names = [
        name
        for name in api_schemas.GetAddedOrExplicitItems()
        if "ArticulationRoot" in name or name == "PhysxArticulationAPI"
    ]
    for name in names:
        if not target.AddAppliedSchema(name):
            raise RuntimeError(f"Failed to apply '{name}' to '{target.GetPath()}'.")
    for attr in source.GetAttributes():
        name = attr.GetName()
        is_root_attribute = name.startswith("physxArticulation:") or name in _ARTICULATION_ROOT_ATTRIBUTES
        if not (is_root_attribute and attr.HasAuthoredValue()):
            continue
        if not target.CreateAttribute(name, attr.GetTypeName(), custom=False).Set(attr.Get()):
            raise RuntimeError(f"Failed to move '{attr.GetPath()}' to '{target.GetPath()}'.")
        if not source.RemoveProperty(name):
            raise RuntimeError(f"Failed to remove '{attr.GetPath()}'.")
    for name in names:
        if not source.RemoveAppliedSchema(name):
            raise RuntimeError(f"Failed to remove '{name}' from '{source.GetPath()}'.")
