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
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING

from ...utils import to_dict, validate
from ...utils.assets import check_file_path
from ...utils.io import dump_yaml
from .asset_converter_base_cfg import AssetConverterBaseCfg

if TYPE_CHECKING:
    from pxr import Sdf, Usd

logger = logging.getLogger(__name__)

_JOINT_AXIS_INSTANCES = {
    "PhysicsRevoluteJoint": ("angular",),
    "PhysicsPrismaticJoint": ("linear",),
    "PhysicsSphericalJoint": ("rotX", "rotY", "rotZ"),
}
"""PhysX joint-axis instances of the revolute, prismatic, and spherical joint types."""

_D6_AXIS_INSTANCES = {
    "PhysicsRevoluteJoint": {"X": "rotX", "Y": "rotY", "Z": "rotZ"},
    "PhysicsPrismaticJoint": {"X": "transX", "Y": "transY", "Z": "transZ"},
}
"""PhysX joint-axis instance that a single-axis joint maps to when the MJCF importer folds it into a D6 joint.

This mirrors the ``physics:axis`` lookup with which the importer assigns the D6 axes. It has to follow the
importer if that assignment changes, as proposed in isaac-sim/IsaacSim#751.
"""


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

    def _convert_joint_dynamics_to_physx(self, layered: bool):
        """Author the joint friction, damping, and armature in the PhysX description of the converted asset.

        The importers write these passive joint dynamics with ``NewtonJointAPI``, whose friction, damping, and
        armature PhysX does not read, and do not convert them to ``PhysxJointAxisAPI``: the URDF importer converts
        neither the friction nor the damping (isaac-sim/IsaacSim#841), and the MJCF importer converts the friction to
        the legacy, load-proportional ``physxJoint:jointFriction`` coefficient and drops the damping. The friction
        becomes equal static and dynamic friction efforts [N or N·m, depending on joint type], and the damping
        becomes the viscous friction coefficient. Both schemas use the USD units, which are per degree for angular
        axes, so the values are copied unchanged. ``physxJoint:armature`` holds a single value per joint, so
        spherical joints and the joints that the MJCF importer folds into one D6 joint get their armature per axis
        as well.

        Remove this correction when the pinned importers author these PhysX attributes themselves, or when PhysX
        reads them from ``NewtonJointAPI``. An importer that maps the damping to the joint drive instead would apply
        it twice; the URDF converter tests detect that.

        Args:
            layered: Whether the asset transformer of the importer split the asset into layers.
        """
        from pxr import Usd

        # a layered asset keeps the Newton and the PhysX attributes in separate physics layers, while a flat
        # asset keeps both in the generated USD file
        if layered:
            # layout written by the asset transformer profile of the importers
            physics_dir = os.path.join(os.path.dirname(self.usd_path), "payloads", "Physics")
            physx_layer_path = os.path.join(physics_dir, "physx.usda")
            if not os.path.isfile(physx_layer_path):
                # an asset without PhysX data, e.g. with only fixed or ball joints, gets no PhysX layer and no
                # "physx" variant
                return
            # the physics layer still holds the joints that the PhysX layer folds into D6 joints
            source_stage = Usd.Stage.Open(os.path.join(physics_dir, "physics.usda"))
            target_stage = Usd.Stage.Open(physx_layer_path)
        else:
            source_stage = target_stage = Usd.Stage.Open(self.usd_path)

        # the physics layers only add opinions under the asset's prims, so they have to be traversed with the
        # all-prims predicate to reach the joints
        d6_joints = {
            _body_pair(prim): prim
            for prim in target_stage.TraverseAll()
            if prim.IsActive() and prim.GetTypeName() == "PhysicsJoint"
        }
        d6_used_axes = defaultdict(set)
        modified = False
        for joint in source_stage.TraverseAll():
            joint_type = joint.GetTypeName()
            if joint_type not in _JOINT_AXIS_INSTANCES:
                continue
            # unauthored Newton attributes hold their zero default
            friction = joint.GetAttribute("newton:friction").Get() or 0.0
            viscous = joint.GetAttribute("newton:damping").Get() or 0.0
            armature = joint.GetAttribute("newton:armature").Get() or 0.0
            target = target_stage.GetPrimAtPath(joint.GetPath())
            if target and target.IsActive() and target.GetTypeName() == joint_type:
                instances = _JOINT_AXIS_INSTANCES[joint_type]
                if joint_type != "PhysicsSphericalJoint":
                    # physxJoint:armature already holds it
                    armature = None
                if not (friction or viscous or armature):
                    continue
            else:
                target = d6_joints.get(_body_pair(joint))
                axis = str(joint.GetAttribute("physics:axis").Get()).upper()
                instance = _D6_AXIS_INSTANCES.get(joint_type, {}).get(axis)
                if target is None or instance is None or instance in d6_used_axes[target.GetPath()]:
                    # the importer gave this joint no D6 axis of its own and warned; that axis stays locked or
                    # free without the joint's dynamics (see isaac-sim/IsaacLab#6854)
                    continue
                # author the axis even when its values are zero, since it would otherwise fall back to the
                # single armature that the D6 joint copied from its first joint
                d6_used_axes[target.GetPath()].add(instance)
                instances = (instance,)
            for instance in instances:
                _author_physx_joint_axis(target, instance, friction, viscous, armature)
            modified = True

        if modified:
            target_stage.GetRootLayer().Save()

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


def _body_pair(prim: Usd.Prim) -> tuple[tuple[Sdf.Path, ...], tuple[Sdf.Path, ...]]:
    """Return the body targets of a joint prim."""
    from pxr import UsdPhysics

    joint = UsdPhysics.Joint(prim)
    return tuple(joint.GetBody0Rel().GetTargets()), tuple(joint.GetBody1Rel().GetTargets())


def _author_physx_joint_axis(joint: Usd.Prim, instance: str, friction: float, viscous: float, armature: float | None):
    """Author the friction and armature of one axis of a PhysX joint and clear the legacy friction.

    Args:
        joint: The joint prim to author on.
        instance: The ``PhysxJointAxisAPI`` instance, e.g. ``"angular"`` or ``"rotX"``.
        friction: The static and dynamic friction effort [N or N·m, depending on joint type].
        viscous: The viscous friction coefficient in USD units [N·s/m or N·m·s/deg, depending on joint type].
        armature: The armature of the axis [kg or kg·m², depending on joint type]. If None, it is not authored.
    """
    from pxr import Sdf

    joint.AddAppliedSchema(f"PhysxJointAxisAPI:{instance}")
    values = {
        "staticFrictionEffort": friction,
        "dynamicFrictionEffort": friction,
        "viscousFrictionCoefficient": viscous,
    }
    if armature is not None:
        values["armature"] = armature
    for name, value in values.items():
        attr_name = f"physxJointAxis:{instance}:{name}"
        joint.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float, custom=False).Set(float(value))
    # the friction efforts replace the load-proportional legacy coefficient
    legacy_friction = joint.GetAttribute("physxJoint:jointFriction")
    if legacy_friction and legacy_friction.HasAuthoredValue():
        legacy_friction.Set(0.0)
