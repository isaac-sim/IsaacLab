# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.sim import converters, schemas
from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg, RigidObjectSpawnerCfg, SpawnerCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass


@configclass
class FileCfg(RigidObjectSpawnerCfg, DeformableObjectSpawnerCfg):
    """Configuration parameters for spawning an asset from a file.

    This class is a base class for spawning assets from files. It includes the common parameters
    for spawning assets from files, such as the path to the file and the function to use for spawning
    the asset.

    Note:
        By default, all properties are set to None. This means that no properties will be added or modified
        to the prim outside of the properties available by default when spawning the prim.

        If they are set to a value, then the properties are modified on the spawned prim in a nested manner.
        This is done by calling the respective function with the specified properties.
    """

    scale: tuple[float, float, float] | None = None
    """Scale of the asset. Defaults to None, in which case the scale is not modified."""

    articulation_props: dict[str, list[schemas.ArticulationRootFragment]] | schemas.ArticulationRootBaseCfg | None = (
        None
    )
    """Properties to apply to the articulation root.

    Accepts either a mapping from target pattern to a list of
    :class:`~isaaclab.sim.schemas.ArticulationRootFragment` fragments
    (e.g. ``{"**": [PhysxArticulationCfg(...), NewtonArticulationCfg(...)]}``) or a single legacy
    cfg (e.g. :class:`~isaaclab.sim.schemas.ArticulationRootBaseCfg`). On the fragment path each
    fragment writes its own namespace.

    Keys are target patterns relative to the spawn prim. Each ``/``-separated token is a regular
    expression matched per level, a trailing ``**`` token matches a prim and all its descendants,
    and an empty string targets the spawn prim itself. Entries are applied in insertion order, so
    on overlapping targets later entries override earlier ones per attribute.
    """

    articulation_props_create_if_missing: bool = False
    """Whether the articulation writer may apply ``UsdPhysics.ArticulationRootAPI`` when no matched
    prim carries it. Defaults to False. The flag applies to every entry of the
    :attr:`articulation_props` mapping.

    Creation applies to every matched prim lacking the API; when enabling this, narrow the
    pattern -- typically ``{"": [...]}`` to anchor the spawn prim itself. Only consumed when
    :attr:`articulation_props` is given as fragments.
    """

    fix_root_link: bool | None = None
    """Whether to fix the root link of the articulation. Defaults to None.

    This is a non-USD, spawner-level behaviour flag consumed by
    :func:`~isaaclab.sim.schemas.apply_articulation_root_properties` on the fragment/topology path,
    including when :attr:`articulation_props` is ``None`` or an empty mapping. When the mapping has
    several entries, the flag is honored on the first entry only, since the root topology must not
    be re-fixed per entry. It is handled independently of whether any schema properties are
    supplied:

    * If set to None, the root link is not modified.
    * If the articulation already has a fixed root link, this flag enables or disables the fixed joint.
    * If the articulation does not have a fixed root link, this flag creates a fixed joint between the
      world frame and the root link (named "FixedJoint" under the articulation prim).

    When :attr:`articulation_props` is given as a legacy cfg, set
    :attr:`~isaaclab.sim.schemas.ArticulationRootBaseCfg.fix_root_link` on that cfg instead.
    """

    fixed_tendons_props: dict[str, list[schemas.FixedTendonFragment]] | schemas.FixedTendonPropertiesCfg | None = None
    """Properties to apply to the fixed tendons (if any).

    Accepts either a mapping from target pattern to a list of
    :class:`~isaaclab.sim.schemas.FixedTendonFragment` fragments or the legacy
    :class:`~isaaclab_physx.sim.schemas.PhysxFixedTendonPropertiesCfg`.

    Keys are target patterns relative to the spawn prim. Each ``/``-separated token is a regular
    expression matched per level, a trailing ``**`` token matches a prim and all its descendants,
    and an empty string targets the spawn prim itself. Entries are applied in insertion order, so
    on overlapping targets later entries override earlier ones per attribute.
    """

    spatial_tendons_props: (
        dict[str, list[schemas.SpatialTendonFragment]] | schemas.SpatialTendonPropertiesCfg | None
    ) = None
    """Properties to apply to the spatial tendons (if any).

    Accepts either a mapping from target pattern to a list of
    :class:`~isaaclab.sim.schemas.SpatialTendonFragment` fragments or the legacy
    :class:`~isaaclab_physx.sim.schemas.PhysxSpatialTendonPropertiesCfg`.

    Keys are target patterns relative to the spawn prim. Each ``/``-separated token is a regular
    expression matched per level, a trailing ``**`` token matches a prim and all its descendants,
    and an empty string targets the spawn prim itself. Entries are applied in insertion order, so
    on overlapping targets later entries override earlier ones per attribute.
    """

    joint_drive_props: dict[str, list[schemas.JointDriveFragment]] | schemas.JointDriveBaseCfg | None = None
    """Properties to apply to a joint.

    Accepts either a mapping from target pattern to a list of
    :class:`~isaaclab.sim.schemas.JointDriveFragment` fragments
    (e.g. ``{"**": [UsdPhysicsDriveCfg(...), PhysxJointCfg(...)]}``) or a single legacy cfg
    (e.g. :class:`~isaaclab.sim.schemas.JointDriveBaseCfg`). On the fragment path,
    ``UsdPhysics.DriveAPI`` is applied (presence-gated) only when a
    :class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg` fragment is present, and each fragment writes
    its own namespace.

    Keys are target patterns relative to the spawn prim. Each ``/``-separated token is a regular
    expression matched per level, a trailing ``**`` token matches a prim and all its descendants,
    and an empty string targets the spawn prim itself. Entries are applied in insertion order, so
    on overlapping targets later entries override earlier ones per attribute.

    .. note::
        The joint drive properties set the USD attributes of all the joint drives in the asset.
        We recommend using this attribute sparingly and only when necessary. Instead, please use the
        :attr:`~isaaclab.assets.ArticulationCfg.actuators` parameter to set the joint drive properties
        for specific joints in an articulation.
    """

    joint_drive_props_create_if_missing: bool = False
    """Whether the joint-drive writer may apply the defining USD drive API to matched joint prims
    that lack it. Defaults to False. The flag applies to every entry of the
    :attr:`joint_drive_props` mapping.

    Only consumed when :attr:`joint_drive_props` is given as fragments. This is independent of
    :attr:`ensure_drives_exist`, which instead patches zero-gain drives with a minimal stiffness.
    """

    ensure_drives_exist: bool = False
    """Whether to ensure every joint drive is active when authoring :attr:`joint_drive_props`.

    When True, any joint drive whose authored stiffness *and* damping are both zero is given a
    minimal stiffness (``1e-3``) so that backends (e.g. Newton) create proper actuators for it.
    This is a spawner-level behavior flag (not a USD attribute and not a fragment field). It is
    only consumed when :attr:`joint_drive_props` is given as fragments, and applies to every entry
    of the mapping; legacy :class:`~isaaclab.sim.schemas.JointDriveBaseCfg` cfgs carry their own
    ``ensure_drives_exist`` field.
    """

    visual_material_path: str = "material"
    """Path to the visual material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `visual_material` is not None.
    """

    visual_material: materials.VisualMaterialCfg | None = None
    """Visual material properties to override the visual material properties in the URDF file.

    Note:
        If None, then no visual material will be added.
    """

    physics_material_path: str = "material"
    """Path to the physics material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `physics_material` is not None.
    """

    physics_material: (
        materials.PhysicsMaterialCfg
        | materials.RigidBodyMaterialFragment
        | list[materials.RigidBodyMaterialFragment]
        | None
    ) = None
    """Physics material properties.

    Accepts either a legacy material cfg, a single
    :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialFragment`, or a list of such
    single-namespace fragments.

    Note:
        If None, then no custom physics material will be added.
    """


@configclass
class UsdFileCfg(FileCfg):
    """USD file to spawn asset from.

    USD files are imported directly into the scene. However, given their complexity, there are various different
    operations that can be performed on them. For example, selecting variants, applying materials, or modifying
    existing properties.

    To prevent the explosion of configuration parameters, the available operations are limited to the most common
    ones. These include:

    - **Selecting variants**: This is done by specifying the :attr:`variants` parameter.
    - **Creating and applying materials**: This is done by specifying the :attr:`visual_material` parameter.
    - **Modifying existing properties**: This is done by specifying the respective properties in the configuration
      class. For instance, to modify the scale of the imported prim, set the :attr:`scale` parameter.

    See :meth:`spawn_from_usd` for more information.

    .. note::
        The configuration parameters include various properties. If not `None`, these properties
        are modified on the spawned prim in a nested manner.

        If they are set to a value, then the properties are modified on the spawned prim in a nested manner.
        This is done by calling the respective function with the specified properties.
    """

    func: Callable | str = "{DIR}.from_files:spawn_from_usd"

    usd_path: str = MISSING
    """Path to the USD file to spawn asset from."""

    variants: object | dict[str, str] | None = None
    """Variants to select from in the input USD file. Defaults to None, in which case no variants are applied.

    This can either be a configclass object, in which case each attribute is used as a variant set name and
    its specified value, or a dictionary mapping between the two. Please check the
    :meth:`~isaaclab.sim.utils.select_usd_variants` function for more information.
    """


@configclass
class UrdfFileCfg(FileCfg, converters.UrdfConverterCfg):
    """URDF file to spawn asset from.

    It uses the :class:`UrdfConverter` class to create a USD file from URDF and spawns the imported
    USD file. Similar to the :class:`UsdFileCfg`, the generated USD file can be modified by specifying
    the respective properties in the configuration class.

    See :meth:`spawn_from_urdf` for more information.

    .. note::
        The configuration parameters include various properties. If not `None`, these properties
        are modified on the spawned prim in a nested manner.

        If they are set to a value, then the properties are modified on the spawned prim in a nested manner.
        This is done by calling the respective function with the specified properties.

    """

    func: Callable | str = "{DIR}.from_files:spawn_from_urdf"


@configclass
class MjcfFileCfg(FileCfg, converters.MjcfConverterCfg):
    """MJCF file to spawn asset from.

    It uses the :class:`MjcfConverter` class to create a USD file from MJCF and spawns the imported
    USD file. Similar to the :class:`UsdFileCfg`, the generated USD file can be modified by specifying
    the respective properties in the configuration class.

    See :meth:`spawn_from_mjcf` for more information.

    .. note::
        The configuration parameters include various properties. If not `None`, these properties
        are modified on the spawned prim in a nested manner.

        If they are set to a value, then the properties are modified on the spawned prim in a nested manner.
        This is done by calling the respective function with the specified properties.

    """

    func: Callable | str = "{DIR}.from_files:spawn_from_mjcf"


"""
Spawning ground plane.
"""


@configclass
class UsdFileWithCompliantContactCfg(UsdFileCfg):
    """Configuration for spawning a USD asset with compliant contact physics material.

    This class extends :class:`UsdFileCfg` to support applying compliant contact properties
    (stiffness and damping) to specific prims in the spawned asset. It uses the
    :meth:`spawn_from_usd_with_compliant_contact_material` function to perform the spawning and
    material application.
    """

    func: Callable | str = "{DIR}.from_files:spawn_from_usd_with_compliant_contact_material"

    compliant_contact_stiffness: float | None = None
    """Stiffness of the compliant contact. Defaults to None.

    This parameter is the same as
    :attr:`~isaaclab.sim.spawners.materials.RigidBodyMaterialCfg.compliant_contact_stiffness`.
    """

    compliant_contact_damping: float | None = None
    """Damping of the compliant contact. Defaults to None.

    This parameter is the same as
    :attr:`isaaclab.sim.spawners.materials.RigidBodyMaterialCfg.compliant_contact_damping`.
    """

    physics_material_prim_path: str | list[str] | None = None
    """Path to the prim or prims to apply the physics material to. Defaults to None, in which case the
    physics material is not applied.

    If the path is relative, then it will be relative to the prim's path.
    If None, then the physics material will not be applied.
    """


@configclass
class GroundPlaneCfg(SpawnerCfg):
    """Create a ground plane prim.

    This uses the USD for the standard grid-world ground plane from Isaac Sim by default.
    """

    func: Callable | str = "{DIR}.from_files:spawn_ground_plane"

    usd_path: str = f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd"
    """Path to the USD file to spawn asset from. Defaults to the grid-world ground plane."""

    color: tuple[float, float, float] | None = (0.0, 0.0, 0.0)
    """The color of the ground plane. Defaults to (0.0, 0.0, 0.0).

    If None, then the color remains unchanged.
    """

    size: tuple[float, float] = (100.0, 100.0)
    """The size of the ground plane. Defaults to 100 m x 100 m."""

    physics_material: (
        materials.RigidBodyMaterialBaseCfg
        | materials.RigidBodyMaterialFragment
        | list[materials.RigidBodyMaterialFragment]
    ) = materials.RigidBodyMaterialCfg()
    """Physics material properties. Defaults to the default rigid body material.

    The ground plane only spawns a collision plane, so this only accepts rigid-body materials: a
    legacy :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg`, a single
    :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialFragment`, or a list of such
    single-namespace fragments.
    """
