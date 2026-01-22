# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import omni.kit.commands
from pxr import Gf, Sdf, Usd

from isaaclab.sim import converters, schemas
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.sim.utils import (
    add_labels,
    bind_physics_material,
    bind_visual_material,
    change_prim_property,
    clone,
    create_prim,
    get_current_stage,
    get_first_matching_child_prim,
    select_usd_variants,
    set_prim_visibility,
)
from isaaclab.utils.assets import check_usd_path_with_timeout

if TYPE_CHECKING:
    from . import from_files_cfg

# import logger
logger = logging.getLogger(__name__)


@clone
def spawn_from_usd(
    prim_path: str,
    cfg: from_files_cfg.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an asset from a USD file and override the settings with the given config.

    In the case of a USD file, the asset is spawned at the default prim specified in the USD file.
    If a default prim is not specified, then the asset is spawned at the root prim.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the USD file is used.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the USD file does not exist at the given path.
    """
    # spawn asset from the given usd file
    return _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)


@clone
def spawn_from_urdf(
    prim_path: str,
    cfg: from_files_cfg.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an asset from a URDF file and override the settings with the given config.

    It uses the :class:`UrdfConverter` class to create a USD file from URDF. This file is then imported
    at the specified prim path.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the generated USD file is used.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the generated USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the URDF file does not exist at the given path.
    """
    # urdf loader to convert urdf to usd
    urdf_loader = converters.UrdfConverter(cfg)
    # spawn asset from the generated usd file
    return _spawn_from_usd_file(prim_path, urdf_loader.usd_path, cfg, translation, orientation)


@clone
def spawn_from_mjcf(
    prim_path: str,
    cfg: from_files_cfg.MjcfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn an asset from a MJCF file and override the settings with the given config.

    It uses the :class:`MjcfConverter` class to create a USD file from MJCF. This file is then imported
    at the specified prim path.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the generated USD file is used.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the generated USD file is used.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the MJCF file does not exist at the given path.
    """
    # mjcf loader to convert mjcf to usd
    mjcf_loader = converters.MjcfConverter(cfg)
    # spawn asset from the generated usd file
    return _spawn_from_usd_file(prim_path, mjcf_loader.usd_path, cfg, translation, orientation)


def spawn_ground_plane(
    prim_path: str,
    cfg: from_files_cfg.GroundPlaneCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawns a ground plane into the scene.

    This function loads the USD file containing the grid plane asset from Isaac Sim. It may
    not work with other assets for ground planes. In those cases, please use the `spawn_from_usd`
    function.

    Note:
        This function takes keyword arguments to be compatible with other spawners. However, it does not
        use any of the kwargs.

    Args:
        prim_path: The path to spawn the asset at.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the USD file is used.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset.

    Raises:
        ValueError: If the prim path already exists.
    """
    # Obtain current stage
    stage = get_current_stage()

    # Spawn Ground-plane
    if not stage.GetPrimAtPath(prim_path).IsValid():
        create_prim(prim_path, usd_path=cfg.usd_path, translation=translation, orientation=orientation, stage=stage)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    # Create physics material
    if cfg.physics_material is not None:
        cfg.physics_material.func(f"{prim_path}/physicsMaterial", cfg.physics_material)
        # Apply physics material to ground plane
        collision_prim = get_first_matching_child_prim(
            prim_path,
            predicate=lambda _prim: _prim.GetTypeName() == "Plane",
            stage=stage,
        )
        if collision_prim is None:
            raise ValueError(f"No collision prim found at path: '{prim_path}'.")
        # bind physics material to the collision prim
        collision_prim_path = str(collision_prim.GetPath())
        bind_physics_material(collision_prim_path, f"{prim_path}/physicsMaterial", stage=stage)

    # Obtain environment prim
    environment_prim = stage.GetPrimAtPath(f"{prim_path}/Environment")
    # Scale only the mesh
    # Warning: This is specific to the default grid plane asset.
    if environment_prim.IsValid():
        # compute scale from size
        scale = (cfg.size[0] / 100.0, cfg.size[1] / 100.0, 1.0)
        # apply scale to the mesh
        environment_prim.GetAttribute("xformOp:scale").Set(scale)

    # Change the color of the plane
    # Warning: This is specific to the default grid plane asset.
    if cfg.color is not None:
        # change the color
        change_prim_property(
            prop_path=f"{prim_path}/Looks/theGrid/Shader.inputs:diffuse_tint",
            value=Gf.Vec3f(*cfg.color),
            stage=stage,
            type_to_create_if_not_exist=Sdf.ValueTypeNames.Color3f,
        )
    # Remove the light from the ground plane
    # It isn't bright enough and messes up with the user's lighting settings
    omni.kit.commands.execute("ToggleVisibilitySelectedPrims", selected_paths=[f"{prim_path}/SphereLight"], stage=stage)

    prim = stage.GetPrimAtPath(prim_path)
    # Apply semantic tags
    if hasattr(cfg, "semantic_tags") and cfg.semantic_tags is not None:
        # note: taken from replicator scripts.utils.utils.py
        for semantic_type, semantic_value in cfg.semantic_tags:
            # deal with spaces by replacing them with underscores
            semantic_type_sanitized = semantic_type.replace(" ", "_")
            semantic_value_sanitized = semantic_value.replace(" ", "_")
            # add labels to the prim
            add_labels(prim, labels=[semantic_value_sanitized], instance_name=semantic_type_sanitized)

    # Apply visibility
    set_prim_visibility(prim, cfg.visible)

    # return the prim
    return prim


"""
Helper functions.
"""


def _spawn_from_usd_file(
    prim_path: str,
    usd_path: str,
    cfg: from_files_cfg.FileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an asset from a USD file and override the settings with the given config.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        usd_path: The path to the USD file to spawn the asset from.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the generated USD file is used.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the generated USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the USD file does not exist at the given path.
    """
    # check if usd path exists with periodic logging until timeout
    if not check_usd_path_with_timeout(usd_path):
        if "4.5" in usd_path:
            usd_5_0_path = usd_path.replace("http", "https").replace("/4.5", "/5.0")
            if not check_usd_path_with_timeout(usd_5_0_path):
                raise FileNotFoundError(f"USD file not found at path at either: '{usd_path}' or '{usd_5_0_path}'.")
            usd_path = usd_5_0_path
        else:
            raise FileNotFoundError(f"USD file not found at path at: '{usd_path}'.")

    # Obtain current stage
    stage = get_current_stage()
    # spawn asset if it doesn't exist.
    if not stage.GetPrimAtPath(prim_path).IsValid():
        # add prim as reference to stage
        create_prim(
            prim_path,
            usd_path=usd_path,
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
            stage=stage,
        )
    else:
        logger.warning(f"A prim already exists at prim path: '{prim_path}'.")

    # modify variants
    if hasattr(cfg, "variants") and cfg.variants is not None:
        select_usd_variants(prim_path, cfg.variants)

    # modify rigid body properties
    if cfg.rigid_props is not None:
        schemas.modify_rigid_body_properties(prim_path, cfg.rigid_props)
    # modify collision properties
    if cfg.collision_props is not None:
        schemas.modify_collision_properties(prim_path, cfg.collision_props)
    # modify mass properties
    if cfg.mass_props is not None:
        schemas.modify_mass_properties(prim_path, cfg.mass_props)

    # modify articulation root properties
    if cfg.articulation_props is not None:
        schemas.modify_articulation_root_properties(prim_path, cfg.articulation_props)
    # modify tendon properties
    if cfg.fixed_tendons_props is not None:
        schemas.modify_fixed_tendon_properties(prim_path, cfg.fixed_tendons_props)
    if cfg.spatial_tendons_props is not None:
        schemas.modify_spatial_tendon_properties(prim_path, cfg.spatial_tendons_props)
    # define drive API on the joints
    # note: these are only for setting low-level simulation properties. all others should be set or are
    #  and overridden by the articulation/actuator properties.
    if cfg.joint_drive_props is not None:
        schemas.modify_joint_drive_properties(prim_path, cfg.joint_drive_props)

    # modify deformable body properties
    if cfg.deformable_props is not None:
        schemas.modify_deformable_body_properties(prim_path, cfg.deformable_props)

    # apply visual material
    if cfg.visual_material is not None:
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{prim_path}/{cfg.visual_material_path}"
        else:
            material_path = cfg.visual_material_path
        # create material
        cfg.visual_material.func(material_path, cfg.visual_material)
        # apply material
        bind_visual_material(prim_path, material_path, stage=stage)

    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_from_usd_with_compliant_contact_material(
    prim_path: str,
    cfg: from_files_cfg.UsdFileWithCompliantContactCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an asset from a USD file and apply physics material to specified prims.

    This function extends the :meth:`spawn_from_usd` function by allowing application of compliant contact
    physics materials to specified prims within the spawned asset. This is useful for configuring
    contact behavior of specific parts within the asset.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance containing the USD file path and physics material settings.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the USD file is used.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset with the physics material applied to the specified prims.

    Raises:
        FileNotFoundError: If the USD file does not exist at the given path.
    """

    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    stiff = cfg.compliant_contact_stiffness
    damp = cfg.compliant_contact_damping
    if cfg.physics_material_prim_path is None:
        logger.warning("No physics material prim path specified. Skipping physics material application.")
        return prim

    if isinstance(cfg.physics_material_prim_path, str):
        prim_paths = [cfg.physics_material_prim_path]
    else:
        prim_paths = cfg.physics_material_prim_path

    if stiff is not None or damp is not None:
        material_kwargs = {}
        if stiff is not None:
            material_kwargs["compliant_contact_stiffness"] = stiff
        if damp is not None:
            material_kwargs["compliant_contact_damping"] = damp
        material_cfg = RigidBodyMaterialCfg(**material_kwargs)

        for path in prim_paths:
            if not path.startswith("/"):
                rigid_body_prim_path = f"{prim_path}/{path}"
            else:
                rigid_body_prim_path = path

            material_path = f"{rigid_body_prim_path}/compliant_material"

            # spawn physics material
            material_cfg.func(material_path, material_cfg)

            bind_physics_material(
                rigid_body_prim_path,
                material_path,
            )
            logger.info(
                f"Applied physics material to prim: {rigid_body_prim_path} with compliance stiffness: {stiff} and"
                f" compliance damping: {damp}."
            )

    return prim
