# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import isaacsim.core.prims as prims
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import omni.kit.commands
import omni.log
from pxr import Gf, Sdf, Usd

from isaaclab.sim import converters, schemas
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone, select_usd_variants

if TYPE_CHECKING:
    from . import from_files_cfg


@clone
def spawn_from_usd(
    prim_path: str,
    cfg: from_files_cfg.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
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

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the USD file does not exist at the given path.
    """
    # spawn asset from the given usd file
    return _spawn_from_usd_file(
        prim_path_in_created_stage=prim_path,
        usd_path=cfg.usd_path,
        prim_path_in_usd=cfg.prim_path,
        cfg=cfg,
        translation=translation,
        orientation=orientation,
    )


@clone
def spawn_from_urdf(
    prim_path: str,
    cfg: from_files_cfg.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
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

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the URDF file does not exist at the given path.
    """
    # urdf loader to convert urdf to usd
    urdf_loader = converters.UrdfConverter(cfg)
    # spawn asset from the generated usd file
    return _spawn_from_usd_file(
        prim_path_in_created_stage=prim_path,
        usd_path=urdf_loader.usd_path,
        cfg=cfg,
        translation=translation,
        orientation=orientation,
    )


def spawn_ground_plane(
    prim_path: str,
    cfg: from_files_cfg.GroundPlaneCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
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

    Returns:
        The prim of the spawned asset.

    Raises:
        ValueError: If the prim path already exists.
    """
    # Spawn Ground-plane
    if not prim_utils.is_prim_path_valid(prim_path):
        prim_utils.create_prim(prim_path, usd_path=cfg.usd_path, translation=translation, orientation=orientation)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    # Create physics material
    if cfg.physics_material is not None:
        cfg.physics_material.func(f"{prim_path}/physicsMaterial", cfg.physics_material)
        # Apply physics material to ground plane
        collision_prim_path = prim_utils.get_prim_path(
            prim_utils.get_first_matching_child_prim(
                prim_path, predicate=lambda x: prim_utils.get_prim_type_name(x) == "Plane"
            )
        )
        bind_physics_material(collision_prim_path, f"{prim_path}/physicsMaterial")

    # Scale only the mesh
    # Warning: This is specific to the default grid plane asset.
    if prim_utils.is_prim_path_valid(f"{prim_path}/Environment"):
        # compute scale from size
        scale = (cfg.size[0] / 100.0, cfg.size[1] / 100.0, 1.0)
        # apply scale to the mesh
        prim_utils.set_prim_property(f"{prim_path}/Environment", "xformOp:scale", scale)

    # Change the color of the plane
    # Warning: This is specific to the default grid plane asset.
    if cfg.color is not None:
        prop_path = f"{prim_path}/Looks/theGrid/Shader.inputs:diffuse_tint"
        # change the color
        omni.kit.commands.execute(
            "ChangePropertyCommand",
            prop_path=Sdf.Path(prop_path),
            value=Gf.Vec3f(*cfg.color),
            prev=None,
            type_to_create_if_not_exist=Sdf.ValueTypeNames.Color3f,
        )
    # Remove the light from the ground plane
    # It isn't bright enough and messes up with the user's lighting settings
    omni.kit.commands.execute("ToggleVisibilitySelectedPrims", selected_paths=[f"{prim_path}/SphereLight"])

    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


"""
Helper functions.
"""


def _create_prim(
    prim_path_in_created_stage: str,
    usd_path: str,
    prim_path_in_usd: str | None = None,
    prim_type: str = "Xform",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    scale: tuple[float, float, float] | None = None,
) -> Usd.Prim | None:
    """Create a prim into current USD stage.

    This is similar to `omni.isaac.utils.prims.create_prim` with the addition
    that it allows to use a specific prim path from the USD file.

    Args:
        prim_path: The path of the new prim.
        prim_type: Prim type name
        position: prim position (applied last)
        translation: prim translation (applied last)
        orientation: prim rotation as quaternion
        scale: scaling factor in x, y, z.
        usd_path: Path to the USD that this prim will reference.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the USD file does not exist at the given path.
    """
    # Create prim in stage.
    prim = prim_utils.define_prim(prim_path=prim_path_in_created_stage, prim_type=prim_type)
    if not prim:
        return None

    # Add reference to USD file.
    if usd_path is not None:
        omni.log.info(f"Loading Asset from path {usd_path}.")

        if prim_path_in_usd is not None:
            # Check that the prim path in the usd file is valid. Unfortunately
            # `AddReference` does not raise an error if the prim path is invalid
            # so we have to manually check this here. This means that the USD is
            # loaded twice.
            stage = Usd.Stage.Open(usd_path)
            target_prim = stage.GetPrimAtPath(prim_path_in_usd)
            if not target_prim.IsValid():
                raise ValueError(f"The prim path {prim_path_in_usd} does not exist in the USD file {usd_path}.")
            success_bool = prim.GetReferences().AddReference(usd_path, primPath=Sdf.Path(prim_path_in_usd))
        else:
            success_bool = prim.GetReferences().AddReference(usd_path)
        if not success_bool:
            raise FileNotFoundError(f"The usd file at path {usd_path} provided wasn't found")

    # Get used device.
    from isaacsim.core.api.simulation_context.simulation_context import SimulationContext

    if SimulationContext.instance() is None:
        import isaacsim.core.utils.numpy as backend_utils

        device = "cpu"
    else:
        backend_utils = SimulationContext.instance().backend_utils
        device = SimulationContext.instance().device

    # Apply the transformations.
    if translation is not None:
        translation = backend_utils.expand_dims(backend_utils.convert(translation, device), 0)
    if orientation is not None:
        orientation = backend_utils.expand_dims(backend_utils.convert(orientation, device), 0)
    if scale is not None:
        scale = backend_utils.expand_dims(backend_utils.convert(scale, device), 0)
    prims.XFormPrim(prim_path_in_created_stage, translations=translation, orientations=orientation, scales=scale)
    return prim


def _spawn_from_usd_file(
    prim_path_in_created_stage: str,
    usd_path: str,
    cfg: from_files_cfg.FileCfg,
    prim_path_in_usd: str | None = None,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
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

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the USD file does not exist at the given path.
    """
    # check file path exists
    stage: Usd.Stage = stage_utils.get_current_stage()
    if not stage.ResolveIdentifierToEditTarget(usd_path):
        raise FileNotFoundError(f"USD file not found at path: '{usd_path}'.")
    # spawn asset if it doesn't exist.
    if not prim_utils.is_prim_path_valid(prim_path_in_created_stage):
        # add prim as reference to stage
        _create_prim(
            prim_path_in_created_stage=prim_path_in_created_stage,
            usd_path=usd_path,
            prim_path_in_usd=prim_path_in_usd,
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
        )
    else:
        omni.log.warn(f"A prim already exists at prim path: '{prim_path_in_created_stage}'.")

    # modify variants
    if hasattr(cfg, "variants") and cfg.variants is not None:
        select_usd_variants(prim_path_in_created_stage, cfg.variants)

    # modify rigid body properties
    if cfg.rigid_props is not None:
        schemas.modify_rigid_body_properties(prim_path_in_created_stage, cfg.rigid_props)
    # modify collision properties
    if cfg.collision_props is not None:
        schemas.modify_collision_properties(prim_path_in_created_stage, cfg.collision_props)
    # modify mass properties
    if cfg.mass_props is not None:
        schemas.modify_mass_properties(prim_path_in_created_stage, cfg.mass_props)

    # modify articulation root properties
    if cfg.articulation_props is not None:
        schemas.modify_articulation_root_properties(prim_path_in_created_stage, cfg.articulation_props)
    # modify tendon properties
    if cfg.fixed_tendons_props is not None:
        schemas.modify_fixed_tendon_properties(prim_path_in_created_stage, cfg.fixed_tendons_props)
    # define drive API on the joints
    # note: these are only for setting low-level simulation properties. all others should be set or are
    #  and overridden by the articulation/actuator properties.
    if cfg.joint_drive_props is not None:
        schemas.modify_joint_drive_properties(prim_path_in_created_stage, cfg.joint_drive_props)

    # modify deformable body properties
    if cfg.deformable_props is not None:
        schemas.modify_deformable_body_properties(prim_path_in_created_stage, cfg.deformable_props)

    # apply visual material
    if cfg.visual_material is not None:
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{prim_path_in_created_stage}/{cfg.visual_material_path}"
        else:
            material_path = cfg.visual_material_path
        # create material
        cfg.visual_material.func(material_path, cfg.visual_material)
        # apply material
        bind_visual_material(prim_path_in_created_stage, material_path)

    # return the prim
    return prim_utils.get_prim_at_path(prim_path_in_created_stage)
