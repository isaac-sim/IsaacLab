# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import fcntl
import logging
import os
import shutil
import tempfile
from typing import TYPE_CHECKING

from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab.sim import converters, schemas
from isaaclab.utils.string import to_camel_case
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
from isaaclab.utils.assets import check_file_path, retrieve_file_path
from isaaclab.utils.version import has_kit

if TYPE_CHECKING:
    from . import from_files_cfg

# import logger
logger = logging.getLogger(__name__)


def _create_modified_usd_with_overrides(
    usd_path: str, cfg: "from_files_cfg.UsdFileCfg"
) -> str | None:
    """Create a modified copy of USD with physics properties baked in for instanced prims.

    File I/O is required because USD prototypes are read-only after composition.
    This function modifies source layers to set physics schema attributes and
    optionally creates randomizable visual materials for per-body color randomization.
    """
    # Collect physics properties to apply
    prop_schema_map = {
        "collision_props": ("CollisionAPI", "physxCollision:"),
        "rigid_props": ("RigidBodyAPI", "physxRigidBody:"),
        "articulation_props": ("ArticulationRootAPI", "physxArticulation:"),
        "joint_drive_props": ("DriveAPI", "physxJoint:"),
        "mass_props": ("MassAPI", "physics:"),
        "deformable_props": ("DeformableAPI", "physxDeformable:"),
    }
    props = []
    for name, (api, prefix) in prop_schema_map.items():
        prop_cfg = getattr(cfg, name, None)
        if prop_cfg:
            props.append((prop_cfg, api, prefix))

    # Check if we need to create randomizable visual materials
    create_visual_materials = getattr(cfg, "randomizable_visual_materials", False)

    if not props and not create_visual_materials:
        return None

    # Copy USD and all referenced layers to temp directory
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        return None
    layers = [l for l in stage.GetUsedLayers() if not l.anonymous]
    paths = [l.identifier for l in layers]
    del stage

    common_root = os.path.commonpath(paths) if len(paths) > 1 else os.path.dirname(paths[0])
    temp_dir = tempfile.mkdtemp(prefix="isaaclab_usd_")
    path_map = {p: os.path.join(temp_dir, os.path.relpath(p, common_root)) for p in paths}
    for src, dst in path_map.items():
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    temp_path = path_map[usd_path]
    temp_stage = Usd.Stage.Open(temp_path)
    if not temp_stage:
        return None

    # Modify physics properties in each layer
    modified = False
    for layer in (l for l in temp_stage.GetUsedLayers() if not l.anonymous):
        stack = list(layer.rootPrims.values())
        while stack:
            prim_spec = stack.pop()
            stack.extend(prim_spec.nameChildren.values())

            # Get applied API schemas
            schemas = prim_spec.GetInfo("apiSchemas")
            apis = set()
            if schemas:
                for attr in ("explicitItems", "prependedItems", "appendedItems"):
                    apis.update(str(s) for s in getattr(schemas, attr, []))

            # Apply matching properties
            for prop_cfg, api_schema, prefix in props:
                if not any(api_schema in a for a in apis):
                    continue
                modified = True
                for field, value in prop_cfg.to_dict().items():
                    if value is None or field.startswith("_") or hasattr(value, "to_dict"):
                        continue
                    vtype = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int,
                             float: Sdf.ValueTypeNames.Float}.get(type(value))
                    if vtype:
                        attr_name = f"{prefix}{to_camel_case(field, 'cC')}"
                        path = prim_spec.path.AppendProperty(attr_name)
                        attr = layer.GetAttributeAtPath(path) or Sdf.AttributeSpec(
                            prim_spec, attr_name, vtype)
                        attr.default = value
        layer.Save()

    # Create randomizable visual materials in the source layer
    if create_visual_materials:
        materials_modified = _add_visual_materials_to_layer(temp_stage, cfg)
        modified = modified or materials_modified

    del temp_stage
    return temp_path if modified else None


def _add_visual_materials_to_layer(stage: Usd.Stage, cfg: "from_files_cfg.FileCfg") -> bool:
    """Create UsdPreviewSurface materials for per-body color randomization.

    This function enables per-body color randomization while keeping visual geometry
    INSTANCED for memory efficiency. Supports two modes:

    - ``"full"``: Each visual body part gets its own unique material (fully random colors)
    - ``"style"``: Body parts sharing the same original material share a randomizable material
      (preserves color groupings, like changing the robot's color scheme)

    The approach:
    1. Find link prims that have 'visuals' children
    2. Create UsdPreviewSurface materials in a RandomizableMaterials container
    3. Bind materials to 'visuals' prims with 'strongerThanDescendants' binding strength
    4. The binding strength overrides instance proxy bindings, enabling per-env colors
       without breaking geometry instancing

    Args:
        stage: The USD stage (opened from temp copy).
        cfg: The configuration instance.

    Returns:
        True if materials were added, False otherwise.
    """
    from pxr import UsdShade, Usd  # noqa: PLC0415

    root_layer = stage.GetRootLayer()
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        logger.warning("[randomizable_visual_materials] No default prim found")
        return False

    default_prim_path = default_prim.GetPath()
    stage.Load()

    # Get mode: "full" (each body gets unique material) or "style" (preserve material groups)
    mode = getattr(cfg, "randomizable_visual_materials_mode", "full")

    # Find link prims that have a 'visuals' child (these are NOT instance proxies)
    link_prims_with_visuals = []
    for prim in stage.TraverseAll():
        path_str = str(prim.GetPath())
        if "/visuals" in path_str or "/collision" in path_str:
            continue
        visuals_child = prim.GetChild("visuals")
        if visuals_child and visuals_child.IsValid():
            link_prims_with_visuals.append(prim)

    if not link_prims_with_visuals:
        logger.info("[randomizable_visual_materials] No link prims with visuals found")
        return False

    # Create materials container
    materials_path = getattr(cfg, "randomizable_visual_materials_path", "RandomizableMaterials")
    container_path = f"{default_prim_path}/{materials_path}"

    container_spec = Sdf.CreatePrimInLayer(root_layer, container_path)
    container_spec.specifier = Sdf.SpecifierDef
    container_spec.typeName = "Scope"

    # For "style" mode, detect original material bindings and group by material name
    original_material_map = {}  # visuals_path -> original_material_name
    if mode == "style":
        for link_prim in link_prims_with_visuals:
            visuals_prim = link_prim.GetChild("visuals")
            if not visuals_prim:
                continue
            # Find the first mesh inside visuals and get its material binding
            for child in stage.Traverse(Usd.TraverseInstanceProxies()):
                if not child.IsA(UsdGeom.Mesh):
                    continue
                child_path = str(child.GetPath())
                visuals_path = str(visuals_prim.GetPath())
                if not child_path.startswith(visuals_path):
                    continue
                if "collision" in child_path.lower():
                    continue
                # Get the bound material
                binding_api = UsdShade.MaterialBindingAPI(child)
                bound_mat, _ = binding_api.ComputeBoundMaterial()
                if bound_mat:
                    mat_name = bound_mat.GetPrim().GetName()
                    original_material_map[visuals_path] = mat_name
                break

    # Create materials - one per group in "style" mode, one per link in "full" mode
    created_materials = {}  # material_name -> mat_path (for style mode)
    mat_index = 0

    def _create_material(mat_name: str) -> Sdf.Path:
        """Create a UsdPreviewSurface material and return its path."""
        nonlocal mat_index
        mat_path = Sdf.Path(f"{container_path}/mat_{mat_index}")
        shader_path = mat_path.AppendChild("Shader")
        mat_index += 1

        # Create Material prim
        mat_spec = Sdf.CreatePrimInLayer(root_layer, mat_path)
        mat_spec.specifier = Sdf.SpecifierDef
        mat_spec.typeName = "Material"

        # Create UsdPreviewSurface Shader
        shader_spec = Sdf.CreatePrimInLayer(root_layer, shader_path)
        shader_spec.specifier = Sdf.SpecifierDef
        shader_spec.typeName = "Shader"

        id_attr = Sdf.AttributeSpec(shader_spec, "info:id", Sdf.ValueTypeNames.Token)
        id_attr.default = "UsdPreviewSurface"

        diffuse_attr = Sdf.AttributeSpec(shader_spec, "inputs:diffuseColor", Sdf.ValueTypeNames.Color3f)
        diffuse_attr.default = Gf.Vec3f(0.8, 0.8, 0.8)

        Sdf.AttributeSpec(shader_spec, "outputs:surface", Sdf.ValueTypeNames.Token)
        mat_surface_attr = Sdf.AttributeSpec(mat_spec, "outputs:surface", Sdf.ValueTypeNames.Token)
        mat_surface_attr.connectionPathList.explicitItems = [shader_path.AppendProperty("outputs:surface")]

        return mat_path

    # Bind materials to visuals prims
    for link_prim in link_prims_with_visuals:
        link_path = link_prim.GetPath()
        visuals_path = link_path.AppendChild("visuals")
        visuals_path_str = str(visuals_path)

        # Determine which material to use
        if mode == "style" and visuals_path_str in original_material_map:
            original_mat_name = original_material_map[visuals_path_str]
            if original_mat_name not in created_materials:
                created_materials[original_mat_name] = _create_material(original_mat_name)
            mat_path = created_materials[original_mat_name]
        else:
            # Full mode: create unique material for each
            mat_path = _create_material(f"body_{mat_index}")

        # Create override for the VISUALS prim to add material binding
        # KEY: Use 'strongerThanDescendants' to override instance proxy bindings
        visuals_spec = root_layer.GetPrimAtPath(visuals_path)
        if not visuals_spec:
            visuals_spec = Sdf.CreatePrimInLayer(root_layer, visuals_path)
            visuals_spec.specifier = Sdf.SpecifierOver

        # Add MaterialBindingAPI
        visuals_spec.SetInfo("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["MaterialBindingAPI"]))

        # Create material:binding with 'strongerThanDescendants' binding strength
        binding_rel = Sdf.RelationshipSpec(visuals_spec, "material:binding", custom=False)
        binding_rel.targetPathList.explicitItems = [mat_path]
        binding_rel.SetInfo("bindMaterialAs", "strongerThanDescendants")

    root_layer.Save()
    num_materials = mat_index
    if mode == "style":
        logger.info(
            f"[randomizable_visual_materials] Style mode: created {num_materials} materials "
            f"for {len(link_prims_with_visuals)} visuals (grouped by original material)"
        )
    else:
        logger.info(
            f"[randomizable_visual_materials] Full mode: created {num_materials} materials "
            f"for {len(link_prims_with_visuals)} visuals (one per body)"
        )

    return True


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
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
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
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
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
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
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
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
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
    # Remove the light from the ground plane (USD API, works without Kit/Newton)
    # It isn't bright enough and messes up with the user's lighting settings
    light_prim = stage.GetPrimAtPath(f"{prim_path}/SphereLight")
    if light_prim.IsValid():
        imageable = UsdGeom.Imageable(light_prim)
        imageable.MakeInvisible()

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
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the generated USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the USD file does not exist at the given path.
    """
    # In distributed training, serialize asset download and USD stage composition
    # across ranks to prevent file I/O races. Concurrent mmap reads/writes on
    # the same cached USD files cause segfaults in Sdf_CrateFile::_MmapStream::Read.
    _world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

    file_status = check_file_path(usd_path)
    if file_status == 0:
        raise FileNotFoundError(f"USD file not found at path: '{usd_path}'.")

    # Handle physics properties via layer copying to preserve instancing
    # Note: We cannot avoid file I/O here because:
    # 1. USD prototypes are read-only after stage composition
    # 2. Modifying prototype prims at runtime doesn't propagate to instance proxies
    # 3. The only way to modify instanced prim properties is to modify the source layer
    spawn_usd_path = usd_path
    resolved_path = usd_path if file_status != 2 else retrieve_file_path(usd_path, force_download=False)
    modified_path = _create_modified_usd_with_overrides(resolved_path, cfg)
    if modified_path:
        spawn_usd_path = modified_path

    if _world_size > 1:
        lock_path = os.path.join(tempfile.gettempdir(), "isaaclab_usd_spawn.lock")
        lock_fd = open(lock_path, "w")  # noqa: SIM115
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        if file_status == 2 and spawn_usd_path == usd_path:
            spawn_usd_path = retrieve_file_path(usd_path, force_download=False)
        stage = get_current_stage()
        if not stage.GetPrimAtPath(prim_path).IsValid():
            create_prim(
                prim_path,
                usd_path=spawn_usd_path,
                translation=translation,
                orientation=orientation,
                scale=cfg.scale,
                stage=stage,
            )
        else:
            logger.warning(f"A prim already exists at prim path: '{prim_path}'.")
    finally:
        if _world_size > 1:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    # modify variants
    if hasattr(cfg, "variants") and cfg.variants is not None:
        select_usd_variants(prim_path, cfg.variants)

    # Modify physics properties
    # Note: Most properties are handled via layer copying above to preserve instancing.
    # Tendons use multi-instance API schemas (e.g. PhysxTendonAxisRootAPI:tendon0) which
    # require special handling via runtime modification.
    if cfg.fixed_tendons_props is not None:
        schemas.modify_fixed_tendon_properties(prim_path, cfg.fixed_tendons_props)
    if cfg.spatial_tendons_props is not None:
        schemas.modify_spatial_tendon_properties(prim_path, cfg.spatial_tendons_props)

    # apply visual material
    if cfg.visual_material is not None:
        if not has_kit():
            logger.warning("Skipping visual material application for '%s' in kitless mode.", prim_path)
            return stage.GetPrimAtPath(prim_path)
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{prim_path}/{cfg.visual_material_path}"
        else:
            material_path = cfg.visual_material_path
        # create material
        cfg.visual_material.func(material_path, cfg.visual_material)
        # apply material
        bind_visual_material(prim_path, material_path, stage=stage)

    # Note: randomizable_visual_materials are created at the layer level in
    # _create_modified_usd_with_overrides to enable per-body color randomization.
    # They get copied with the asset when cloner uses Sdf.CopySpec.

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
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
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
