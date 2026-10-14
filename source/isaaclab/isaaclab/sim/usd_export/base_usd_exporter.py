# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from pxr import Gf, Plug, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

from isaaclab import cloner
from isaaclab.utils.assets import check_file_path, retrieve_file_path, unmirror_file_path

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

logger = logging.getLogger(__name__)

WORLD_POSE = "isaaclab:worldPose"
"""Pseudo attribute of a :class:`UsdEdit` that places a prim at a world transform."""


class UsdEdit(NamedTuple):
    """One value the exported USD must author so that an import reproduces the engine."""

    path: str
    """Prim path in the exported stage."""

    attribute: str
    """Attribute name, or :data:`WORLD_POSE` for a world transform given as a :class:`Gf.Matrix4d`."""

    value: Any
    """Value in USD units."""

    type_name: str = "float"
    """Value type used when the attribute does not exist yet."""

    schema: str | None = None
    """Applied API schema the importer requires for the attribute, e.g. ``"PhysicsDriveAPI:angular"``."""


class BaseUsdExporter(ABC):
    """Export one environment of an initialized scene: its USD plus the values its engine holds.

    The exported layer is the scene's in-memory root layer reduced to one environment, so assets
    stay references to their source files. A backend writes every engine value that the layer does
    not already state, so the file does not depend on any importer's defaults.
    """

    def __init__(self, scene: InteractiveScene, env_id: int = 0):
        """Initialize the exporter.

        Args:
            scene: Scene whose assets are initialized and whose physics has not stepped yet.
            env_id: Environment to export.
        """
        if not 0 <= env_id < scene.num_envs:
            raise ValueError(f"Environment {env_id} is outside [0, {scene.num_envs}).")
        if scene.sim.get_physics_step_count() != 0:
            raise ValueError("USD export must precede the first physics step.")
        self.scene = scene
        self.env_id = env_id

    def export(self, path: str) -> str:
        """Write the environment to a USD file.

        Args:
            path: Destination USD file.

        Returns:
            The destination path.
        """
        _schema_defaults.cache_clear()  # include schemas registered since the last export
        layer = Sdf.Layer.CreateAnonymous(".usda")
        layer.TransferContent(self.scene.stage.GetRootLayer())
        stage = Usd.Stage.Open(layer)
        self._select_environment(stage)
        for edit in self.compute_diff(stage):
            _write(stage, edit)
        # Cached copies of remote assets exist only on this machine; reference their source instead.
        UsdUtils.ModifyAssetPaths(layer, lambda asset: unmirror_file_path(asset) or asset)
        UsdUtils.ModifyAssetPaths(layer, functools.partial(_keep_temporary_file, export_path=path))
        if not layer.Export(path):
            raise RuntimeError(f"Could not write USD to '{path}'.")
        return path

    @abstractmethod
    def compute_diff(self, stage: Usd.Stage) -> list[UsdEdit]:
        """Return the engine values that ``stage`` does not already state.

        Args:
            stage: The exported stage, reduced to the selected environment.
        """

    def _select_environment(self, stage: Usd.Stage):
        """Keep only the selected environment, copying prototypes the cloner replicated in physics only."""
        plan = self.scene.clone_plan
        if plan is not None:
            layer = stage.GetRootLayer()
            templates, starts = cloner.path.get_world_prototype_asset_templates(plan)
            sources = cloner.path.get_asset_prototype_paths(plan)
            assets = np.asarray(plan.topology.world_prototypes)
            group = int(np.asarray(plan.topology.world_prototype_layout)[self.env_id]) + 1
            for index in range(int(starts[group]), int(starts[group + 1])):
                target = templates[index].format(self.env_id)
                source = sources[assets[index]]
                if source is not None and not stage.GetPrimAtPath(target):
                    Sdf.CreatePrimInLayer(layer, Sdf.Path(target).GetParentPath())
                    Sdf.CopySpec(layer, source, layer, target)
        for index, env_path in enumerate(self.scene.env_prim_paths):
            if index != self.env_id:
                stage.RemovePrim(env_path)


def _keep_temporary_file(asset: str, export_path: str) -> str:
    """Copy a file the run wrote to the temporary directory next to the export, and reference the copy.

    Such files, e.g. re-saved actuator networks, are removed with the machine's temporary directory.
    USD layers stay referenced in place, since a copy would not carry the layers they reference.
    """
    temporary = os.path.realpath(tempfile.gettempdir())
    source = os.path.realpath(asset) if os.path.isabs(asset) else None
    if source is None or not os.path.isfile(source) or Sdf.FileFormat.FindByExtension(source):
        return asset
    if os.path.commonpath([source, temporary]) != temporary:
        return asset
    directory = os.path.dirname(os.path.abspath(export_path))
    folder = os.path.splitext(os.path.abspath(export_path))[0] + "_files"
    os.makedirs(folder, exist_ok=True)
    return "./" + os.path.relpath(shutil.copy2(source, folder), directory)


def open_exported_stage(path: str) -> Usd.Stage:
    """Open an exported USD file, downloading referenced remote assets as Isaac Lab does when spawning them.

    Args:
        path: Exported USD file.

    Returns:
        A stage whose layer references local copies of the remote assets.
    """
    source = Sdf.Layer.FindOrOpen(path)
    if source is None:
        raise FileNotFoundError(f"Could not open '{path}'.")
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.TransferContent(source)
    UsdUtils.ModifyAssetPaths(
        layer,
        lambda asset: retrieve_file_path(asset) if check_file_path(asset) == 2 else source.ComputeAbsolutePath(asset),
    )
    return Usd.Stage.Open(layer)


UNLIMITED = 1e10
"""Engine values of this magnitude mean "no limit", which an unset USD attribute already states.

Newton's ``MAXVAL`` sentinel is 1e10; PhysX uses the float maximum.
"""


def stated(prim: Usd.Prim, attribute: str) -> Any:
    """Value the USD authors for ``attribute`` on ``prim``, or None if it states none."""
    value = prim.GetAttribute(attribute) if prim else None
    return value.Get() if value and value.HasAuthoredValue() else None


def schema_default(attribute: str) -> Any:
    """Value the registered USD schemas give ``attribute`` when unset.

    Returns None if no schema defines one, schemas disagree, or the default is non-finite, which
    leaves the value to the engine.
    """
    parts = attribute.split(":")
    templates = [":".join([*parts[:i], "__INSTANCE_NAME__", *parts[i + 1 :]]) for i in range(len(parts))]
    return next((_schema_defaults()[name] for name in (attribute, *templates) if name in _schema_defaults()), None)


@functools.cache
def _schema_defaults() -> dict[str, Any]:
    """Finite default of every attribute in the registered schema plugins; multi-apply names are templates.

    Read from each plugin's generated schema, since the schema registry misses plugins registered after its first use.
    """
    defaults = {}
    for plugin in Plug.Registry().GetAllPlugins():
        path = plugin.FindPluginResource("generatedSchema.usda", True)
        layer = Sdf.Layer.FindOrOpen(path) if path else None
        for prim in layer.rootPrims if layer else ():
            for attribute in prim.attributes:
                value = attribute.default if _finite(attribute.default) else None
                defaults[attribute.name] = value if defaults.get(attribute.name, value) == value else None
    return defaults


def _finite(value) -> bool:
    if isinstance(value, Gf.Quatf):
        return True
    try:
        return bool(np.all(np.isfinite(np.asarray(value, dtype=float))))
    except (TypeError, ValueError):
        return False


def value_edit(prim: Usd.Prim, attribute: str, value, type_name: str = "float", schema: str | None = None):
    """Edit writing ``value`` [USD units] unless the USD states it already, authored or as its schema default.

    An unset "no limit" stays unset.

    Args:
        prim: Prim owning the attribute.
        attribute: Attribute name.
        value: Scalar, vector, or quaternion (x, y, z, w) value.
        type_name: Value type of the attribute. Defaults to ``"float"``.
        schema: Applied API schema the attribute requires. Defaults to None.

    Returns:
        The edit, or None if nothing needs writing.
    """
    array = np.asarray(value, dtype=float)
    current = stated(prim, attribute)
    if current is None:
        if array.size == 1 and not abs(array.item()) < UNLIMITED:
            return None
        current = schema_default(attribute)
    if current is not None:
        if type_name == "quatf":
            current = np.asarray([*current.GetImaginary(), current.GetReal()])
            if _close(current, array) or _close(current, -array):
                return None
        elif _close(np.asarray(current, dtype=float), array):
            return None
    if type_name == "quatf":
        usd_value = Gf.Quatf(float(array[3]), Gf.Vec3f(*array[:3].tolist()))
    elif array.size == 3:
        usd_value = Gf.Vec3f(*array.tolist())
    elif type_name == "bool":
        usd_value = bool(array.item())
    else:
        usd_value = float(array.item())
    return UsdEdit(str(prim.GetPath()), attribute, usd_value, type_name, schema)


def mass_edits(prim: Usd.Prim, mass: float, com: np.ndarray, inertia: np.ndarray) -> list[UsdEdit]:
    """Edits writing a body's mass [kg], COM [m] and body-axes inertia [kg*m^2, 3x3 or 9] the USD does not state."""
    edits = [
        value_edit(prim, "physics:mass", mass, schema="PhysicsMassAPI"),
        value_edit(prim, "physics:centerOfMass", com, "point3f", "PhysicsMassAPI"),
    ]
    inertia = np.asarray(inertia, dtype=float).reshape(3, 3)
    moments, axes = stated(prim, "physics:diagonalInertia"), stated(prim, "physics:principalAxes") or Gf.Quatf(1)
    if moments is not None:
        rotation = np.asarray(Gf.Matrix3d(Gf.Quatd(axes))).T
        if _close(rotation @ np.diag(np.asarray(moments, dtype=float)) @ rotation.T, inertia):
            return [edit for edit in edits if edit is not None]
    moments, axes = principal_inertia(inertia)
    edits.append(UsdEdit(str(prim.GetPath()), "physics:diagonalInertia", moments, "float3", "PhysicsMassAPI"))
    edits.append(UsdEdit(str(prim.GetPath()), "physics:principalAxes", axes, "quatf", "PhysicsMassAPI"))
    return [edit for edit in edits if edit is not None]


def pose_edit(prim: Usd.Prim, pose: np.ndarray) -> UsdEdit | None:
    """Edit placing ``prim`` at a world pose (x, y, z, qx, qy, qz, qw) unless it is there already."""
    world = Gf.Transform(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
    rotation = world.GetRotation().GetQuat()
    current = np.asarray([*world.GetTranslation(), *rotation.GetImaginary(), rotation.GetReal()])
    pose = np.asarray(pose, dtype=float)
    if _close(current[:3], pose[:3]) and (_close(current[3:], pose[3:]) or _close(current[3:], -pose[3:])):
        return None
    x, y, z, w = pose[3:7].tolist()
    matrix = Gf.Matrix4d().SetRotate(Gf.Quatd(w, Gf.Vec3d(x, y, z))).SetTranslateOnly(Gf.Vec3d(*pose[:3].tolist()))
    return UsdEdit(str(prim.GetPath()), WORLD_POSE, matrix)


def gravity_edits(stage: Usd.Stage, scene_path: str, gravity) -> list[UsdEdit]:
    """Edits making the physics scene's USD gravity equal ``gravity`` [m/s^2]."""
    scene = UsdPhysics.Scene(stage.GetPrimAtPath(scene_path))
    direction, magnitude = scene.GetGravityDirectionAttr().Get(), scene.GetGravityMagnitudeAttr().Get()
    gravity = np.asarray(gravity, dtype=float)
    if direction is not None and magnitude is not None and _close(np.asarray(direction) * magnitude, gravity):
        return []
    norm = float(np.linalg.norm(gravity))
    direction = gravity / norm if norm else np.array([0.0, 0.0, -1.0])
    return [
        UsdEdit(scene_path, "physics:gravityDirection", Gf.Vec3f(*direction.tolist()), "vector3f"),
        UsdEdit(scene_path, "physics:gravityMagnitude", norm),
    ]


def _close(a, b) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=1e-5, atol=1e-7))


def material_edits(
    stage: Usd.Stage,
    values: dict[str, dict[str, float]],
    without_material: Callable[[str, dict[str, float]], list[UsdEdit]] | None = None,
) -> list[UsdEdit]:
    """Edits writing collider material values that the colliders' bound physics materials do not state.

    Colliders sharing a material must have equal values. A stated value is never changed on a material
    also used by colliders missing from ``values``; unset values may still be filled in.

    Args:
        stage: The exported stage.
        values: Engine material values [USD units], keyed by collider path and then by attribute.
        without_material: Edits for a collider without a physics material. Defaults to None, which
            leaves its material values to the importer's default.

    Returns:
        The edits.
    """
    users, groups, unbound, edits = {}, {}, 0, []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            if material:
                users.setdefault(str(material.GetPath()), []).append(str(prim.GetPath()))
    for collider, engine in values.items():
        material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(collider)).ComputeBoundMaterial("physics")
        if not material:
            unbound += 1
            if without_material is not None:
                edits += without_material(collider, engine)
            continue
        path = str(material.GetPath())
        if path in groups and not all(_close(groups[path][name], value) for name, value in engine.items()):
            raise NotImplementedError(f"Colliders sharing material '{path}' have different values.")
        groups[path] = engine
    for path, engine in groups.items():
        prim = stage.GetPrimAtPath(path)
        pending = [
            value_edit(prim, name, value, schema="PhysicsMaterialAPI" if name.startswith("physics:") else None)
            for name, value in engine.items()
        ]
        pending = [edit for edit in pending if edit is not None]
        if any(user not in values for user in users[path]) and any(
            stated(prim, e.attribute) is not None for e in pending
        ):
            raise NotImplementedError(f"Material '{path}' changed but is shared with colliders not exported.")
        edits += pending
    if unbound:
        logger.warning(f"{unbound} colliders have no physics material; their friction stays the importer's default.")
    return edits


def export_configured_scene(scene: InteractiveScene) -> str | None:
    """Export the scene to its configured path during environment setup.

    A failed export is logged rather than raised, so training continues without the file.

    Args:
        scene: Initialized scene with :attr:`~isaaclab.scene.InteractiveSceneCfg.export_usd_path` set.

    Returns:
        The written path, or None if the export failed.
    """
    path = scene.cfg.export_usd_path
    try:
        return scene.export_to_usd(path)
    except Exception:
        logger.exception(f"USD export to '{path}' failed; continuing without it.")
        return None


def principal_inertia(inertia: np.ndarray) -> tuple[Gf.Vec3f, Gf.Quatf]:
    """Split a body-frame inertia tensor [kg*m^2] into USD principal moments and principal axes."""
    if np.allclose(inertia, np.diag(np.diag(inertia))):
        # Keep a diagonal tensor in body axes; sorting its moments would rotate them.
        return Gf.Vec3f(*np.maximum(np.diag(inertia), 0.0).tolist()), Gf.Quatf(1)
    moments, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    # Gf matrices act on row vectors, so the rotation taking principal to body axes is transposed.
    rotation = Gf.Matrix3d(*axes.T.flatten().tolist()).ExtractRotation().GetQuat()
    return Gf.Vec3f(*np.maximum(moments, 0.0).tolist()), Gf.Quatf(rotation)


def _write(stage: Usd.Stage, edit: UsdEdit):
    """Author one edit, un-instancing the asset instance that contains the prim."""
    prim = stage.GetPrimAtPath(edit.path)
    if not prim:
        raise RuntimeError(f"Exported stage has no prim at '{edit.path}'.")
    while prim.IsInstanceProxy():
        instance = prim.GetParent()
        while instance.IsInstanceProxy():
            instance = instance.GetParent()
        instance.SetInstanceable(False)
        prim = stage.GetPrimAtPath(edit.path)
    if edit.attribute == WORLD_POSE:
        # Replace the local transform, keeping the prim's scale.
        cache = UsdGeom.XformCache()
        scale = Gf.Transform(cache.GetLocalToWorldTransform(prim)).GetScale()
        parent = cache.GetLocalToWorldTransform(prim.GetParent())
        local = Gf.Matrix4d().SetScale(scale) * edit.value * parent.GetInverse()
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(local)
        return
    if edit.schema and edit.schema not in prim.GetAppliedSchemas():
        prim.AddAppliedSchema(edit.schema)
    attribute = prim.GetAttribute(edit.attribute)
    if not attribute:
        attribute = prim.CreateAttribute(edit.attribute, Sdf.ValueTypeNames.Find(edit.type_name))
    attribute.Set(edit.value)
