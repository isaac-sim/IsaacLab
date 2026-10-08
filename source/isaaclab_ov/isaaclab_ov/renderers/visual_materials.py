# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Device-only visual-material writes for OVRTX-owned scenes."""

from __future__ import annotations

import itertools
import logging
import weakref
from typing import TYPE_CHECKING, Any

import torch
import warp as wp
from ovrtx import BindingFlag, DataAccess, PrimMode

from pxr import Sdf, Usd, UsdGeom, UsdShade

from isaaclab.cloner import ClonePlan
from isaaclab.cloner import path as cloner_path
from isaaclab.sim.utils import find_matching_prim_paths

if TYPE_CHECKING:
    from isaaclab.assets import VisualMaterial
    from isaaclab.renderers.base_renderer import VisualMaterialBatch

    from .ovrtx_renderer import OVRTXRenderer

logger = logging.getLogger(__name__)

_COLOR_PRIMVAR = "isaaclab:materialColor"


def _prepare_material_color_primvars(
    usd_string: str, materials: tuple[VisualMaterial, ...], plan: ClonePlan
) -> tuple[str, dict[str, str]]:
    """Route independently owned PreviewSurface colors through a detached scene's constant primvars."""
    material_patterns = [
        material.cfg.prim_path for material in materials if material._input_names.get("color") == "diffuseColor"
    ]
    if not material_patterns:
        return usd_string, {}
    layer = Sdf.Layer.CreateAnonymous()
    layer.ImportFromString(usd_string)
    stage = Usd.Stage.Open(layer)
    material_paths = {path for pattern in material_patterns for path in find_matching_prim_paths(pattern, stage)}
    bindings = {path: [] for path in material_paths}
    external_connections: set[str] = set()
    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            for target in attribute.GetConnections():
                for ancestor in target.GetPrimPath().GetPrefixes():
                    if str(ancestor) in material_paths and not prim.GetPath().HasPrefix(ancestor):
                        external_connections.add(str(ancestor))
        for relation in prim.GetRelationships():
            if relation.GetName().startswith("material:binding"):
                for target in relation.GetForwardedTargets():
                    if str(target) in bindings:
                        bindings[str(target)].append(relation)

    sources = cloner_path.get_asset_prototype_paths(plan)
    templates, starts, world_ids, world_starts = cloner_path.get_world_prototype_asset_templates(
        plan, include_world_indices=True
    )
    destinations = {}
    # Shared assets need no address expansion and do not own a replicated subtree.
    for group in range(1, len(starts) - 1):
        ids = world_ids[world_starts[group] : world_starts[group + 1]]
        for index in range(starts[group], starts[group + 1]):
            source = sources[plan.topology.world_prototypes[index]]
            if source is not None and len(ids):
                destinations.setdefault(source, []).append((templates[index], ids))
    roots = sorted(destinations, key=len, reverse=True)
    addresses = {}
    for material_path, relations in bindings.items():
        if material_path in external_connections or len(relations) != 1 or relations[0].GetName() != "material:binding":
            continue
        if relations[0].GetTargets() != [Sdf.Path(material_path)]:
            continue
        geometry = relations[0].GetPrim()
        if not (geometry.IsA(UsdGeom.Cube) or geometry.IsA(UsdGeom.Mesh)) or geometry.GetChildren():
            continue
        if geometry.IsInstance() or geometry.IsInstanceProxy() or geometry.IsInPrototype():
            continue
        if any(relation.GetName().startswith("material:binding:") for relation in geometry.GetRelationships()):
            continue
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        if not material or material.GetPrim().IsInstance() or material.GetPrim().IsInstanceProxy():
            continue
        if any(
            output.GetFullName() != "outputs:surface" and output.GetAttr().HasAuthoredConnections()
            for output in material.GetSurfaceOutputs()
        ):
            continue
        bound_material, bound_relation = UsdShade.MaterialBindingAPI(geometry).ComputeBoundMaterial()
        if bound_material.GetPrim() != material.GetPrim() or bound_relation != relations[0]:
            continue
        connected = material.GetSurfaceOutput().GetConnectedSource()
        if connected is None:
            continue
        shader = UsdShade.Shader(connected[0].GetPrim())
        if not shader or shader.GetShaderId() != "UsdPreviewSurface":
            continue
        shader_path = shader.GetPath()
        if not shader_path.HasPrefix(Sdf.Path(material_path)):
            continue
        diffuse = shader.GetInput("diffuseColor")
        reader_path = Sdf.Path(material_path).AppendChild("IsaacLabColorReader")
        primvars = UsdGeom.PrimvarsAPI(geometry)
        if (
            not diffuse
            or diffuse.Get() is None
            or diffuse.GetAttr().HasAuthoredConnections()
            or stage.GetPrimAtPath(reader_path)
            or primvars.FindPrimvarWithInheritance(_COLOR_PRIMVAR)
        ):
            continue
        geometry_path = geometry.GetPath()
        owner = next((root for root in roots if Sdf.Path(material_path).HasPrefix(Sdf.Path(root))), None)
        geometry_owner = next((root for root in roots if geometry_path.HasPrefix(Sdf.Path(root))), None)
        if owner != geometry_owner:
            continue
        primvars.CreatePrimvar(_COLOR_PRIMVAR, Sdf.ValueTypeNames.Color3f, UsdGeom.Tokens.constant).Set(diffuse.Get())
        reader = UsdShade.Shader.Define(stage, reader_path)
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(_COLOR_PRIMVAR)
        reader.CreateInput("fallback", Sdf.ValueTypeNames.Float3).Set(diffuse.Get())
        diffuse.ConnectToSource(reader.CreateOutput("result", Sdf.ValueTypeNames.Float3))
        addresses[str(shader_path)] = str(geometry_path)
        if owner is not None:
            for template, ids in destinations[owner]:
                addresses.update(
                    (
                        template.format(world) + str(shader_path)[len(owner) :],
                        template.format(world) + str(geometry_path)[len(owner) :],
                    )
                    for world in ids
                )
    return (stage.ExportToString(), addresses) if addresses else (usd_string, {})


class OVRTXVisualMaterialWriter:
    """Compile OVRTX material addresses once and publish dirty device buffers."""

    def __init__(self, renderer: OVRTXRenderer, batches: tuple[VisualMaterialBatch, ...]):
        self._renderer_ref = weakref.ref(renderer)
        self._buffers: dict[str, torch.Tensor] = {}
        self._dirty_channels: set[str] = set()
        self._operations: tuple[Any, ...] = ()
        self._addresses: list[tuple[str, Any, Any | None, str, slice]] = []

        groups = []
        device = None
        for batch in batches:
            values = batch.values.detach()
            if values.dtype == torch.float32 and values.ndim == 1:
                dtype, shape = "float32", None
            elif values.dtype == torch.float32 and values.ndim == 2 and values.shape[1] in (2, 3):
                dtype, shape = "float32", (values.shape[1],)
            else:
                raise TypeError(
                    f"OVRTX visual-material channel {batch.channel!r} requires float, float2, or float3; "
                    f"got dtype={values.dtype}, shape={tuple(values.shape)}."
                )
            if not values.is_cuda or (device is not None and values.device != device):
                raise RuntimeError("OVRTX visual-material attributes must reside on one CUDA device.")
            device = values.device
            self._buffers[batch.channel] = values
            start = 0
            targets = []
            for shader_path, input_name in zip(batch.shader_paths, batch.input_names, strict=True):
                geometry_path = renderer._visual_material_color_paths.get(shader_path)
                if batch.channel == "color" and input_name == "diffuseColor" and geometry_path is not None:
                    targets.append((f"primvars:{_COLOR_PRIMVAR}", geometry_path))
                else:
                    targets.append((f"inputs:{input_name}", shader_path))
            for attribute_name, target_group in itertools.groupby(targets, key=lambda target: target[0]):
                paths = [target[1] for target in target_group]
                end = start + len(paths)
                rows = slice(start, end)
                groups.append((batch.channel, attribute_name, paths, rows, dtype, shape))
                start = end
        self._device = str(device)
        self._event = wp.Event(device=self._device)
        try:
            for channel, attribute_name, shader_paths, rows, dtype, shape in groups:
                if renderer._use_ovstage:
                    path_list = renderer.backend.paths.create_path_list_from_strings(shader_paths)
                    try:
                        address = renderer.backend.stage.query_from_path_list(path_list)
                    except Exception:
                        renderer.backend.paths.destroy_path_list(path_list)
                        raise
                else:
                    path_list = None
                    address = renderer.backend.renderer.bind_attribute(
                        prim_paths=shader_paths,
                        attribute_name=attribute_name,
                        dtype=dtype,
                        shape=shape,
                        prim_mode=PrimMode.EXISTING_ONLY,
                        flags=BindingFlag.OPTIMIZE,
                    )
                self._addresses.append((channel, address, path_list, attribute_name, rows))
        except Exception:
            self._release_backend_addresses(renderer)
            raise

    def __call__(self, material_offsets: dict[str, Any] | None = None, env_ids: Any | None = None) -> None:
        """Mark channels dirty; OVRTX currently copies each dirty channel's full device buffer."""
        del env_ids
        selected = self._buffers if material_offsets is None else material_offsets
        self._dirty_channels.update(selected)
        wp.record_event(self._event)

    def publish(self) -> None:
        """Submit dirty buffers to OVRTX after ordering their producer streams."""
        channels = self._dirty_channels
        if not channels:
            return
        renderer = self._renderer_ref()
        operations = []
        try:
            for channel, address, _path_list, attribute_name, rows in self._addresses:
                if channel not in channels:
                    continue
                if renderer._use_ovstage:
                    operation = renderer.backend.stage.write_attribute(
                        address,
                        attribute_name,
                        ordinal=renderer._current_ordinal,
                        tensors=self._buffers[channel][rows],
                        is_array=False,
                        cuda_event=self._event.cuda_event,
                    )
                else:
                    # The event orders the read after the producers (access sync). The stream
                    # receives the done fence: work later enqueued on it waits for the read, so
                    # the next refill of these zero-copy buffers cannot race it. Fills and
                    # refills run on this device's current Warp stream.
                    operation = address.write_async(
                        self._buffers[channel][rows],
                        data_access=DataAccess.ASYNC,
                        cuda_event=self._event.cuda_event,
                        cuda_stream=wp.get_stream(self._device).cuda_stream or 1,
                    )
                operations.append(operation)
        finally:
            self._operations = tuple(operations)
        channels.clear()

    def drain(self) -> None:
        """Complete submitted writes before their scene buffers may change.

        Every op is waited even when an earlier one fails. Skipping the rest would leave their
        writes pending with no remaining reference to wait on.
        """
        operations, self._operations = self._operations, ()
        errors = []
        for operation in operations:
            try:
                operation.wait()
            except Exception as e:
                errors.append(e)
        if errors:
            raise RuntimeError(f"{len(errors)} OVRTX material write(s) failed to complete") from errors[0]

    def _release_backend_addresses(self, renderer: OVRTXRenderer) -> None:
        for _channel, address, path_list, _attribute_name, _rows in self._addresses:
            if renderer._use_ovstage:
                renderer.backend.stage.release_query(address).wait()
                renderer.backend.paths.destroy_path_list(path_list)
            else:
                address.unbind()
        self._addresses.clear()

    def close(self) -> None:
        """Drain writes and release every compiled backend address."""
        try:
            self.drain()
        finally:
            renderer = self._renderer_ref()
            if renderer is not None:
                # RenderContext.close() closes writers before the renderer, so an asynchronous
                # render can still be in flight here and still read these bindings. Deliver every
                # queued render before the release below. One failed render must not leave the
                # others in flight while their bindings are released.
                for error in renderer.drain_pending_renders():
                    logger.warning("Error draining in-flight render before material release: %s", error)
                self._release_backend_addresses(renderer)
            self._dirty_channels.clear()
            self._buffers.clear()
