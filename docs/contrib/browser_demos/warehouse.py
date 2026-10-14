# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pack MicroDuck's authored warehouse with instanced visual meshes and static collision proxies."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import trimesh
from bundle import write_binary_files
from PIL import Image
from scipy.spatial import cKDTree

import omni.client
from pxr import Sdf, Usd, UsdGeom, UsdShade, UsdUtils

SOURCE = (
    "omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Environments/MicroDuckWarehouse/microduck_warehouse_static.usdc"
)
SOURCE_SHA256 = "4ac3e17232179a72086a3ff7c5af4d4546694f15a4931a0d21478a4a523da419"
ROOT = "/World/envs/env_0/Factory"
BOUNDS = (-5.85, 5.85, -8.85, 3.65)


def resolve_warehouse(output: Path) -> Path:
    """Cache the pinned scene and resolve its USD references without modifying server assets."""
    output.mkdir(parents=True, exist_ok=True)
    public = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/"
    paths, pending, provenance = {}, [SOURCE], []

    def path(url: str) -> Path:
        return output / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".usdc")

    async def download(url: str) -> None:
        result, _, content = await asyncio.wait_for(omni.client.read_file_async(url), 30)
        if result != omni.client.Result.OK:
            raise RuntimeError(f"Warehouse source unavailable: {url}: {result}")
        raw = bytes(content)
        sha = hashlib.sha256(raw).hexdigest()
        if url == SOURCE and sha != SOURCE_SHA256:
            raise ValueError("Warehouse scene changed; review its layout and collision proxies before exporting")
        temporary = path(url).with_suffix(".input.usdc")
        temporary.write_bytes(raw)
        layer = Sdf.Layer.FindOrOpen(str(temporary))
        if url == SOURCE:
            edit = Sdf.BatchNamespaceEdit()
            edit.Add(Sdf.NamespaceEdit.Remove("/World/envs/env_0/Robot"))
            if not layer.Apply(edit):
                raise ValueError("Warehouse robot path changed")

        def remap(value: str) -> str:
            if not value:
                return value
            if value.startswith("../../../"):
                resolved = public + value.removeprefix("../../../")
            elif "://" in value:
                resolved = value
            elif url.startswith("omniverse://"):
                resolved = url.rsplit("/", 1)[0] + "/" + value.removeprefix("./")
            else:
                resolved = urljoin(url, value)
            if resolved.lower().endswith((".usd", ".usda", ".usdc")):
                if resolved not in paths:
                    paths[resolved] = path(resolved)
                    pending.append(resolved)
                return str(path(resolved))
            return resolved

        UsdUtils.ModifyAssetPaths(layer, remap)
        layer.Export(str(path(url)))
        provenance.append({"url": url, "sha256": sha})
        temporary.unlink()

    async def collect() -> None:
        paths[SOURCE] = path(SOURCE)
        while pending:
            batch = pending[:6]
            del pending[:6]
            await asyncio.gather(*(download(url) for url in batch))
            if len(paths) > 400:
                raise ValueError("Warehouse dependency layout changed")

    asyncio.run(collect())
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return path(SOURCE)


def _material(prim: Usd.Prim) -> tuple[dict, str | None, str]:
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    result = {"color": [0.65, 0.67, 0.69], "roughness": 0.7}
    texture, uv_name = None, "st"
    if not material:
        return result, texture, uv_name
    for shader_prim in Usd.PrimRange(material.GetPrim()):
        if shader_prim.GetTypeName() != "Shader":
            continue
        shader = UsdShade.Shader(shader_prim)
        if not any(
            shader.GetInput(name)
            for name in ("diffuseColor", "ColorAlbedo", "diffuse_color_constant", "AlbedoTexture", "diffuse_texture")
        ):
            continue
        for name in ("diffuseColor", "ColorAlbedo", "diffuse_color_constant"):
            value = shader.GetInput(name)
            if value and value.Get() is not None:
                result["color"] = list(value.Get())[:3]
        value = shader.GetInput("roughness")
        if value and value.Get() is not None:
            result["roughness"] = float(value.Get())
        value = shader.GetInput("opacity")
        if value and value.Get() is not None:
            result["opacity"] = float(value.Get())
        diffuse = shader.GetInput("diffuseColor")
        connected = diffuse.GetConnectedSource() if diffuse else None
        image_shader = UsdShade.Shader(connected[0]) if connected else shader
        for name in ("AlbedoTexture", "diffuse_texture", "file"):
            value = image_shader.GetInput(name)
            if value and value.Get() is not None and hasattr(value.Get(), "path"):
                texture = value.Get().path
                # The collected scene's base-material references use the Isaac prefix;
                # the public copy is under NVIDIA and contains the same texture bytes.
                texture = texture.replace("/6.2/Isaac/Materials/", "/6.2/NVIDIA/Materials/")
                coordinates = image_shader.GetInput("st")
                reader = coordinates.GetConnectedSource() if coordinates else None
                if reader:
                    variable = UsdShade.Shader(reader[0]).GetInput("varname")
                    if variable and variable.Get():
                        uv_name = str(variable.Get())
                break
    if texture:
        result["color"] = [1.0, 1.0, 1.0]
    return result, texture, uv_name


def _geometry(prim: Usd.Prim, uv_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    triangles, start = [], 0
    for count in mesh.GetFaceVertexCountsAttr().Get():
        triangles.extend((start, start + i, start + i + 1) for i in range(1, count - 1))
        start += count
    corners = np.asarray(triangles)
    faces = indices[corners]
    uv = UsdGeom.PrimvarsAPI(prim).GetPrimvar(uv_name)
    if uv and uv.HasValue():
        values = np.asarray(uv.ComputeFlattened(), dtype=np.float64)
        coordinates = values[corners] if uv.GetInterpolation() == "faceVarying" else values[faces]
    else:
        coordinates = np.zeros((*faces.shape, 2))
    if mesh.GetOrientationAttr().Get() == "leftHanded":
        faces = faces[:, ::-1]
        coordinates = coordinates[:, ::-1]
    original = trimesh.Trimesh(points, faces, process=False)
    valid = original.nondegenerate_faces()
    faces, coordinates = faces[valid], coordinates[valid]
    original = trimesh.Trimesh(points, faces, process=False)
    original.merge_vertices()
    geometry = original
    if len(original.faces) > 800:
        reduced = original.simplify_quadric_decimation(face_count=800)
        # These are decorative meshes; physics uses separate box proxies.
        # Preserve silhouette and volume rather than tiny scanned interior holes.
        if np.all(np.abs(reduced.bounds - original.bounds) <= np.maximum(original.extents * 0.02, 1e-6)) and (
            not original.is_watertight or np.isclose(reduced.volume, original.volume, rtol=0.02, atol=0.0)
        ):
            geometry = reduced
            # Transfer each reduced face through one source UV patch, preserving texture seams.
            source = cKDTree(points[faces].mean(axis=1)).query(geometry.triangles_center)[1]
            barycentric = trimesh.triangles.points_to_barycentric(
                np.repeat(points[faces[source]], 3, axis=0), geometry.triangles.reshape(-1, 3)
            ).reshape(-1, 3, 3)
            coordinates = np.einsum("fvi,fij->fvj", barycentric, coordinates[source])
    vertices = geometry.triangles.astype("<f4").reshape(-1, 3)
    coordinates = coordinates.astype("<f4").reshape(-1, 2)
    if not np.isfinite(vertices).all() or not np.isfinite(coordinates).all():
        raise ValueError(f"Non-finite warehouse geometry: {prim.GetPath()}")
    return vertices, np.arange(len(vertices), dtype="<u4"), coordinates


def pack_warehouse(stage_path: Path, output: Path) -> tuple[dict, list[dict]]:
    """Use the resolved Factory subset; the captured robot and RTX cameras are excluded."""
    stage = Usd.Stage.Open(str(stage_path))
    root = stage.GetPrimAtPath(ROOT)
    if not root:
        raise ValueError("Warehouse Factory root changed")
    output.mkdir(parents=True, exist_ok=True)
    cache, prototypes, textures = UsdGeom.XformCache(), {}, {}
    data, meshes = bytearray(), []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if (
            mesh.ComputeVisibility() == "invisible"
            or mesh.ComputePurpose() in ("guide", "proxy")
            or "collision" in prim.GetName().lower()
            or not mesh.GetPointsAttr().Get()
        ):
            continue
        material, texture, uv_name = _material(prim)
        # Identical shelf contents share one reduced geometry and one instanced draw.
        primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar(uv_name)
        uv_identity = np.asarray(primvar.ComputeFlattened()).tobytes() if primvar and primvar.HasValue() else b""
        identity = hashlib.sha256(
            np.asarray(mesh.GetPointsAttr().Get()).tobytes()
            + np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).tobytes()
            + np.asarray(mesh.GetFaceVertexCountsAttr().Get()).tobytes()
            + uv_identity
            + str((uv_name, material, texture, mesh.GetOrientationAttr().Get())).encode()
        ).hexdigest()
        transform = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float32)
        if identity in prototypes:
            prototypes[identity]["instances"].append(transform.reshape(-1).tolist())
            continue
        vertices, indices, uv = _geometry(prim, uv_name)
        record = {
            "body": -1,
            "name": prim.GetName(),
            "instances": [transform.reshape(-1).tolist()],
            **material,
            "flatShading": False,
            "vertexOffset": len(data),
            "vertexCount": len(vertices),
        }
        data.extend(vertices.tobytes())
        record.update(indexOffset=len(data), indexCount=len(indices), indexType="uint32")
        data.extend(indices.tobytes())
        record["uvOffset"] = len(data)
        data.extend(uv.tobytes())
        if texture:
            if texture not in textures:
                result, _, content = omni.client.read_file(texture)
                if result != omni.client.Result.OK:
                    raise RuntimeError(f"Warehouse texture unavailable: {texture}: {result}")
                image = Image.open(io.BytesIO(bytes(content)))
                image.thumbnail((512, 512))
                path = output / (hashlib.sha256(texture.encode()).hexdigest()[:16] + ".webp")
                image.save(path, format="WEBP", quality=85)
                textures[texture] = path.name
            record["texture"] = textures[texture]
        meshes.append(record)
        prototypes[identity] = record
    files = write_binary_files(gzip.compress(bytes(data), mtime=0), output, "warehouse.bin.gz")
    for stale in output.glob("warehouse.bin*"):
        if stale.name not in files:
            stale.unlink()
    description = {"file": "warehouse.bin.gz", "files": files, "byteLength": len(data), "meshes": meshes}
    (output / "warehouse.json").write_text(json.dumps(description, indent=2) + "\n")
    colliders = warehouse_colliders(stage)
    (output / "colliders.json").write_text(json.dumps(colliders, indent=2) + "\n")
    print(
        f"Warehouse: {len(meshes)} prototypes, {sum(len(m['instances']) for m in meshes)} instances, "
        f"{len(colliders)} static boxes",
        flush=True,
    )
    return description, colliders


def warehouse_colliders(stage: Usd.Stage) -> list[dict]:
    """Bound the walkable floor and block shelf/clutter footprints with static boxes."""
    bbox = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)
    x0, x1, y0, y1 = BOUNDS
    height, width = 1.0, 0.3
    colliders = [
        {"position": [x0 - width / 2, (y0 + y1) / 2, height / 2], "size": [width, y1 - y0 + 2 * width, height]},
        {"position": [x1 + width / 2, (y0 + y1) / 2, height / 2], "size": [width, y1 - y0 + 2 * width, height]},
        {"position": [(x0 + x1) / 2, y0 - width / 2, height / 2], "size": [x1 - x0, width, height]},
        {"position": [(x0 + x1) / 2, y1 + width / 2, height / 2], "size": [x1 - x0, width, height]},
    ]
    obstacles = list(stage.GetPrimAtPath(ROOT + "/Shelves").GetChildren())
    obstacles += list(stage.GetPrimAtPath(ROOT + "/Clutter").GetChildren())
    obstacles.append(stage.GetPrimAtPath(ROOT + "/SpillBox"))
    for prim in obstacles:
        if not prim:
            continue
        bounds = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        low, high = np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())
        if high[2] < 0.08:
            continue
        extent = high[:2] - low[:2]
        if np.any(extent < 0.01):
            continue
        if np.all(low[:2] < 0.15) and np.all(high[:2] > -0.15):
            raise ValueError(f"Warehouse proxy intersects MicroDuck reset: {prim.GetPath()}")
        height = min(high[2], 1.0)
        colliders.append({"position": [*((low[:2] + high[:2]) / 2), height / 2], "size": [*extent, height]})
    return colliders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage = resolve_warehouse(args.output / "source")
    pack_warehouse(stage, args.output)
    print(f"Resolved warehouse: {stage}", flush=True)


if __name__ == "__main__":
    main()
