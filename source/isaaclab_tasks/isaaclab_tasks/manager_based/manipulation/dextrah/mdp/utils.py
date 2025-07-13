import torch
import hashlib
import os
import numpy as np
import trimesh
from pxr import UsdGeom
import isaacsim.core.utils.prims as prim_utils
from isaaclab.sim.utils import get_all_matching_child_prims


def sample_object_point_cloud(num_envs: int, num_points: int, prim_path: str, use_cache: bool = False):
    # allocate output
    points = torch.zeros((num_envs, num_points, 3))
    # on-disk cache setup
    if use_cache:
        CACHE_DIR = "/tmp/isaaclab/sample_point_cloud"
        os.makedirs(CACHE_DIR, exist_ok=True)

    for i in range(num_envs):
        # find any of the supported prim types
        object_prim_path = prim_path.replace(".*", str(i))
        prims = get_all_matching_child_prims(
            object_prim_path,
            predicate=lambda p: p.GetTypeName() in ("Mesh","Cube","Sphere","Cylinder","Capsule", "Cone")
        )
        if len(prims) == 0:
            raise Exception("Do not find any prim that can generate point cloud")

        prim, prim_type = prims[0], prims[0].GetTypeName()

        # read USD transform scale
        object_prim = prim_utils.get_prim_at_path(object_prim_path)
        usd_scale = torch.tensor(object_prim.GetAttribute("xformOp:scale").Get())
        if prim.GetPath() != object_prim.GetPath() and prim.HasAttribute("xformOp:scale"):
            usd_scale *= torch.tensor(prim.GetAttribute("xformOp:scale").Get())

        # build (or load) samples
        samples = None
        if use_cache:
            # hash type + relevant parameters instead of just vertex bytes
            hasher = hashlib.sha256()
            hasher.update(prim_type.encode())
            if prim_type == "Mesh":
                mesh = UsdGeom.Mesh(prim)
                verts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
                hasher.update(verts.tobytes())
            else:
                # encode primitive parameters
                if prim_type == "Cube":
                    size = UsdGeom.Cube(prim).GetSizeAttr().Get()
                    hasher.update(np.float32(size).tobytes())
                elif prim_type == "Sphere":
                    r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
                    hasher.update(np.float32(r).tobytes())
                elif prim_type == "Cylinder":
                    c = UsdGeom.Cylinder(prim)
                    hasher.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                    hasher.update(np.float32(c.GetHeightAttr().Get()).tobytes())
                elif prim_type == "Capsule":
                    c = UsdGeom.Capsule(prim)
                    hasher.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                    hasher.update(np.float32(c.GetHeightAttr().Get()).tobytes())
                elif prim_type == "Cone":
                    c = UsdGeom.Cone(prim)
                    hasher.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                    hasher.update(np.float32(c.GetHeightAttr().Get()).tobytes())

            geom_id = hasher.hexdigest()
            cache_file = os.path.join(CACHE_DIR, f"{geom_id}.npy")
            if os.path.exists(cache_file):
                arr = np.load(cache_file)
                if arr.shape[0] >= num_points:
                    samples = arr[:num_points]

        if samples is None:
            # build a trimesh primitive
            if prim_type == "Mesh":
                mesh = UsdGeom.Mesh(prim)
                verts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
                counts  = mesh.GetFaceVertexCountsAttr().Get()
                indices = mesh.GetFaceVertexIndicesAttr().Get()
                # fan-triangulate
                faces = []
                it = iter(indices)
                for cnt in counts:
                    poly = [next(it) for _ in range(cnt)]
                    for k in range(1, cnt-1):
                        faces.append([poly[0], poly[k], poly[k+1]])
                mesh_tm = trimesh.Trimesh(vertices=verts, faces=np.array(faces, dtype=np.int64), process=False)
            elif prim_type == "Cube":
                size = UsdGeom.Cube(prim).GetSizeAttr().Get()
                mesh_tm = trimesh.creation.box(extents=(size, size, size))
            elif prim_type == "Sphere":
                r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
                mesh_tm = trimesh.creation.icosphere(subdivisions=3, radius=r)
            elif prim_type == "Cylinder":
                c = UsdGeom.Cylinder(prim)
                mesh_tm = trimesh.creation.cylinder(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
            elif prim_type == "Capsule":
                c = UsdGeom.Capsule(prim)
                mesh_tm = trimesh.creation.capsule(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
            elif prim_type == "Cone":
                c = UsdGeom.Cone(prim)
                mesh_tm = trimesh.creation.cone(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
            # sample on surface
            samples, _ = mesh_tm.sample(num_points, return_index=True)
            if prim_type == "Cone":
                # trimesh is half height higher, has a differnt origin than usd cone
                samples[:, 2] -= c.GetHeightAttr().Get() / 2
            if use_cache:
                np.save(cache_file, samples)

        # store and apply USD scale
        points[i] = torch.from_numpy(samples) * usd_scale.unsqueeze(0)

    return points