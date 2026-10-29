# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Geometry, contacts and batched predicates for the success terms of the VLA benchmark tasks.

Object geometry is read once from the USD prims: the convex hull of an object's meshes and the bounding box of its
geometry, both in the object frame at world scale. A container's hull without its upward faces is open to the air
above the rim. In an environment that switches tasks, an object is a slot of
:class:`~isaaclab.sim.SwitchableAssetsCfg` candidates, and each environment uses the candidate of its task.
Contacts are the normal forces between rigid objects that the scene's ``contact`` sensor reports.

A predicate is evaluated for all environments at once, each environment with the parameters of its row.
"""

from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.spatial import ConvexHull

from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

GRIPPER = "left_inner_finger"
"""Contact sensor body of the gripper link that holds objects."""

FORCE_THRESHOLD = 0.1
"""Normal force [N] above which two bodies are in contact."""

UPRIGHT_TOLERANCE = 0.1
"""Largest angle [rad] between an upright object's z axis and the world's, as in RoboLab's ``object_upright``."""

_LOGICAL = {"all": 0, "any": 1, "choose": 2}


class Scene:
    """Geometry of every candidate of some rigid objects, and their poses and contacts in the current step.

    Objects are referred to by their index in :attr:`names`; the gripper link by :attr:`gripper`.
    """

    def __init__(self, env: ManagerBasedRLEnv, names: list[str]):
        self.env = env
        self.names = names
        self.gripper = len(names)
        self._assets = [env.scene[name] for name in names]
        self._switchable = [isinstance(asset.cfg.spawn, sim_utils.SwitchableAssetsCfg) for asset in self._assets]
        geometry, offsets = [], []
        for asset, switchable in zip(self._assets, self._switchable):
            prim = sim_utils.find_first_matching_prim(asset.cfg.prim_path)
            offsets.append(len(geometry))
            geometry += [_geometry(candidate) for candidate in (prim.GetChildren() if switchable else [prim])]
        faces = max(len(planes) for *_, planes in geometry)
        # Padding planes (n = 0, d = -1) hold for every point.
        padding = np.array([0.0, 0.0, 0.0, -1.0])
        planes = [np.vstack([p, np.tile(padding, (faces - len(p), 1))]) for *_, p in geometry]
        device = env.device
        self._corners = torch.tensor(np.stack([g[0] for g in geometry]), dtype=torch.float32, device=device)
        self._box_center = torch.tensor(np.stack([g[1] for g in geometry]), dtype=torch.float32, device=device)
        self._hull_center = torch.tensor(np.stack([g[2] for g in geometry]), dtype=torch.float32, device=device)
        self._planes = torch.tensor(np.stack(planes), dtype=torch.float32, device=device)
        self._offsets = torch.tensor(offsets, device=device)
        self._sensor = env.scene["contact"]
        bodies = [*names, GRIPPER]
        self._sensor_rows = torch.tensor([self._sensor.sensor_names.index(body) for body in bodies], device=device)
        self._sensor_columns = torch.tensor(
            [self._sensor.filter_object_names.index(body) for body in bodies], device=device
        )
        self._envs = torch.arange(env.num_envs, device=device)

    def update(self) -> None:
        """Read the poses, candidates and contact forces of the objects in this step."""
        pose = torch.stack([asset.data.root_link_pose_w.torch for asset in self._assets], dim=1)  # (N, M, 7)
        self.pos, self.quat = pose[..., :3], pose[..., 3:]
        zeros = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.env.device)
        candidates = [
            self.env.task_manager.candidate_ids(name) if switchable else zeros
            for name, switchable in zip(self.names, self._switchable)
        ]
        # Rows of the geometry tables (N, M).
        self._rows = self._offsets + torch.stack(candidates, dim=1)
        self.corners = self._to_world(self._corners[self._rows])  # (N, M, 8, 3)
        self.box_center = self._to_world(self._box_center[self._rows].unsqueeze(2)).squeeze(2)  # (N, M, 3)
        self.hull_center = self._to_world(self._hull_center[self._rows].unsqueeze(2)).squeeze(2)  # (N, M, 3)
        self._forces = self._sensor.data.normal_force_matrix_w.torch

    def _to_world(self, points: torch.Tensor) -> torch.Tensor:
        """Transform object-frame points (N, M, P, 3) to the world."""
        quat = self.quat.unsqueeze(2).expand(*points.shape[:-1], 4)
        return quat_apply(quat, points) + self.pos.unsqueeze(2)

    def take(self, values: torch.Tensor, objects: torch.Tensor) -> torch.Tensor:
        """Values (N, M, ...) of the objects (N, ...) of each environment."""
        return values[self._envs.view(-1, *[1] * (objects.dim() - 1)), objects]

    def contains(self, container: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Whether points (N, K, 3) lie in the open-top convex hull of the container (N,) of each environment."""
        pos, quat = self.pos[self._envs, container], self.quat[self._envs, container]
        local = quat_apply_inverse(quat.unsqueeze(1).expand(*points.shape[:-1], 4), points - pos.unsqueeze(1))
        planes = self._planes[self._rows[self._envs, container]]  # (N, F, 4)
        return (torch.einsum("nkc,nfc->nkf", local, planes[..., :3]) + planes[:, None, :, 3]).amax(dim=-1) <= 0.0

    def force(self, body: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """Normal force on the bodies from the others (N, ...) of each environment (N, ..., 3) [N]."""
        envs = self._envs.view(-1, *[1] * (body.dim() - 1))
        return self._forces[envs, self._sensor_rows[body], self._sensor_columns[other]]

    def touching(self, body: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        return (self.force(body, other).abs() > FORCE_THRESHOLD).any(dim=-1)

    def released(self, objects: torch.Tensor) -> torch.Tensor:
        """Whether the gripper does not touch the objects (N, K)."""
        return ~self.touching(objects, torch.full_like(objects, self.gripper))

    def upright(self, objects: torch.Tensor) -> torch.Tensor:
        """Whether the objects (N, K) stand upright: their z axes within :data:`UPRIGHT_TOLERANCE` of the world's."""
        quat = self.take(self.quat, objects)
        up = quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(*quat.shape[:-1], 3))
        return up[..., 2] >= math.cos(UPRIGHT_TOLERANCE)

    def placed(self, objects: torch.Tensor, container: torch.Tensor, p: dict[str, torch.Tensor]) -> torch.Tensor:
        """Whether the objects' hull centers are in the container, with the contact and requirements of ``p``."""
        result = self.contains(container, self.take(self.hull_center, objects))
        result &= self.touching(objects, container.unsqueeze(1).expand_as(objects)) | ~p["require_contact"].unsqueeze(1)
        return result & self.requirements(objects, p)

    def requirements(self, objects: torch.Tensor, p: dict[str, torch.Tensor]) -> torch.Tensor:
        """Whether the objects (N, K) are released and upright where the parameters ``p`` require."""
        result = self.released(objects) | ~p["require_gripper_detached"].unsqueeze(1)
        return result & (self.upright(objects) | ~p["require_upright"].unsqueeze(1))


def entity_names(params: list[dict]) -> list[str]:
    """Names of the scene entities in the parameters, in order of appearance."""
    names = {}
    for row in params:
        for value in row.values():
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, SceneEntityCfg):
                    names[item.name] = None
    return list(names)


def _combine(results: torch.Tensor, mask: torch.Tensor, logical: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """Combine per-object results (N, K) of the listed objects (``mask``): ``all``, ``any``, or exactly ``count``."""
    hits = (results & mask).sum(dim=1)
    return torch.where(logical == 0, hits == mask.sum(dim=1), torch.where(logical == 1, hits > 0, hits == count))


class Predicate:
    """A predicate of every environment, with the parameters of the environment's row.

    The rows hold all parameters of the predicate's term (see :func:`with_defaults`). They are stacked: entity lists
    as padded object indices with a ``<name>_mask``, entities as object indices, ``logical`` as codes, and other
    values as they are.
    """

    def __init__(self, scene: Scene, rows: list[dict]):
        self.scene = scene
        device = scene.env.device
        self._params = {}
        for key in rows[0]:
            values = [row[key] for row in rows]
            if isinstance(values[0], list):
                width = max(map(len, values))
                indices = [[scene.names.index(v.name) for v in value] + [0] * (width - len(value)) for value in values]
                mask = [[True] * len(value) + [False] * (width - len(value)) for value in values]
                self._params[key] = torch.tensor(indices, device=device)
                self._params[f"{key}_mask"] = torch.tensor(mask, device=device)
            elif isinstance(values[0], SceneEntityCfg):
                self._params[key] = torch.tensor([scene.names.index(v.name) for v in values], device=device)
            elif key == "logical":
                self._params[key] = torch.tensor([_LOGICAL[v] for v in values], device=device)
            else:
                self._params[key] = torch.tensor(values, device=device)

    def __call__(self, rows: torch.Tensor) -> torch.Tensor:
        """Evaluate the predicate with the rows (N,) of the environments, after :meth:`Scene.update`."""
        return self.evaluate({key: value[rows] for key, value in self._params.items()})

    def evaluate(self, p: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def reset(self, env_ids: torch.Tensor | None) -> None:
        pass


class InContainer(Predicate):
    def evaluate(self, p):
        placed = self.scene.placed(p["objects"], p["container"], p)
        return _combine(placed, p["objects_mask"], p["logical"], p["count"])


class OutsideOf(Predicate):
    def evaluate(self, p):
        scene, objects = self.scene, p["objects"]
        result = ~scene.contains(p["container"], scene.take(scene.hull_center, objects)) & scene.requirements(
            objects, p
        )
        return _combine(result, p["objects_mask"], p["logical"], p["count"])


class OnTopOf(Predicate):
    def evaluate(self, p):
        scene, objects, reference = self.scene, p["objects"], p["reference"]
        corners = scene.take(scene.corners, reference)  # (N, 8, 3)
        low, high = corners.amin(dim=1) - 0.01, corners.amax(dim=1) + 0.01
        force = scene.force(objects, reference.unsqueeze(1).expand_as(objects))  # (N, K, 3)
        magnitude = force.norm(dim=-1)
        supported = (magnitude >= FORCE_THRESHOLD) & (force[..., 2] > 0.0) & (force[..., 2] >= magnitude * 0.5**0.5)
        center = scene.take(scene.box_center, objects)[..., :2]
        over = ((center >= low[:, None, :2]) & (center <= high[:, None, :2])).all(dim=-1)
        result = supported & over & scene.requirements(objects, p)
        return _combine(result, p["objects_mask"], p["logical"], p["count"])


class Stacked(Predicate):
    def evaluate(self, p):
        scene, objects, mask = self.scene, p["objects"], p["objects_mask"]
        corners = scene.take(scene.corners, objects)  # (N, K, 8, 3)
        low, high = corners.amin(dim=2), corners.amax(dim=2)  # (N, K, 3)
        bottom = low[..., 2]
        # above[e, i, j]: object i lies above object j.
        xy = scene.take(scene.box_center, objects)[:, :, None, :2]

        def over(slack: float) -> torch.Tensor:
            return ((xy >= low[:, None, :, :2] - slack) & (xy <= high[:, None, :, :2] + slack)).all(dim=-1)

        above = over(0.01) & (bottom[:, :, None] >= high[:, None, :, 2])
        above |= over(0.0) & (bottom[:, :, None] >= low[:, None, :, 2])
        pairs = (objects.unsqueeze(2).expand(-1, -1, objects.shape[1]), objects.unsqueeze(1).expand_as(above))
        supported = scene.touching(*pairs) & above
        # Unlisted objects sort below the listed ones; the pairs above them hold.
        heights = scene.take(scene.pos, objects)[..., 2].masked_fill(~mask, -torch.inf)
        order = heights.argsort(dim=1, descending=True)
        upper, lower = order[:, :-1], order[:, 1:]
        envs = torch.arange(objects.shape[0], device=objects.device).unsqueeze(1)
        return (supported[envs, upper, lower] | ~mask.gather(1, lower)).all(dim=1)


class InOrder(Predicate):
    def __init__(self, scene: Scene, rows: list[dict]):
        super().__init__(scene, rows)
        size = (scene.env.num_envs, self._params["objects"].shape[1])
        self._first = torch.full(size, -1, dtype=torch.long, device=scene.env.device)

    def evaluate(self, p):
        mask = p["objects_mask"]
        placed = self.scene.placed(p["objects"], p["container"], p)
        step = self.scene.env.episode_length_buf.unsqueeze(1).expand_as(self._first)
        self._first = torch.where((self._first < 0) & placed & mask, step, self._first)
        # Objects placed now have a first placement step.
        ordered = (self._first[:, :-1] <= self._first[:, 1:]) | ~mask[:, 1:]
        return (placed | ~mask).all(dim=1) & ordered.all(dim=1)

    def reset(self, env_ids: torch.Tensor | None) -> None:
        self._first[slice(None) if env_ids is None else env_ids] = -1


def with_defaults(term: type, params: dict) -> dict:
    """The parameters of a predicate term class, completed with the defaults of its ``__call__`` signature."""
    signature = inspect.signature(term.__call__).parameters.values()
    return {p.name: p.default for p in signature if p.default is not inspect.Parameter.empty} | params


def _geometry(prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bounding box corners and center, hull vertex mean and open-top hull planes of a prim, in its frame.

    The frame keeps the world scale. The hull is that of the default-purpose meshes, or of the box without meshes.
    """
    xforms = UsdGeom.XformCache()
    to_local = xforms.GetLocalToWorldTransform(prim).RemoveScaleShear().GetInverse()
    bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim)
    box = bound.GetRange()
    corners = np.array([to_local.Transform(bound.GetMatrix().Transform(box.GetCorner(i))) for i in range(8)])
    center = np.array(to_local.Transform(bound.ComputeCentroid()))
    points = []
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        mesh = UsdGeom.Mesh(child)
        if not mesh or mesh.GetPurposeAttr().Get() not in (None, UsdGeom.Tokens.default_):
            continue
        matrix = np.array(xforms.GetLocalToWorldTransform(child) * to_local)
        points.append(np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64) @ matrix[:3, :3] + matrix[3, :3])
    points = np.concatenate(points) if points else corners
    hull = ConvexHull(points)
    # Faces facing up within 45 degrees form the rim cap of an open container.
    planes = hull.equations[hull.equations[:, 2] < 0.7]
    return corners, center, points[hull.vertices].mean(axis=0), planes
