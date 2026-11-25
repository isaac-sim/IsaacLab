# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collision-free tabletop resets and the task bindings afforded by sampled objects."""

from __future__ import annotations

import itertools
import math
import re
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
import warp as wp
from scipy.spatial import ConvexHull, HalfspaceIntersection

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg

    from ..heterogeneous_tasks_env_cfg import TableCfg


class Task(NamedTuple):
    """One available instruction of a world.

    ``object`` and ``target`` name the first pair of bodies that the instruction can refer to. Every object of the
    object's name counts for it, and every target of the target's name.
    """

    family: str
    object: str
    target: str
    instruction: str


def _collision_geometry(spawn: sim_utils.SpawnerCfg) -> tuple[np.ndarray, np.ndarray]:
    """Object-frame bounds and convex XY footprint of collision geometry, including USD transforms and scale."""
    if isinstance(spawn, sim_utils.CuboidCfg):
        half = np.asarray(spawn.size) / 2
        bounds = np.array([-half, half])
        return bounds, bounds[np.array([(0, 0), (1, 0), (1, 1), (0, 1)]), np.arange(2)]
    if isinstance(spawn, sim_utils.SphereCfg):
        angles = np.arange(32) * (2 * math.pi / 32)
        footprint = np.column_stack((np.cos(angles), np.sin(angles))) * spawn.radius / math.cos(math.pi / 32)
        return np.array([[-spawn.radius] * 3, [spawn.radius] * 3]), footprint
    if not isinstance(spawn, sim_utils.UsdFileCfg):
        raise ValueError(f"Tabletop placement needs USD meshes, cuboids or spheres, got {type(spawn).__name__}.")
    stage = Usd.Stage.Open(retrieve_file_path(spawn.usd_path))
    if stage is None:
        raise ValueError(f"Cannot read collision geometry: {spawn.usd_path}")
    root = stage.GetDefaultPrim()
    xforms = UsdGeom.XformCache()
    to_local = xforms.GetLocalToWorldTransform(root).RemoveScaleShear().GetInverse()
    points = []
    boxes = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "proxy", "render", "guide"], False, True)
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        matrix = np.array(xforms.GetLocalToWorldTransform(prim) * to_local)
        if prim.IsA(UsdGeom.Mesh):
            vertices = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
        else:
            bound = boxes.ComputeUntransformedBound(prim).ComputeAlignedRange()
            if bound.IsEmpty():
                raise ValueError(f"Empty collision bounds on {prim.GetPath()} in {spawn.usd_path}")
            vertices = np.array([bound.GetCorner(i) for i in range(8)])
        points.append(vertices @ matrix[:3, :3] + matrix[3, :3])
    if not points:
        raise ValueError(f"No collision meshes in tabletop asset: {spawn.usd_path}")
    vertices = np.concatenate(points) * np.asarray(spawn.scale or (1.0, 1.0, 1.0))
    footprint = vertices[ConvexHull(vertices[:, :2]).vertices, :2]
    if len(footprint) > 32:
        # Circumscribe detailed meshes with a bounded-size polygon for batched reset sampling.
        angles = np.arange(32) * (2 * math.pi / 32)
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        supports = (normals @ footprint.T).max(1)
        footprint = HalfspaceIntersection(np.column_stack((normals, -supports)), footprint.mean(0)).intersections
        footprint = footprint[ConvexHull(footprint).vertices]
    return np.array([vertices.min(axis=0), vertices.max(axis=0)]), footprint


def spawn_table(prim_path: str, cfg: sim_utils.UsdFileCfg, translation=None, orientation=None):
    """Keep a table's full geometry at its authored assembly pose as one kinematic body.

    SimReady packing, lift and bedside tables include articulated equipment or drawers. Freeze those parts
    before variant import so the selected table has one body, while preserving every mesh and its transform.
    Triangle collisions preserve open space beneath shelves and above the worktop.
    """
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation)
    for part in list(Usd.PrimRange(prim, Usd.TraverseInstanceProxies())):
        if part.IsA(UsdPhysics.Joint):
            part.SetActive(False)
            continue
        for schema in (
            "PhysicsRigidBodyAPI",
            "PhysxRigidBodyAPI",
            "PhysicsArticulationRootAPI",
            "PhysxArticulationAPI",
        ):
            if schema in part.GetAppliedSchemas():
                part.RemoveAppliedSchema(schema)
        if part.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(part).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(part).CreateApproximationAttr("none")
    body = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.CreateRigidBodyEnabledAttr(True)
    body.CreateKinematicEnabledAttr(True)
    return prim


_ASSET_CODE = re.compile(r"[A-Za-z]{1,2}\d+|\d+(in)?")
"""A word of a catalog label that identifies the asset rather than names it: ``A01``, ``Ph1``, ``03``, ``16in``."""


def _plain_name(label: str) -> str:
    """Return a catalog label in plain lower-case words: ``"Socket_1_2in_B01"`` reads ``"socket"``."""
    words = [word for word in label.replace("_", " ").split() if not _ASSET_CODE.fullmatch(word)]
    return " ".join(words).lower() or label


class reset_tabletop(ManagerTermBase):
    """Place sampled objects on the reachable part of a table, then select an available task.

    Every slot supports the complete catalog and a non-colliding placeholder. Resets sample cardinality,
    categories and concrete assets before selecting native mesh variants and placing their COMs. Convex
    footprints reject overlaps; Newton's collision pipeline screens actual geometry without stepping.
    Task IDs are stable indices into ``env.cfg.tasks``, shared by all environments.

    ``env.tabletop.available_tasks[world]`` lists the instructions that category- and size-compatible pairs of
    the world's objects afford. Instructions name objects in plain words, by the ``name`` semantic tag of the asset
    or else its catalog label without asset codes. A world that holds several objects of one name offers one task
    for them: any of those objects and any of those targets counts (``object_mask`` and ``target_mask``), while
    ``object_ids`` and ``target_ids`` give the first such pair. With ``graspable_size``, the bound object
    also has to suit the gripper: (smallest extent, narrowest horizontal extent) limits [m], below which the
    gripper cannot pinch the object and above which it does not fit between the open fingers. A reset chooses one unless
    :meth:`set_task` requests a task family. Reset reserves a compatible pair for that family; geometrically
    infeasible compositions are retried at the same cardinality.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.objects = env.scene[cfg.params["objects"].name]
        self.names = list(self.objects.body_names)
        self.table = env.scene[cfg.params["table"].name]
        self.gripper_body = cfg.params["gripper_body"]
        self.families = env.cfg.tasks
        for name, rule in self.families.items():
            if set(rule) != {"target", "relation", "instruction"} or rule["relation"] not in ("on", "in"):
                raise ValueError(f"Task family {name!r} requires a target, relation ('on' or 'in') and instruction.")
        self.table_variants = cfg.params["table_variants"]
        if len(self.table_variants) != self.table.num_mesh_variants:
            raise ValueError("Declare one measured surface and pose for each table mesh variant, in spawner order.")
        # Keep the measured rings as edges. Footprint containment and boundary clearance are checked
        # directly on the device, including openings and concave outlines; no polygon Boolean is needed.
        surfaces = []
        for table in self.table_variants:
            rings = [np.asarray(ring, dtype=np.float32) for ring in (table.surface_outline, *table.surface_holes)]
            for ring in rings:
                if ring.ndim != 2 or ring.shape[1] != 2 or len(ring) < 3 or not np.isfinite(ring).all():
                    raise ValueError("Tabletop requires finite XY support rings with at least three vertices.")
                edges = np.roll(ring, -1, axis=0) - ring
                area = abs(np.sum(ring[:, 0] * edges[:, 1] - ring[:, 1] * edges[:, 0])) / 2
                if area <= 1e-9 or (np.linalg.norm(edges, axis=1) <= 1e-9).any():
                    raise ValueError("Tabletop requires non-degenerate support rings.")
            surfaces.append(np.concatenate([np.stack((ring, np.roll(ring, -1, axis=0)), axis=1) for ring in rings]))
        width = max(len(edges) for edges in surfaces)
        self._table_edges = torch.zeros(len(surfaces), width, 2, 2, device=env.device)
        for index, edges in enumerate(surfaces):
            self._table_edges[index] = torch.as_tensor(edges[-1, -1], device=env.device)
            self._table_edges[index, : len(edges)] = torch.as_tensor(edges, device=env.device)
        self._placement_edges = torch.empty(env.num_envs, width, 2, 2, device=env.device)
        self._placement_bounds = torch.empty(env.num_envs, 2, 2, device=env.device)
        if any(not math.isfinite(table.surface_height) for table in self.table_variants):
            raise ValueError("Tabletop requires finite local surface heights.")
        self.tabletop_heights = torch.empty(env.num_envs, device=env.device)
        candidates = [member.spawn.assets_cfg for member in self.objects.cfg.rigid_objects.values()]
        labels = [dict(spawn.semantic_tags or ())["_catalog_key"] for spawn in candidates[0]]
        if any([dict(spawn.semantic_tags or ())["_catalog_key"] for spawn in row] != labels for row in candidates):
            raise ValueError("Every object slot must declare the same complete catalog and placeholder, in order.")
        self._labels = tuple(labels)
        self._empty = labels.index("__empty__")
        self._spoken = tuple(
            dict(spawn.semantic_tags or ()).get("name") or _plain_name(label)
            for label, spawn in zip(labels, candidates[0], strict=True)
        )
        self._categories = tuple(
            {value for key, value in spawn.semantic_tags or () if key == "category"} for spawn in candidates[0]
        )
        categories = sorted(set().union(*self._categories) - {"__empty__"})
        self._category_assets = tuple(
            [i for i, tags in enumerate(self._categories) if category in tags and i != self._empty]
            for category in categories
        )
        geometry = [_collision_geometry(spawn) for spawn in candidates[0]]
        bounds = np.asarray([bound for bound, _ in geometry], dtype=np.float32)
        vertices = max(len(polygon) for _, polygon in geometry)
        footprints = np.zeros((len(geometry), vertices, 3), dtype=np.float32)
        for index, (_, polygon) in enumerate(geometry):
            footprints[index, :, :2] = polygon[-1]
            footprints[index, : len(polygon), :2] = polygon
        self._catalog_bounds = torch.as_tensor(bounds, device=env.device)
        self._catalog_footprints = torch.as_tensor(footprints, device=env.device)
        self._corners = np.array(list(itertools.product((0, 1), repeat=3)))
        extent = bounds[:, 1] - bounds[:, 0]
        self._sizes = np.sort(extent[:, :2], axis=1)
        self._graspable = np.arange(len(labels)) != self._empty
        limits = cfg.params.get("graspable_size")
        if limits is not None:
            self._graspable &= (extent.min(1) >= limits[0]) & (extent[:, :2].min(1) <= limits[1])
        self._sources, self._targets = [], []
        for family, rule in self.families.items():
            targets = np.array([i for i, tags in enumerate(self._categories) if rule["target"] in tags], dtype=int)
            viable = np.zeros(len(labels), dtype=bool)
            for target in targets:
                viable |= (self._sizes + 2 * cfg.params["clearance"] <= self._sizes[target] * 0.8).all(1)
            viable &= self._graspable
            groups = tuple([i for i in group if viable[i]] for group in self._category_assets if viable[group].any())
            if not groups:
                raise ValueError(f"Task {family!r} has no category- and size-compatible catalog pair.")
            self._sources.append(groups)
            self._targets.append(targets)
        shape = (env.num_envs, len(self.names))
        self.object_names = [tuple("" for _ in self.names) for _ in range(env.num_envs)]
        self.active = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.local_bounds = torch.zeros((*shape, 2, 3), device=env.device)
        self.local_corners = torch.zeros((*shape, 8, 3), device=env.device)
        self.local_footprints = torch.zeros((*shape, vertices, 3), device=env.device)
        self.available_tasks = [() for _ in range(env.num_envs)]
        self._task_slots = [[] for _ in range(env.num_envs)]
        self.instructions = [""] * env.num_envs
        self.task_ids = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.object_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.target_ids = torch.zeros_like(self.object_ids)
        self.object_mask = torch.zeros_like(self.active)
        self.target_mask = torch.zeros_like(self.active)
        self.in_container = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._requested = [-1] * env.num_envs
        self._backend = env.sim.backend_manager.backend
        self._contacts = None
        object_bodies = []
        for label, world in zip(self._backend.model.body_label, self._backend.model.body_world.numpy(), strict=True):
            slot = next((i for i, name in enumerate(self.names) if f"/{name}/" in label + "/"), None)
            object_bodies.append(slot is not None and world >= 0)
        self._object_bodies = torch.tensor(object_bodies, device=env.device)
        env.tabletop = self

    def set_task(self, env_ids: list[int], task_ids: list[int]) -> None:
        """Request a task-family index from ``env.cfg.tasks`` on any world; -1 samples a family each reset."""
        if len(env_ids) != len(task_ids):
            raise ValueError("Each world requires one task index.")
        for world, task in zip(env_ids, task_ids, strict=True):
            if not 0 <= world < len(self.available_tasks) or not -1 <= task < len(self.families):
                raise ValueError(f"Unavailable task {task} for world {world}.")
        for world, task in zip(env_ids, task_ids, strict=True):
            self._requested[world] = task

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | slice,
        objects: SceneEntityCfg,
        table: SceneEntityCfg,
        table_variants: tuple[TableCfg, ...],
        workspace: tuple[tuple[float, float, float], tuple[float, float, float]],
        gripper_body: str,
        clearance: float = 0.005,
        max_attempts: int = 128,
        graspable_size: tuple[float, float] | None = None,
        object_count: tuple[int, int] = (2, 8),
    ) -> None:
        ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        if not len(ids):
            return
        if clearance <= 0 or max_attempts < 1:
            raise ValueError("Placement requires positive clearance and max_attempts.")
        workspace = torch.as_tensor(workspace, dtype=torch.float32, device=env.device)
        if workspace.shape != (2, 3) or not torch.isfinite(workspace).all() or (workspace[0] >= workspace[1]).any():
            raise ValueError("Workspace must contain finite, ordered XYZ lower and upper bounds.")
        if not 2 <= object_count[0] <= object_count[1] <= len(self.names):
            raise ValueError("Object count must be an inclusive range between two and the number of slots.")
        counts = torch.randint(object_count[0], object_count[1] + 1, (len(ids),), device=env.device)
        # Table and object placement are sampled together: a small table must not invalidate a world
        # that fits on another variant. All candidates still use their measured surface/workspace intersection.
        table_edges, region_bounds, heights = [], [], []
        for variant, edges in zip(self.table_variants, self._table_edges, strict=True):
            x, y, z, w = variant.init_state.rot
            cosine, sine = 1 - 2 * (y * y + z * z), 2 * (w * z + x * y)
            if abs(x) > 1e-6 or abs(y) > 1e-6:
                raise ValueError("Tabletop reset requires a horizontal table; yaw rotation is supported.")
            tx, ty, tz = variant.init_state.pos
            rotation = workspace.new_tensor([[cosine, -sine], [sine, cosine]])
            edges = edges @ rotation.T + workspace.new_tensor([tx, ty])
            lower = torch.maximum(edges.amin((0, 1)), workspace[0, :2]) + clearance
            upper = torch.minimum(edges.amax((0, 1)), workspace[1, :2]) - clearance
            height = tz + variant.surface_height
            if height < workspace[0, 2] - 1e-5 or height + clearance >= workspace[1, 2]:
                upper = lower.clone()
            table_edges.append(edges)
            region_bounds.append(torch.stack((lower, upper)))
            heights.append(height)
        table_edges, region_bounds = torch.stack(table_edges), torch.stack(region_bounds)
        # This is only an upper bound for an inexpensive area rejection. Every proposed footprint
        # must also pass the exact outline/hole boundary test and the XYZ workspace check below.
        region_areas = (region_bounds[:, 1] - region_bounds[:, 0]).clamp_min(0).prod(1)
        previous = self.table.mesh_variant_ids.torch[ids].long()
        variant_poses = workspace.new_tensor(
            [(*variant.init_state.pos, *variant.init_state.rot) for variant in self.table_variants]
        )
        remaining_rows = torch.arange(len(ids), device=env.device)
        for _ in range(max_attempts):
            remaining = ids[remaining_rows]
            self._select_objects(remaining, counts[remaining_rows], clearance)
            footprints = self.local_footprints[remaining, ..., :2]
            following = footprints.roll(-1, dims=2)
            areas = (footprints[..., 0] * following[..., 1] - footprints[..., 1] * following[..., 0]).sum(2).abs() / 2
            occupied = (areas * self.active[remaining]).sum(1)
            eligible = occupied[:, None] < region_areas[None]
            if len(self.table_variants) > 1:
                eligible[torch.arange(len(remaining), device=env.device), previous[remaining_rows]] = False
            possible = eligible.any(1)
            if not possible.any():
                if not (region_areas > 0).any():
                    raise RuntimeError("No usable table/workspace intersection.")
                continue
            # Retry infeasible worlds independently; one dense composition must not stall the whole batch.
            pending_rows = remaining_rows[~possible]
            remaining_rows = remaining_rows[possible]
            remaining = ids[remaining_rows]
            eligible = eligible[possible].float()
            selected = torch.multinomial(eligible, 1).flatten()
            if len(self.table_variants) > 1:
                self.table.write_mesh_variant_to_sim(selected, env_ids=remaining)
            table_poses = variant_poses[selected].clone()
            table_poses[:, :3] += env.scene.env_origins[remaining]
            self.table.write_root_pose_to_sim_index(root_pose=table_poses, env_ids=remaining)
            self.table.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(len(remaining), 6, device=env.device), env_ids=remaining
            )
            self.tabletop_heights[remaining] = (
                workspace.new_tensor(heights)[selected] + env.scene.env_origins[remaining, 2]
            )
            origins = env.scene.env_origins[remaining, :2]
            self._placement_edges[remaining] = table_edges[selected] + origins[:, None, None]
            self._placement_bounds[remaining] = region_bounds[selected] + origins[:, None]
            poses, placed = self._sample(remaining, workspace, clearance, min(max_attempts, 32))
            # Publish every pending world's new geometry and poses together. Unplaced slots stay apart
            # outside the workspace rather than inheriting overlapping poses from a previous composition.
            self.objects.write_body_com_pose_to_sim_index(body_poses=poses.contiguous(), env_ids=remaining)
            self.objects.write_body_velocity_to_sim_index(
                body_velocities=torch.zeros(len(remaining), len(self.names), 6, device=env.device), env_ids=remaining
            )
            if placed.any():
                bad = self.penetrating_worlds()
                placed &= ~bad[remaining]
            remaining_rows = torch.cat((pending_rows, remaining_rows[~placed]))
            if not len(remaining_rows):
                break
        else:
            raise RuntimeError(
                f"No collision-free table/workspace intersection placement for worlds {ids[remaining_rows].tolist()}."
            )
        self._bind_tasks(ids, clearance)

    def _select_objects(self, ids: torch.Tensor, counts: torch.Tensor, clearance: float) -> None:
        """Sample categories before assets, reserve a compatible task pair, then permute occupied and empty slots."""
        width = len(self.names)
        selected = np.full((len(ids), width), self._empty, dtype=np.int32)
        for row, (world, count) in enumerate(zip(ids.tolist(), counts.tolist(), strict=True)):
            family = self._requested[world]
            if family < 0:
                family = np.random.randint(len(self.families))
            groups = self._sources[family]
            group = groups[np.random.randint(len(groups))]
            source = group[np.random.randint(len(group))]
            targets = self._targets[family]
            targets = targets[(self._sizes[source] + 2 * clearance <= self._sizes[targets] * 0.8).all(1)]
            selected[row, :2] = (source, targets[np.random.randint(len(targets))])
            for slot in range(2, count):
                category = self._category_assets[np.random.randint(len(self._category_assets))]
                selected[row, slot] = category[np.random.randint(len(category))]
            np.random.shuffle(selected[row])
        selected = torch.as_tensor(selected, device=self._env.device)
        self.objects.write_mesh_variant_to_sim(selected, env_ids=ids)
        self.active[ids] = selected != self._empty
        self.local_bounds[ids] = self._catalog_bounds[selected]
        self.local_corners[ids] = self.local_bounds[ids][:, :, self._corners, np.arange(3)]
        self.local_footprints[ids] = self._catalog_footprints[selected]
        for world, row in zip(ids.tolist(), selected.tolist(), strict=True):
            self.object_names[world] = tuple(self._labels[index] for index in row)

    def _bind_tasks(self, ids: torch.Tensor, clearance: float) -> None:
        """Rebuild bindings from current slot selections, using global task-family IDs on every world."""
        selected = self.objects.mesh_variant_ids.torch[ids].cpu().tolist()
        for world, variants in zip(ids.tolist(), selected, strict=True):
            pairs = {}
            for family, rule in self.families.items():
                for source, target in itertools.permutations(range(len(self.names)), 2):
                    source_variant, target_variant = variants[source], variants[target]
                    if (
                        self._graspable[source_variant]
                        and rule["target"] in self._categories[target_variant]
                        and (self._sizes[source_variant] + 2 * clearance <= self._sizes[target_variant] * 0.8).all()
                    ):
                        pairs.setdefault(
                            (family, self._spoken[variants[source]], self._spoken[variants[target]]), (source, target)
                        )
            tasks, slots = [], []
            for (family, source_name, target_name), (source, target) in pairs.items():
                instruction = self.families[family]["instruction"].format(object=source_name, target=target_name)
                tasks.append(Task(family, self.names[source], self.names[target], instruction))
                slots.append(
                    (
                        [
                            i
                            for i, variant in enumerate(variants)
                            if variant != self._empty and self._spoken[variant] == source_name
                        ],
                        [
                            i
                            for i, variant in enumerate(variants)
                            if variant != self._empty
                            and self._spoken[variant] == target_name
                            and self.families[family]["target"] in self._categories[variant]
                        ],
                    )
                )
            self.available_tasks[world], self._task_slots[world] = tuple(tasks), slots
            requested = self._requested[world]
            choices = [
                i for i, task in enumerate(tasks) if requested < 0 or task.family == tuple(self.families)[requested]
            ]
            if not choices:
                raise RuntimeError(f"No evaluable requested task after reset of world {world}.")
            task = choices[int(torch.randint(len(choices), (), device=self._env.device))]
            binding = tasks[task]
            self.task_ids[world] = tuple(self.families).index(binding.family)
            self.object_mask[world] = self.target_mask[world] = False
            sources, targets = slots[task]
            self.object_mask[world, sources] = self.target_mask[world, targets] = True
            self.object_ids[world], self.target_ids[world] = (
                self.names.index(binding.object),
                self.names.index(binding.target),
            )
            self.in_container[world] = self.families[binding.family]["relation"] == "in"
            self.instructions[world] = binding.instruction

    @staticmethod
    def _supported(points: torch.Tensor, edges: torch.Tensor, clearance: float) -> torch.Tensor:
        """Keep a convex footprint within a measured outline, outside holes, with boundary clearance.

        Ray crossings test footprint vertices against all rings. Edge intersections and minimum segment
        distances catch concave cutouts between vertices; a ring vertex inside the footprint catches a
        completely enclosed hole. Repeated vertices pad the batched polygons and are ignored as edges.
        """
        a, b = points[:, :, None], points.roll(-1, 1)[:, :, None]
        c, d = edges[:, None, :, 0], edges[:, None, :, 1]
        ab, cd = b - a, d - c
        ac, ad = c - a, d - a
        valid_edges = cd.square().sum(-1) > 1e-18
        dy = cd[..., 1]
        fraction = (a[..., 1] - c[..., 1]) / torch.where(dy.abs() > 1e-12, dy, 1.0)
        crosses = ((c[..., 1] > a[..., 1]) != (d[..., 1] > a[..., 1])) & (a[..., 0] < c[..., 0] + fraction * cd[..., 0])
        inside = (crosses.sum(2) % 2 == 1).all(1)
        side_c = ab[..., 0] * ac[..., 1] - ab[..., 1] * ac[..., 0]
        side_d = ab[..., 0] * ad[..., 1] - ab[..., 1] * ad[..., 0]
        side_a = cd[..., 0] * (-ac[..., 1]) - cd[..., 1] * (-ac[..., 0])
        side_b = cd[..., 0] * (b - c)[..., 1] - cd[..., 1] * (b - c)[..., 0]
        intersects = (side_c * side_d <= 0) & (side_a * side_b <= 0)
        intersects &= valid_edges & (ab.square().sum(-1) > 1e-18)
        enclosed = ((side_c >= 0).all(1) | (side_c <= 0).all(1)) & valid_edges[:, 0]
        # The closest points of disjoint line segments include at least one segment endpoint.
        directions = torch.stack((ab.expand_as(ac), ab.expand_as(ac), cd.expand_as(ac), cd.expand_as(ac)), -2)
        offsets = torch.stack((ac, ad, -ac, b - c), -2)
        projection = ((offsets * directions).sum(-1) / directions.square().sum(-1).clamp_min(1e-18)).clamp(0, 1)
        distance = (offsets - projection[..., None] * directions).square().sum(-1).amin(-1)
        near_boundary = ((distance < clearance * clearance) | intersects) & valid_edges
        return inside & ~near_boundary.any(dim=(1, 2)) & ~enclosed.any(1)

    def _sample(
        self, ids: torch.Tensor, workspace: torch.Tensor, clearance: float, max_attempts: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample COM positions and link orientations using geometry relative to each body's COM."""
        env = self._env
        count, width = len(ids), len(self.names)
        # The simulator owns the COM offset; geometry remains stored in the asset's link frame.
        com = self.objects.data.body_com_pos_b.torch[ids]
        local_bounds = self.local_bounds[ids] - com[:, :, None]
        local_footprints = self.local_footprints[ids] - com[:, :, None]
        origin = env.scene.env_origins[ids]
        lo, hi = self._placement_bounds[ids].unbind(1)
        top = self.tabletop_heights[ids]
        poses = torch.zeros(count, width, 7, device=env.device)
        poses[..., 6] = 1.0
        poses[..., :3] = origin[:, None] + poses.new_tensor([0.0, -10.0, 0.5])
        spacing = local_bounds.abs().amax(dim=(1, 2, 3)) * 2 + clearance
        poses[..., 0] += torch.arange(width, device=env.device) * spacing[:, None]
        low, high = torch.zeros_like(poses[..., :3]), torch.zeros_like(poses[..., :3])
        footprints = torch.zeros(count, width, self.local_footprints.shape[2], 3, device=env.device)
        placed = torch.zeros(count, width, dtype=torch.bool, device=env.device)
        sizes = local_bounds[:, :, 1] - local_bounds[:, :, 0]
        order = (sizes[..., 0] * sizes[..., 1]).masked_fill(~self.active[ids], -1).argsort(dim=1, descending=True)
        rows = torch.arange(count, device=env.device)
        failed = torch.zeros(count, dtype=torch.bool, device=env.device)
        for rank in range(width):
            slots = order[:, rank]
            pending = self.active[ids, slots] & ~failed
            for _ in range(max_attempts):
                if not pending.any():
                    break
                yaw = torch.rand(count, device=env.device) * (2 * math.pi)
                quat = torch.zeros(count, 4, device=env.device)
                quat[:, 2], quat[:, 3] = (yaw / 2).sin(), (yaw / 2).cos()
                rotated = quat_apply(
                    quat[:, None].expand(count, local_footprints.shape[2], 4), local_footprints[rows, slots]
                )
                extent_lo, extent_hi = rotated.amin(1), rotated.amax(1)
                extent_lo[:, 2], extent_hi[:, 2] = (
                    local_bounds[rows, slots, 0, 2],
                    local_bounds[rows, slots, 1, 2],
                )
                lower, upper = lo - extent_lo[:, :2] + 1e-5, hi - extent_hi[:, :2] - 1e-5
                pos = torch.zeros(count, 3, device=env.device)
                # Boundary proposals leave contiguous free space; interior samples retain pose variety.
                fraction = torch.rand(count, 2, device=env.device)
                fraction = torch.where(torch.rand_like(fraction) < 0.5, fraction.round(), fraction)
                pos[:, :2] = lower + fraction * (upper - lower).clamp_min(0)
                pos[:, 2] = top - extent_lo[:, 2] + clearance
                world_points = rotated + pos[:, None]
                valid = (upper >= lower).all(1)
                valid &= self._supported(world_points[..., :2], self._placement_edges[ids], clearance)
                candidate_low, candidate_high = extent_lo + pos, extent_hi + pos
                valid &= (candidate_low >= workspace[0] + origin).all(1) & (
                    candidate_high <= workspace[1] + origin
                ).all(1)
                overlap = (
                    (candidate_low[:, None, :2] < high[..., :2] + clearance)
                    & (candidate_high[:, None, :2] > low[..., :2] - clearance)
                ).all(2)
                # AABB overlap only identifies pairs needing the convex separating-axis test.
                pairs = (overlap & placed & pending[:, None] & valid[:, None]).nonzero()
                if len(pairs):
                    left = world_points[pairs[:, 0], :, :2]
                    right = footprints[pairs[:, 0], pairs[:, 1], :, :2]
                    edges = torch.cat((left.roll(-1, 1) - left, right.roll(-1, 1) - right), dim=1)
                    axes = torch.stack((-edges[..., 1], edges[..., 0]), dim=-1)
                    lengths = axes.norm(dim=-1)
                    axes /= lengths.clamp_min(1e-9)[..., None]
                    left_projection = torch.bmm(axes, left.transpose(1, 2))
                    right_projection = torch.bmm(axes, right.transpose(1, 2))
                    separated = (left_projection.amax(2) + clearance <= right_projection.amin(2)) | (
                        right_projection.amax(2) + clearance <= left_projection.amin(2)
                    )
                    intersects = ~(separated & (lengths > 1e-9)).any(1)
                    valid[pairs[intersects, 0]] = False
                accepted = valid & pending
                chosen_rows, chosen_slots = rows[accepted], slots[accepted]
                poses[chosen_rows, chosen_slots, :3], poses[chosen_rows, chosen_slots, 3:] = (
                    pos[accepted],
                    quat[accepted],
                )
                low[chosen_rows, chosen_slots], high[chosen_rows, chosen_slots] = (
                    candidate_low[accepted],
                    candidate_high[accepted],
                )
                footprints[chosen_rows, chosen_slots] = world_points[accepted]
                placed[chosen_rows, chosen_slots] = True
                pending &= ~accepted
            failed |= pending
        return poses, ~failed

    def penetrating_worlds(self, tolerance: float = 1e-4) -> torch.Tensor:
        """Screen actual Newton contacts involving task objects, without stepping or using sensor history."""
        # Root writes author joint coordinates; publish FK before reading native contact geometry.
        self._env.sim.forward()
        backend = self._backend
        if self._contacts is None:
            # Manager terms bind before the solver constructs its collision pipeline.
            if backend.collision_pipeline is None:
                raise ValueError("Tabletop reset requires Newton's external collision pipeline.")
            self._contacts = backend.collision_pipeline.contacts()
        contacts = self._contacts
        backend.collision_pipeline.collide(backend.state_0, contacts)
        counters = wp.to_torch(contacts.contact_counters)
        count = int(counters[0])
        if count > contacts.rigid_contact_max or counters[2]:
            raise RuntimeError("Tabletop reset collision buffers overflowed; increase the collision pipeline budgets.")
        bad = torch.zeros(self._env.num_envs, dtype=torch.bool, device=self._env.device)
        if not count:
            return bad
        shape_body = wp.to_torch(backend.model.shape_body)
        b0 = shape_body[wp.to_torch(contacts.rigid_contact_shape0)[:count].long()].long()
        b1 = shape_body[wp.to_torch(contacts.rigid_contact_shape1)[:count].long()].long()
        object0, object1 = (
            self._object_bodies[b0.clamp_min(0)] & (b0 >= 0),
            self._object_bodies[b1.clamp_min(0)] & (b1 >= 0),
        )
        poses = wp.to_torch(backend.state_0.body_q)
        points = []
        for bodies, local in ((b0, contacts.rigid_contact_point0), (b1, contacts.rigid_contact_point1)):
            pose = poses[bodies.clamp_min(0)]
            local = wp.to_torch(local)[:count]
            points.append(torch.where((bodies >= 0)[:, None], quat_apply(pose[:, 3:], local) + pose[:, :3], local))
        gap = (wp.to_torch(contacts.rigid_contact_normal)[:count] * (points[1] - points[0])).sum(1)
        gap -= wp.to_torch(contacts.rigid_contact_margin0)[:count] + wp.to_torch(contacts.rigid_contact_margin1)[:count]
        body_world = wp.to_torch(backend.model.body_world)
        worlds = body_world[torch.where(object0, b0, b1).clamp_min(0)].long()
        invalid = (object0 | object1) & (gap < -tolerance) & (worlds >= 0)
        bad[worlds[invalid]] = True
        return bad
