# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collision-free tabletop resets and the task bindings afforded by sampled objects."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import shapely
import torch
import warp as wp
from scipy.spatial import ConvexHull, HalfspaceIntersection
from shapely.affinity import affine_transform

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg


class Task(NamedTuple):
    """One available instruction bound to two distinct object instances in a world."""

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
    stage = Usd.Stage.Open(spawn.usd_path)
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
    """Spawn the visual table asset and enable triangle collisions on its actual meshes."""
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation)
    for mesh in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if mesh.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr("none")
    return prim


class reset_tabletop(ManagerTermBase):
    """Place sampled objects on the reachable part of a table, then select an available task.

    Object combinations come from the scene's inclusion set and stay fixed for the lifetime of the
    scene. Resets randomize their poses. Convex footprints reject overlaps; Newton's collision
    pipeline screens the resulting poses against the robot and the actual collision shapes without
    stepping physics. Infeasible placements raise instead of accepting overlaps.

    ``env.tabletop.available_tasks[world]`` lists category- and size-compatible bindings. A reset
    chooses one unless :meth:`set_task` requests a particular binding. A world with no available task
    has an empty instruction and never reports task success; its object combination is retained.
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
        self.surface = shapely.Polygon(cfg.params["surface_outline"], cfg.params.get("surface_holes", ()))
        self.surface_height = cfg.params["surface_height"]
        if not self.surface.is_valid or self.surface.is_empty or not math.isfinite(self.surface_height):
            raise ValueError("Tabletop requires a valid support outline and a finite local surface height.")
        self.placement_regions = np.empty(env.num_envs, dtype=object)
        plan = env.sim.get_clone_plan()
        topology = plan.topology
        slots = {cfg.prim_path: index for index, cfg in enumerate(self.objects.cfg.rigid_objects.values())}
        geometry, metadata, cache = {}, {}, {}
        for prototype in np.unique(topology.world_prototypes):
            declaration = plan.asset_cfgs[prototype]
            if declaration.prim_path not in slots:
                continue
            spawn = declaration.spawn.assets_cfg[0]
            tags = spawn.semantic_tags or ()
            label = next(value for key, value in tags if key == "_catalog_key")
            if label not in cache:
                cache[label] = _collision_geometry(spawn)
            geometry[int(prototype)] = cache[label]
            metadata[int(prototype)] = (label, {value for key, value in tags if key == "category"})
        bounds = np.zeros((env.num_envs, len(self.names), 2, 3), dtype=np.float32)
        active = np.zeros((env.num_envs, len(self.names)), dtype=bool)
        vertices = max(len(polygon) for _, polygon in geometry.values())
        footprints = np.zeros((env.num_envs, len(self.names), vertices, 3), dtype=np.float32)
        labels, categories = [], []
        for world, composition in enumerate(topology.world_prototype_layout):
            start, end = topology.world_prototype_starts[composition + 1 : composition + 3]
            world_labels, world_categories = [""] * len(self.names), [set() for _ in self.names]
            for prototype in topology.world_prototypes[start:end]:
                declaration = plan.asset_cfgs[prototype]
                if declaration.prim_path not in slots:
                    continue
                slot = slots[declaration.prim_path]
                label, tags = metadata[int(prototype)]
                bounds[world, slot], polygon = geometry[int(prototype)]
                footprints[world, slot, :, :2] = polygon[-1]
                footprints[world, slot, : len(polygon), :2] = polygon
                active[world, slot] = label != "__empty__"
                world_labels[slot], world_categories[slot] = label, tags
            labels.append(world_labels)
            categories.append(world_categories)
        self.object_names = tuple(tuple(row) for row in labels)
        self.active = torch.as_tensor(active, device=env.device)
        corners = np.array(list(itertools.product((0, 1), repeat=3)))
        self.local_corners = torch.as_tensor(bounds[:, :, corners, np.arange(3)], device=env.device)
        self.local_bounds = torch.as_tensor(bounds, device=env.device)
        self.local_footprints = torch.as_tensor(footprints, device=env.device)
        self.available_tasks = []
        for world in range(env.num_envs):
            tasks = []
            for family, rule in self.families.items():
                for source, target in itertools.permutations(range(len(self.names)), 2):
                    if not active[world, source] or rule["target"] not in categories[world][target]:
                        continue
                    size = np.sort(np.diff(bounds[world, source], axis=0)[0, :2])
                    support = np.sort(np.diff(bounds[world, target], axis=0)[0, :2])
                    # A conservative inset reserves room for a rim or container walls.
                    if np.any(size + 2 * cfg.params["clearance"] > support * 0.8):
                        continue
                    instruction = rule["instruction"].format(
                        object=labels[world][source].replace("_", " "), target=labels[world][target].replace("_", " ")
                    )
                    tasks.append(Task(family, self.names[source], self.names[target], instruction))
            self.available_tasks.append(tuple(tasks))
        self.available_tasks = tuple(self.available_tasks)
        self.instructions = [""] * env.num_envs
        self.task_ids = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.object_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.target_ids = torch.zeros_like(self.object_ids)
        self.in_container = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._requested = [-1] * env.num_envs
        self._backend = env.sim.backend_manager.backend
        self._contacts = None
        object_bodies = []
        for label, world in zip(self._backend.model.body_label, self._backend.model.body_world.numpy(), strict=True):
            slot = next((i for i, name in enumerate(self.names) if f"/{name}/" in label + "/"), None)
            object_bodies.append(slot is not None and world >= 0 and active[world, slot])
        self._object_bodies = torch.tensor(object_bodies, device=env.device)
        env.tabletop = self

    def set_task(self, env_ids: list[int], task_ids: list[int]) -> None:
        """Request available binding indices on the next reset; -1 chooses randomly each episode."""
        if len(env_ids) != len(task_ids):
            raise ValueError("Each world requires one task index.")
        for world, task in zip(env_ids, task_ids, strict=True):
            if not 0 <= world < len(self.available_tasks) or not -1 <= task < len(self.available_tasks[world]):
                raise ValueError(f"Unavailable task {task} for world {world}.")
        for world, task in zip(env_ids, task_ids, strict=True):
            self._requested[world] = task

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | slice,
        objects: SceneEntityCfg,
        table: SceneEntityCfg,
        surface_outline: tuple[tuple[float, float], ...],
        surface_height: float,
        workspace: tuple[tuple[float, float, float], tuple[float, float, float]],
        gripper_body: str,
        clearance: float = 0.005,
        max_attempts: int = 128,
        surface_holes: tuple[tuple[tuple[float, float], ...], ...] = (),
    ) -> None:
        ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        if not len(ids):
            return
        if clearance <= 0 or max_attempts < 1:
            raise ValueError("Placement requires positive clearance and max_attempts.")
        workspace = torch.as_tensor(workspace, dtype=torch.float32, device=env.device)
        if workspace.shape != (2, 3) or not torch.isfinite(workspace).all() or (workspace[0] >= workspace[1]).any():
            raise ValueError("Workspace must contain finite, ordered XYZ lower and upper bounds.")
        table_poses = self.table.data.root_link_pose_w.torch[ids]
        up = quat_apply(table_poses[:, 3:], table_poses.new_tensor([0.0, 0.0, 1.0]).expand(len(ids), 3))
        if (up[:, 2] < 1 - 1e-6).any():
            raise ValueError("Tabletop reset requires a horizontal table; yaw rotation is supported.")
        origins = env.scene.env_origins[ids].cpu().numpy()
        limits = workspace.cpu().numpy()
        for world, pose, origin in zip(ids.tolist(), table_poses.cpu().numpy(), origins, strict=True):
            x, y, z, w = pose[3:]
            cosine, sine = 1 - 2 * (y * y + z * z), 2 * (w * z + x * y)
            tabletop = affine_transform(self.surface, [cosine, -sine, sine, cosine, pose[0], pose[1]])
            reachable = shapely.box(*(limits[0, :2] + origin[:2]), *(limits[1, :2] + origin[:2]))
            region = tabletop.intersection(reachable).buffer(-clearance)
            top = pose[2] + self.surface_height
            if region.is_empty or top < limits[0, 2] + origin[2] - 1e-5 or top + clearance >= limits[1, 2] + origin[2]:
                raise RuntimeError("The measured table surface and workspace have no usable intersection.")
            self.placement_regions[world] = region
        remaining = ids
        for _ in range(max_attempts):
            poses, placed = self._sample(remaining, workspace, clearance, max_attempts)
            accepted = remaining[placed]
            if len(accepted):
                self.objects.write_body_pose_to_sim_index(body_poses=poses[placed].contiguous(), env_ids=accepted)
                self.objects.write_body_velocity_to_sim_index(
                    body_velocities=torch.zeros(len(accepted), len(self.names), 6, device=env.device), env_ids=accepted
                )
                bad = self.penetrating_worlds()
                placed &= ~bad[remaining]
            remaining = remaining[~placed]
            if not len(remaining):
                break
        else:
            raise RuntimeError(f"No collision-free tabletop reset for worlds {remaining.tolist()}.")
        for world in ids.tolist():
            available = self.available_tasks[world]
            task = self._requested[world]
            if available and task < 0:
                task = int(torch.randint(len(available), (), device=env.device))
            self.task_ids[world] = task
            if task < 0:
                self.instructions[world] = ""
                continue
            binding = available[task]
            self.object_ids[world], self.target_ids[world] = (
                self.names.index(binding.object),
                self.names.index(binding.target),
            )
            self.in_container[world] = self.families[binding.family]["relation"] == "in"
            self.instructions[world] = binding.instruction

    def _sample(
        self, ids: torch.Tensor, workspace: torch.Tensor, clearance: float, max_attempts: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        env = self._env
        count, width = len(ids), len(self.names)
        origin = env.scene.env_origins[ids]
        regions = self.placement_regions[ids.cpu().numpy()]
        bounds = torch.as_tensor(shapely.bounds(regions), dtype=torch.float32, device=env.device)
        lo, hi = bounds[:, :2], bounds[:, 2:]
        top = self.table.data.root_link_pose_w.torch[ids, 2] + self.surface_height
        poses = torch.zeros(count, width, 7, device=env.device)
        poses[..., 6] = 1.0
        poses[..., :3] = origin[:, None] + poses.new_tensor([0.0, -10.0, 0.5])
        poses[..., 0] += torch.arange(width, device=env.device) * 0.02
        low, high = torch.zeros_like(poses[..., :3]), torch.zeros_like(poses[..., :3])
        footprints = torch.zeros(count, width, self.local_footprints.shape[2], 3, device=env.device)
        placed = torch.zeros(count, width, dtype=torch.bool, device=env.device)
        sizes = self.local_bounds[ids, :, 1] - self.local_bounds[ids, :, 0]
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
                    quat[:, None].expand(count, self.local_footprints.shape[2], 4), self.local_footprints[ids, slots]
                )
                extent_lo, extent_hi = rotated.amin(1), rotated.amax(1)
                extent_lo[:, 2], extent_hi[:, 2] = (
                    self.local_bounds[ids, slots, 0, 2],
                    self.local_bounds[ids, slots, 1, 2],
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
                supported = shapely.covers(regions, shapely.polygons(world_points[..., :2].cpu().numpy()))
                valid &= torch.as_tensor(supported, device=env.device)
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
        count = int(wp.to_torch(contacts.rigid_contact_count)[0])
        if count > contacts.rigid_contact_max:
            raise RuntimeError("Tabletop reset collision buffer overflowed.")
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
