# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prototype topology and placement shared by every clone backend."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import warp as wp

from .cloner_cfg import DEFAULT_ENV_TEMPLATE
from .cloner_strategies import sequential


@dataclass(frozen=True, eq=False)
class PrototypeWorldTopology:
    """Numeric asset-prototype and world relationships.

    Arrays use NumPy storage for planning or Warp storage on one device for runtime queries.
    Treat the topology as read-only after planning; :func:`to_warp` explicitly materializes
    its numeric arrays on a device. Host declarations and naming belong to :class:`ClonePlan`.
    """

    num_asset_prototypes: int
    """Number of asset definitions, including unused prototypes."""

    world_prototypes: np.ndarray | wp.array
    """Flat int32 asset-prototype indices. Repeated indices represent distinct instances."""

    world_prototype_starts: np.ndarray | wp.array
    """Offsets into :attr:`world_prototypes`, starting with the shared world `-1`.

    Shared assets occupy ``world_prototypes[world_prototype_starts[0]:world_prototype_starts[1]]``.
    World prototype ``i`` occupies ``world_prototypes[world_prototype_starts[i + 1]:world_prototype_starts[i + 2]]``.
    An empty shared world starts with ``[0, 0]``. Offsets have dtype int64.
    """

    world_prototype_layout: np.ndarray | wp.array
    """Int32 world-prototype index per world, indexed by world ID; shared assets are not sampled."""


@dataclass(frozen=True, eq=False)
class ClonePlan:
    """Prototype topology and placement used to instantiate a scene."""

    topology: PrototypeWorldTopology
    """Numeric asset-prototype and world membership, independent of naming and placement."""

    asset_cfgs: tuple[Any, ...]
    """Host declarations indexed by asset-prototype ID, retained once by reference."""

    env_template: str = DEFAULT_ENV_TEMPLATE
    """Destination-world path template, with one ``{}`` slot for the world ID."""

    positions: np.ndarray | None = None
    """Destination-world origins [m], shape [num_worlds, 3]; None preserves authored placement."""


def to_warp(topology: PrototypeWorldTopology, device: str) -> PrototypeWorldTopology:
    """Materialize numeric topology on an explicitly selected device.

    Args:
        topology: Host topology. Contiguous arrays with matching dtypes are borrowed on CPU and copied on CUDA.
        device: Warp device, such as ``"cpu"`` or ``"cuda:0"``.

    Returns:
        A new topology holding its arrays alive independently of the host topology. Call once during
        initialization and share the result; this function does not cache, synchronize later
        host edits, or transfer cfgs. Queries never call it implicitly.
    """
    return PrototypeWorldTopology(
        num_asset_prototypes=topology.num_asset_prototypes,
        world_prototypes=wp.array(topology.world_prototypes, dtype=wp.int32, device=device, copy=False),
        world_prototype_starts=wp.array(topology.world_prototype_starts, dtype=wp.int64, device=device, copy=False),
        world_prototype_layout=wp.array(topology.world_prototype_layout, dtype=wp.int32, device=device, copy=False),
    )


def make_clone_plan(
    asset_cfgs: Sequence[Any],
    world_prototypes: Sequence[Sequence[int]],
    num_worlds: int,
    *,
    weights: Sequence[float] | None = None,
    shared_assets: Sequence[int] = (),
    clone_strategy: Callable[[np.ndarray, int], np.ndarray] = sequential,
    env_template: str = DEFAULT_ENV_TEMPLATE,
    positions: np.ndarray | None = None,
) -> ClonePlan:
    """Select world compositions and retain their optional placement without creating native resources.

    Args:
        asset_cfgs: Asset prototype definitions, retained by reference.
        world_prototypes: Asset indices in each world prototype, including repeated instances.
        num_worlds: Number of destination worlds.
        weights: Relative world-prototype weights; ``None`` gives every prototype equal weight.
        shared_assets: Asset indices instantiated once in the shared world ``-1``.
        clone_strategy: Function selecting world-prototype indices from weights.
        env_template: Destination-world path template with one ``{}`` slot for the world ID.
        positions: Destination-world origins [m], shape [num_worlds, 3]; None preserves authored placement.

    Returns:
        A plan holding the topology and placement. Topology starts with a shared-world slice.
    """
    asset_cfgs = tuple(asset_cfgs)
    compositions = (tuple(shared_assets), *(tuple(world) for world in world_prototypes))
    if len(compositions) == 1:
        raise ValueError("At least one world prototype is required; an empty world is ().")
    members = np.asarray([asset for world in compositions for asset in world])
    if members.size and (
        not np.issubdtype(members.dtype, np.integer) or (members < 0).any() or (members >= len(asset_cfgs)).any()
    ):
        raise ValueError("World members must be integer indices into asset_cfgs.")
    weights = np.ones(len(compositions) - 1) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(compositions) - 1,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Each world prototype requires one finite, non-negative weight.")
    if weights.sum() <= 0 or num_worlds < 0:
        raise ValueError("Weights must have positive total mass and num_worlds must be non-negative.")
    world_prototype_layout = np.asarray(clone_strategy(weights, num_worlds))
    if (
        world_prototype_layout.shape != (num_worlds,)
        or not np.issubdtype(world_prototype_layout.dtype, np.integer)
        or (world_prototype_layout < 0).any()
        or (world_prototype_layout >= len(weights)).any()
    ):
        raise ValueError("clone_strategy must select one valid world-prototype index per destination.")
    return ClonePlan(
        topology=PrototypeWorldTopology(
            num_asset_prototypes=len(asset_cfgs),
            world_prototypes=np.ascontiguousarray(members, dtype=np.int32),
            world_prototype_starts=np.cumsum([0, *(len(world) for world in compositions)], dtype=np.int64),
            world_prototype_layout=np.ascontiguousarray(world_prototype_layout, dtype=np.int32),
        ),
        asset_cfgs=asset_cfgs,
        env_template=env_template,
        positions=positions,
    )


def grid_transforms(N: int, spacing: float = 1.0, up_axis: str = "z") -> tuple[np.ndarray, np.ndarray]:
    """Create centered grid transforms as host arrays.

    Args:
        N: Number of instances.
        spacing: Distance between neighboring grid positions [m].
        up_axis: Up axis for positions (``"z"``, ``"y"``, or ``"x"``).

    Returns:
        Positions [m], shape ``[N, 3]``, and identity xyzw orientations, shape ``[N, 4]``.
    """
    num_rows = int(math.ceil(N / math.sqrt(N)))
    num_cols = int(math.ceil(N / num_rows))
    ii, jj = np.meshgrid(np.arange(num_rows, dtype=np.float32), np.arange(num_cols, dtype=np.float32), indexing="ij")
    ii = ii.reshape(-1)[:N]
    jj = jj.reshape(-1)[:N]
    x = -(ii - (num_rows - 1) / 2) * spacing
    y = (jj - (num_cols - 1) / 2) * spacing
    zero = np.zeros(N, dtype=np.float32)
    if up_axis.lower() == "z":
        positions = np.stack((x, y, zero), axis=1)
    elif up_axis.lower() == "y":
        positions = np.stack((x, zero, y), axis=1)
    else:
        positions = np.stack((zero, x, y), axis=1)
    orientations = np.zeros((N, 4), dtype=np.float32)
    orientations[:, 3] = 1.0
    return positions.astype(np.float32, copy=False), orientations


class TemplateMatch(NamedTuple):
    """The ``"{}"`` text a template captured (``"3"``, or a wildcard ``".*"``), and the path below it."""

    instance: str
    suffix: str


class path:
    """Stateless prim-path operations for clone plans.

    Concrete roots are matched on segment boundaries; templates carry one ``"{}"`` instance slot.
    Call these functions through ``cloner.path`` without constructing an instance.
    """

    @staticmethod
    def get_asset_prototypes(plan: ClonePlan, path_expr: str | None = None) -> np.ndarray:
        """Select asset-prototype IDs by their declared cfg paths, without expanding instances.

        Args:
            plan: Host asset declarations and their numeric topology.
            path_expr: Exact cfg ``prim_path`` or a regular expression matching the complete declared
                path string. None selects all definitions, including unused prototypes.

        Returns:
            Ascending asset-prototype IDs, shape [num_matches], dtype int32, each included once.
            Generated native paths are not matched.
        """
        if path_expr is None:
            return np.arange(len(plan.asset_cfgs), dtype=np.int32)
        pattern = re.compile(path_expr)
        paths = (cfg.prim_path for cfg in plan.asset_cfgs)
        return np.fromiter(
            (index for index, path in enumerate(paths) if path == path_expr or pattern.fullmatch(path)), dtype=np.int32
        )

    @staticmethod
    def get_world_prototypes(plan: ClonePlan, path_expr: str | None = None) -> np.ndarray:
        """Select world-prototype IDs containing assets matched by their declared cfg paths.

        Args:
            plan: Host asset declarations and their numeric topology.
            path_expr: Asset-path filter interpreted by :meth:`path.get_asset_prototypes`. None selects
                all world definitions, including empty and unused prototypes and shared world -1.

        Returns:
            Ascending world-prototype IDs, shape [num_matches], dtype int32, not destination world IDs.
            Filtering selects complete compositions; repeated asset memberships remain in the topology.
        """
        topology = plan.topology
        prototype_ids = np.arange(-1, len(topology.world_prototype_starts) - 2, dtype=np.int32)
        if path_expr is None:
            return prototype_ids
        matched_assets = np.isin(topology.world_prototypes, path.get_asset_prototypes(plan, path_expr))
        match_counts = np.r_[0, np.cumsum(matched_assets)]
        return prototype_ids[np.diff(match_counts[topology.world_prototype_starts]) > 0]

    @staticmethod
    def get_instance_paths(plan: ClonePlan) -> tuple[tuple[int, str | None, str, np.ndarray], ...]:
        """Resolve native names for the plan's declared asset instances without accessing a stage.

        Args:
            plan: Host plan supplying declarations, membership, and the destination-world template.

        Returns:
            Asset-prototype ID, authored source path, destination template, and world IDs per named
            occurrence. Shared instances use world -1; unused occurrences have no source path.
            Repeated memberships inherit the prototype pose and receive distinct sibling names.
        """
        topology = plan.topology
        targets = {}
        sorted_world_ids = np.argsort(topology.world_prototype_layout, kind="stable")
        counts = np.bincount(topology.world_prototype_layout, minlength=len(topology.world_prototype_starts) - 2)
        offsets = np.r_[0, np.cumsum(counts)]
        for world_prototype_id in path.get_world_prototypes(plan):
            start, end = topology.world_prototype_starts[world_prototype_id + 1 : world_prototype_id + 3]
            world_ids = (
                np.array([-1])
                if world_prototype_id == -1
                else sorted_world_ids[offsets[world_prototype_id] : offsets[world_prototype_id + 1]]
            )
            names = set()
            for asset_prototype_id in topology.world_prototypes[start:end]:
                cfg = plan.asset_cfgs[asset_prototype_id]
                matched = path.match(cfg.prim_path, plan.env_template)
                template = plan.env_template + matched.suffix if matched is not None else cfg.prim_path
                if world_prototype_id == -1:
                    template = template.format("shared")
                elif matched is None:
                    template = plan.env_template + "/" + cfg.prim_path.rsplit("/", 1)[-1]
                name, occurrence = template, 0
                while template in names:
                    occurrence += 1
                    template = f"{name}_{occurrence}"
                names.add(template)
                targets.setdefault((int(asset_prototype_id), template), []).append(world_ids)
        targets = [(index, template, np.concatenate(groups)) for (index, template), groups in targets.items()]
        source_paths = {}
        for asset_prototype_id, template, world_ids in targets:
            if len(world_ids) and asset_prototype_id not in source_paths:
                spawn = getattr(plan.asset_cfgs[asset_prototype_id], "spawn", None)
                source_path = getattr(spawn, "spawn_path", None)
                source_paths[asset_prototype_id] = (
                    source_path if source_path is not None else template.format(int(world_ids[0]))
                )
        return tuple(
            (asset_prototype_id, source_paths.get(asset_prototype_id), template, world_ids)
            for asset_prototype_id, template, world_ids in targets
        )

    @staticmethod
    def get_shared_paths(instances: Iterable[tuple[int, str | None, str, np.ndarray]]) -> tuple[str, ...]:
        """Return minimal shared roots from :meth:`path.get_instance_paths`, without inferring membership from names."""
        paths = tuple(template for _, _, template, world_ids in instances if len(world_ids) and world_ids[0] == -1)
        return tuple(
            prim_path
            for prim_path in paths
            if not any(prim_path != root and path.under(prim_path, root) for root in paths)
        )

    @staticmethod
    def iter_subtree_copies(
        instances: Iterable[tuple[int, str | None, str, np.ndarray]],
    ) -> Iterator[tuple[int, str, str, np.ndarray]]:
        """Yield parent-first subtree copies, preserving independently sourced child overrides.

        Args:
            instances: Routed instance paths from :meth:`path.get_instance_paths`.

        Yields:
            Asset-prototype ID, source path, destination template, and world IDs requiring a copy.
        """
        instances = sorted(
            (instance for instance in instances if len(instance[3])),
            key=lambda item: item[2].count("/"),
        )
        for index, (asset_prototype_id, source, destination, world_ids) in enumerate(instances):
            covered = np.zeros(len(world_ids), dtype=np.bool_)
            redundant = covered.copy()
            for _, parent_source, parent_destination, parent_world_ids in reversed(instances[:index]):
                if destination == parent_destination or not path.under(destination, parent_destination):
                    continue
                inherited = np.isin(world_ids, parent_world_ids) & ~covered
                if path.rebase(source, parent_source, parent_destination) == destination:
                    redundant |= inherited
                covered |= inherited
            if not redundant.all():
                yield asset_prototype_id, source, destination, world_ids[~redundant]

    @staticmethod
    def split(template: str) -> tuple[str, str]:
        """Split a clone destination template around its ``"{}"`` clone slot.

        The clone slot represents one concrete environment/instance path segment.

        Args:
            template: Destination path template with exactly one ``"{}"`` for the instance id.

        Returns:
            The ``(prefix, suffix)`` strings around the clone slot. A trailing slash is
            insignificant, so an instance-root template (``".../env_{}"``) yields an empty suffix.

        Raises:
            ValueError: If ``template`` does not hold exactly one clone slot. A second slot would
                survive into the suffix and break the later ``str.format`` that fills the first.
        """
        template = template.rstrip("/") or "/"
        slots = template.count("{}")
        if slots != 1:
            raise ValueError(
                f"Clone destination template must contain exactly one '{{}}', found {slots}: {template!r}."
            )
        prefix, _, suffix = template.partition("{}")
        return prefix, suffix

    @staticmethod
    def match(path_expr: str, template: str) -> TemplateMatch | None:
        """Match ``path_expr`` against a destination template, capturing the instance slot.

        The ``"{}"`` slot matches one path segment's worth of text: a concrete id (``3``) or a
        wildcard standing for one segment (``.*``, ``[^/]+``). The captured text identifies
        the instance without slicing the path by hand.

        Args:
            path_expr: Path or path expression on the clone (destination) side.
            template: Destination path template with ``"{}"`` for the instance id.

        Returns:
            A :class:`TemplateMatch` with the captured instance text and the asset-relative
            suffix, or ``None`` when ``path_expr`` is not under the template's instance root.

        Example:
            >>> path.match("/World/envs/env_3/Robot/base", "/World/envs/env_{}/Robot")
            TemplateMatch(instance='3', suffix='/base')
        """
        prefix, template_suffix = path.split(template)
        # the slot holds one segment's worth of text: a concrete id, or a wildcard standing for one.
        # A segment-safe wildcard is written as a character class, whose text contains a '/' that is
        # not a separator, so it is matched as a class rather than by the one-segment alternative.
        pattern = re.compile(re.escape(prefix) + r"(\[\^?[^]]*\][*+?]?|[^/]+)" + re.escape(template_suffix))
        matched = pattern.match(path_expr)
        if matched is None:
            return None
        suffix = path_expr[matched.end() :]
        if suffix and not suffix.startswith("/"):
            return None
        return TemplateMatch(matched.group(1), suffix)

    @staticmethod
    def relative_to(path: str, root: str) -> str | None:
        """Strip a concrete ``root`` prefix off ``path`` on a segment boundary.

        Unlike slicing or :meth:`str.removeprefix`, this returns ``None`` rather than a
        mid-segment remainder when ``path`` is not under ``root``.

        Args:
            path: Path to make relative.
            root: Concrete subtree root. A trailing slash is insignificant, and ``"/"`` is the
                root of every path.

        Returns:
            The suffix below ``root`` (starting with ``/``, or ``""`` when ``path`` equals
            ``root``), or ``None`` when ``path`` is not under ``root``.

        """
        root = root.rstrip("/") or "/"
        if path == root:
            return ""
        # "/" prefixes every path but contributes no segment of its own.
        prefix = "" if root == "/" else root
        if not path.startswith(prefix):
            return None
        suffix = path[len(prefix) :]
        return suffix if suffix.startswith("/") else None

    @classmethod
    def under(cls, path: str, root: str) -> bool:
        """Return whether ``path`` lies within the subtree rooted at ``root``.

        Boundary-correct membership test: unlike :meth:`str.startswith`, it does not match
        across a segment boundary (``".../Robot"`` does not contain ``".../RobotArm"``).

        Args:
            path: Candidate descendant path.
            root: Concrete subtree root.

        Returns:
            ``True`` when ``path`` equals ``root`` or is a descendant of it.
        """
        return cls.relative_to(path, root) is not None

    @classmethod
    def rebase(cls, path: str, src_root: str, dst_root: str) -> str:
        """Rebase ``path`` from one concrete root prefix onto another on a segment boundary.

        Unlike :meth:`str.replace`, this swaps only a boundary-aligned prefix and touches only
        the leading occurrence.

        Args:
            path: Path to rebase.
            src_root: Concrete source root prefix.
            dst_root: Concrete destination root prefix.

        Returns:
            The rebased path, or ``path`` unchanged when it is not under ``src_root``.
        """
        suffix = cls.relative_to(path, src_root)
        if suffix is None:
            return path
        return (dst_root.rstrip("/") + suffix) or "/"

    @staticmethod
    def relativize(path_expr: str, template: str) -> str | None:
        """Return the part of ``path_expr`` below a template's instance root.

        The suffix half of :meth:`path.match`, for callers that do not need the captured instance.

        Args:
            path_expr: Path or path expression on the clone (destination) side.
            template: Destination path template with ``"{}"`` for the instance id.

        Returns:
            The asset-relative suffix (starting with ``/``, or ``""`` when ``path_expr`` is
            exactly the template root), or ``None`` when ``path_expr`` is not under the root.

        """
        matched = path.match(path_expr, template)
        return None if matched is None else matched.suffix


class query:
    """Batched numeric topology queries. Resolve declared paths separately with :class:`path`.

    All queries return ``(world_indices, world_starts)``. Indices are flat int32 values;
    starts are int64 offsets with shape [num_queries, num_worlds + 2], including shared world -1.
    For query q and world w, ``world_starts[q, w + 1 : w + 3]`` bounds its selected instances.
    Each row's first/last offset bounds the entire query. Repeated IDs retain separate results.

    NumPy queries allocate exact-sized results. Warp queries require resident int32 query IDs
    and preallocated ``out`` arrays on the topology's device; no upload or readback is implicit.
    Warm up before CUDA graph capture. For nonempty batches, the valid prefix ends at ``world_starts[-1, -1]``.
    If that required size exceeds capacity, starts are still reported but indices are left untouched:
    the caller must provide enough capacity for its selection domain, not consume a partial result.
    """

    @staticmethod
    def get_asset_prototype_world_index(
        topology: PrototypeWorldTopology,
        asset_prototype: int | np.ndarray | wp.array,
        *,
        out: tuple[np.ndarray, np.ndarray] | tuple[wp.array, wp.array] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[wp.array, wp.array]:
        """Return one world index per asset instance, retaining repeated memberships.

        Args:
            topology: Numeric topology with NumPy or Warp storage.
            asset_prototype: One integer (NumPy only), or a 1-D integer array of asset-prototype IDs.
            out: Optional NumPy outputs, required Warp outputs. See the namespace's result/capacity contract.

        Returns:
            Flat world indices and per-query world boundaries. A scalar is a batch of length one.
            Missing or unused asset IDs produce empty slices; shared instances use world -1.
        """
        return query._world_index(topology, asset_prototype, by_asset=True, unique=False, out=out)

    @staticmethod
    def get_asset_prototype_unique_world_index(
        topology: PrototypeWorldTopology,
        asset_prototype: int | np.ndarray | wp.array,
        *,
        out: tuple[np.ndarray, np.ndarray] | tuple[wp.array, wp.array] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[wp.array, wp.array]:
        """Return each world containing an asset once, independently for every requested asset.

        Args:
            topology: Numeric topology with NumPy or Warp storage.
            asset_prototype: One integer (NumPy only), or a 1-D integer array of asset-prototype IDs.
            out: Optional NumPy outputs, required Warp outputs. See the namespace's result/capacity contract.

        Returns:
            Flat world indices and per-query world boundaries, as in :meth:`query.get_asset_prototype_world_index`.
            Every world slice has length zero or one. Separate queries are not deduplicated together.
        """
        return query._world_index(topology, asset_prototype, by_asset=True, unique=True, out=out)

    @staticmethod
    def get_world_prototype_world_index(
        topology: PrototypeWorldTopology,
        world_prototype: int | np.ndarray | wp.array,
        *,
        out: tuple[np.ndarray, np.ndarray] | tuple[wp.array, wp.array] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[wp.array, wp.array]:
        """Return the destination worlds using each requested world prototype.

        Args:
            topology: Numeric topology with NumPy or Warp storage.
            world_prototype: One integer (NumPy only), or a 1-D integer array of world-prototype IDs.
                Index -1 selects the shared world, even when empty.
            out: Optional NumPy outputs, required Warp outputs. See the namespace's result/capacity contract.

        Returns:
            Flat world indices and per-query world boundaries, as in :meth:`query.get_asset_prototype_world_index`.
            Unused world prototypes produce empty slices.
        """
        return query._world_index(topology, world_prototype, by_asset=False, unique=True, out=out)

    @staticmethod
    def _world_index(
        topology: PrototypeWorldTopology,
        prototype_ids: int | np.ndarray | wp.array,
        *,
        by_asset: bool,
        unique: bool,
        out: tuple[np.ndarray, np.ndarray] | tuple[wp.array, wp.array] | None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[wp.array, wp.array]:
        num_worlds = len(topology.world_prototype_layout)
        if isinstance(topology.world_prototypes, np.ndarray):
            if isinstance(prototype_ids, wp.array):
                raise TypeError("Warp query IDs require to_warp(topology, device); queries do not transfer arrays.")
            prototype_ids = np.atleast_1d(prototype_ids)
            if prototype_ids.ndim != 1 or prototype_ids.dtype.kind not in "iu":
                raise TypeError("Topology queries require integer IDs; resolve expressions with cloner.path.")
            layout = np.r_[-1, topology.world_prototype_layout]
            if by_asset:
                matches = prototype_ids[:, None] == topology.world_prototypes
                prefix = np.zeros((len(prototype_ids), len(topology.world_prototypes) + 1), dtype=np.int64)
                np.cumsum(matches, axis=1, out=prefix[:, 1:])
                counts = np.diff(prefix[:, topology.world_prototype_starts], axis=1)[:, layout + 1]
                if unique:
                    counts = counts > 0
            else:
                counts = prototype_ids[:, None] == layout
            starts = np.zeros((len(prototype_ids), num_worlds + 2), dtype=np.int64)
            starts[:, 1:] = counts
            np.cumsum(starts.ravel(), out=starts.ravel())
            indices = np.repeat(np.tile(np.arange(-1, num_worlds, dtype=np.int32), len(prototype_ids)), counts.ravel())
            if out is None:
                return indices, starts
            out[1][:] = starts
            if len(indices) <= len(out[0]):
                out[0][: len(indices)] = indices
        else:
            if out is None:
                raise ValueError("Warp queries require preallocated out=(world_indices, world_starts).")
            indices, starts = out
            if starts.shape != (len(prototype_ids), num_worlds + 2) or not starts.is_contiguous:
                raise ValueError("world_starts must be contiguous with shape [num_queries, num_worlds + 2].")
            wp.launch(
                _count_world_instances,
                dim=starts.shape,
                inputs=[
                    topology.world_prototypes,
                    topology.world_prototype_starts,
                    topology.world_prototype_layout,
                    prototype_ids,
                    by_asset,
                    unique,
                ],
                outputs=[starts],
                device=starts.device,
            )
            wp.utils.array_scan(starts.flatten(), starts.flatten())
            wp.launch(
                _fill_world_indices, (len(prototype_ids), num_worlds + 1), [starts, indices], device=starts.device
            )
        return out


@wp.kernel
def _count_world_instances(
    world_prototypes: wp.array(dtype=wp.int32),
    world_prototype_starts: wp.array(dtype=wp.int64),
    world_prototype_layout: wp.array(dtype=wp.int32),
    prototype_ids: wp.array(dtype=wp.int32),
    by_asset: bool,
    unique: bool,
    starts: wp.array2d(dtype=wp.int64),
):
    query, column = wp.tid()
    count = wp.int64(0)
    if column > 0:
        world_prototype = -1
        if column > 1:
            world_prototype = world_prototype_layout[column - 2]
        if by_asset:
            member = world_prototype_starts[world_prototype + 1]
            end = world_prototype_starts[world_prototype + 2]
            while member < end:
                if world_prototypes[member] == prototype_ids[query]:
                    count += wp.int64(1)
                member += wp.int64(1)
            if unique:
                count = wp.min(count, wp.int64(1))
        elif world_prototype == prototype_ids[query]:
            count = wp.int64(1)
    starts[query, column] = count


@wp.kernel
def _fill_world_indices(starts: wp.array2d(dtype=wp.int64), indices: wp.array(dtype=wp.int32)):
    query, world = wp.tid()
    # Never write a partial result or past capacity, including during graph replay.
    if starts[starts.shape[0] - 1, starts.shape[1] - 1] <= wp.int64(indices.shape[0]):
        index = starts[query, world]
        while index < starts[query, world + 1]:
            indices[index] = world - 1
            index += wp.int64(1)
