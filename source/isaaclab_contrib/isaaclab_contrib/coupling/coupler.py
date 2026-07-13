# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton coupler for named solver configurations."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, fields
from functools import wraps
from typing import TYPE_CHECKING

from isaaclab_newton.physics import (
    KaminoSolverCfg,
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonSolverCfg,
)
from isaaclab_newton.physics.mpm_manager import NewtonMPMManager
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import Model, ModelBuilder, ShapeFlags
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM, SolverCoupledProxy

from isaaclab.managers import SceneEntityCfg
from isaaclab.physics import PhysicsManager
from isaaclab.utils.string import resolve_matching_names

from ..deformable.vbd_manager import NewtonVBDManager
from .coupler_cfg import (
    CouplerAdmmCfg,
    CouplerCfg,
    CouplerEntryCfg,
    CouplerProxyCfg,
    CouplerProxyMappingCfg,
)

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveSceneCfg


logger = logging.getLogger(__name__)


class NewtonCouplerManager(NewtonVBDManager):
    """Couple named Newton solver entries through proxy or ADMM interfaces."""

    @dataclass
    class _ResolvedEntry:
        """Entry configuration with model selectors resolved to indices."""

        config: CouplerEntryCfg
        bodies: list[int]
        particles: list[int]
        joints: list[int]
        shapes: list[int]

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: CouplerCfg):
        """Reject recursive use as a nested coupled-solver entry."""
        del model, solver_cfg
        raise NotImplementedError("Nested Newton couplers are not supported.")

    @staticmethod
    def _requires_external_contacts(solver_cfg: NewtonSolverCfg) -> bool:
        """Return whether a sub-solver expects contacts from Newton's collision pipeline.

        Unknown solver configs conservatively opt in to external contacts.
        """
        if isinstance(solver_cfg, MJWarpSolverCfg):
            return not solver_cfg.use_mujoco_contacts
        if isinstance(solver_cfg, KaminoSolverCfg):
            return not solver_cfg.use_collision_detector
        if isinstance(solver_cfg, MPMSolverCfg):
            return False
        return True

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CouplerCfg) -> None:
        """Resolve ownership and construct the selected coupled solver."""
        if NewtonManager._report_contacts:
            raise NotImplementedError(
                "Newton contact sensors are not yet supported by coupled solvers because contact forces live "
                "in per-entry buffers. Remove the contact sensor."
            )

        cls._validate_config(solver_cfg)
        scene_cfg = solver_cfg.scene_cfg

        resolved_entries = [cls._resolve_entry(model, entry, scene_cfg) for entry in solver_cfg.entries]
        cls._validate_resolved_entries(model, resolved_entries, solver_cfg)
        entries = [cls._build_entry(entry) for entry in resolved_entries]

        if isinstance(solver_cfg, CouplerProxyCfg):
            proxies = [cls._resolve_proxy(model, proxy, scene_cfg) for proxy in solver_cfg.proxies]
            NewtonManager._solver = cls._build_proxy_coupled_solver(model, entries, proxies, solver_cfg)
            proxy_destinations = {proxy.destination for proxy in proxies}
            outer_contact_destinations = {proxy.destination for proxy in proxies if proxy.collision_pipeline is None}
            needs_collision_pipeline = any(
                (entry.config.name not in proxy_destinations or entry.config.name in outer_contact_destinations)
                and cls._requires_external_contacts(entry.config.solver_cfg)
                for entry in resolved_entries
            )
        elif isinstance(solver_cfg, CouplerAdmmCfg):
            NewtonManager._solver = cls._build_admm_coupled_solver(model, entries, solver_cfg)
            needs_collision_pipeline = True
        else:
            raise TypeError(
                f"CouplerCfg subclass {type(solver_cfg).__name__!r} is not supported; "
                "use CouplerProxyCfg or CouplerAdmmCfg."
            )

        NewtonManager._use_single_state = False
        NewtonManager._supports_contact_sensors = False
        NewtonManager._needs_collision_pipeline = needs_collision_pipeline
        NewtonManager._needs_fk_before_step = any(
            isinstance(entry.config.solver_cfg, MPMSolverCfg) for entry in resolved_entries
        )

    @classmethod
    def _validate_config(cls, solver_cfg: CouplerCfg) -> None:
        """Validate references and unsupported nested-manager behavior before construction."""
        if not isinstance(solver_cfg, (CouplerProxyCfg, CouplerAdmmCfg)):
            raise TypeError(
                f"CouplerCfg subclass {type(solver_cfg).__name__!r} is not supported; "
                "use CouplerProxyCfg or CouplerAdmmCfg."
            )
        if not solver_cfg.entries:
            raise ValueError("CouplerCfg.entries must contain at least one solver entry.")

        names = [entry.name for entry in solver_cfg.entries]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("CouplerCfg entry names must be non-empty strings.")
        duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicate_names:
            raise ValueError(f"CouplerCfg entry names must be unique; duplicates: {duplicate_names}.")
        name_set = set(names)

        for entry in solver_cfg.entries:
            nested_cfg = entry.solver_cfg
            if not isinstance(nested_cfg, NewtonSolverCfg):
                raise TypeError(
                    f"CouplerEntryCfg {entry.name!r} solver_cfg must be a NewtonSolverCfg, "
                    f"got {type(nested_cfg).__name__}."
                )
            if isinstance(nested_cfg, CouplerCfg):
                raise ValueError(
                    f"CouplerEntryCfg {entry.name!r} contains a nested CouplerCfg; nested couplers are not supported."
                )
            if getattr(nested_cfg, "model_cfg", None) is not None:
                raise ValueError(
                    f"CouplerEntryCfg {entry.name!r} sets solver_cfg.model_cfg, but model parameters are global. "
                    "Set model_cfg on the outer CouplerCfg instead."
                )
            manager = nested_cfg.class_type
            factory = getattr(manager, "_create_solver", None)
            if not callable(factory) or getattr(factory, "__func__", factory) is NewtonManager._create_solver.__func__:
                raise TypeError(
                    f"CouplerEntryCfg {entry.name!r} uses {type(nested_cfg).__name__}, whose manager "
                    "does not implement nested solver construction."
                )
            if isinstance(nested_cfg, KaminoSolverCfg):
                raise NotImplementedError(
                    f"CouplerEntryCfg {entry.name!r} uses KaminoSolverCfg, whose manager-specific FK/reset "
                    "lifecycle cannot yet be preserved inside Newton's coupled-solver entry API."
                )
            if isinstance(nested_cfg, MPMSolverCfg) and nested_cfg.project_outside_colliders:
                raise NotImplementedError(
                    f"CouplerEntryCfg {entry.name!r} enables MPMSolverCfg.project_outside_colliders, whose "
                    "manager-level post-step projection cannot yet run inside a coupled-solver entry."
                )
            if isinstance(nested_cfg, MPMSolverCfg) and not entry.in_place:
                raise ValueError(f"CouplerEntryCfg {entry.name!r} uses MPMSolverCfg and must set in_place=True.")
            if isinstance(nested_cfg, MJWarpSolverCfg) and nested_cfg.use_mujoco_cpu:
                raise NotImplementedError(
                    f"CouplerEntryCfg {entry.name!r} enables MJWarpSolverCfg.use_mujoco_cpu, whose global reset "
                    "state cannot yet preserve the manager's per-world reset-mask lifecycle inside a coupled entry."
                )

        if isinstance(solver_cfg, CouplerProxyCfg):
            if len(solver_cfg.entries) > 2:
                raise ValueError("CouplerProxyCfg supports at most two solver entries.")
            for proxy in solver_cfg.proxies:
                cls._validate_entry_reference(name_set, proxy.source, "proxy source")
                cls._validate_entry_reference(name_set, proxy.destination, "proxy destination")
                if proxy.source == proxy.destination:
                    raise ValueError(
                        f"CouplerProxyMappingCfg source and destination must differ, got {proxy.source!r}."
                    )
                if proxy.collision_pipeline is not None and not callable(proxy.collision_pipeline):
                    raise TypeError("CouplerProxyMappingCfg.collision_pipeline must be callable or None.")
                if proxy.collision_pipeline is None and proxy.collide_interval is not None:
                    raise ValueError(
                        "CouplerProxyMappingCfg.collide_interval requires a proxy-local collision_pipeline."
                    )
        else:
            if solver_cfg.contact_pairs is not None:
                for source, destination in solver_cfg.contact_pairs:
                    cls._validate_entry_reference(name_set, source, "ADMM contact-pair source")
                    cls._validate_entry_reference(name_set, destination, "ADMM contact-pair destination")
                    if source == destination:
                        raise ValueError(f"ADMM contact-pair entries must differ, got {source!r}.")
            if solver_cfg.joint_proximal_destination_entries is not None:
                for name in solver_cfg.joint_proximal_destination_entries:
                    cls._validate_entry_reference(name_set, name, "ADMM joint-proximal destination")

    @staticmethod
    def _validate_entry_reference(entry_names: set[str], name: str, field: str) -> None:
        """Raise when a coupling reference does not name a configured entry."""
        if name not in entry_names:
            raise ValueError(f"CouplerCfg {field} {name!r} is not one of the configured entries {sorted(entry_names)}.")

    @classmethod
    def _validate_resolved_entries(
        cls,
        model: Model,
        entries: list[_ResolvedEntry],
        solver_cfg: CouplerCfg,
    ) -> None:
        """Validate resolved ownership and report intentionally unassigned model elements."""
        for entry in entries:
            counts = (len(entry.bodies), len(entry.particles), len(entry.joints), len(entry.shapes))
            if not any(counts):
                raise ValueError(f"CouplerEntryCfg {entry.config.name!r} owns no bodies, particles, joints, or shapes.")
            logger.info(
                "[COUPLER] Entry %r owns %d bodies, %d particles, %d joints, and %d shapes.",
                entry.config.name,
                *counts,
            )

        owners = {
            "bodies": cls._build_ownership_map(model.body_count, entries, "bodies"),
            "particles": cls._build_ownership_map(model.particle_count, entries, "particles"),
            "joints": cls._build_ownership_map(model.joint_count, entries, "joints"),
            "shapes": cls._build_ownership_map(model.shape_count, entries, "shapes"),
        }
        cls._validate_partial_joint_ownership(model, owners["bodies"])

        unassigned = {name: sum(owner is None for owner in values) for name, values in owners.items()}
        logger.info(
            "[COUPLER] Unassigned model elements: %d bodies, %d particles, %d joints, and %d shapes. "
            "Unassigned elements remain outside nested solver views.",
            unassigned["bodies"],
            unassigned["particles"],
            unassigned["joints"],
            unassigned["shapes"],
        )

        if isinstance(solver_cfg, CouplerProxyCfg):
            cls._validate_no_cross_entry_proxy_joints(model, {entry.config.name: entry for entry in entries})

    @staticmethod
    def _build_ownership_map(count: int, entries: list[_ResolvedEntry], field: str) -> list[str | None]:
        """Build one ownership map while validating ranges and overlap."""
        owners: list[str | None] = [None] * int(count)
        for entry in entries:
            for raw_index in getattr(entry, field):
                index = int(raw_index)
                if not 0 <= index < count:
                    raise ValueError(
                        f"CouplerEntryCfg {entry.config.name!r} {field} index {index} is out of range [0, {count})."
                    )
                if owners[index] is not None:
                    raise ValueError(
                        f"CouplerCfg {field} index {index} is owned by both {owners[index]!r} "
                        f"and {entry.config.name!r}."
                    )
                owners[index] = entry.config.name
        return owners

    @staticmethod
    def _validate_partial_joint_ownership(model: Model, body_owners: list[str | None]) -> None:
        """Reject articulation joints with exactly one endpoint assigned to an entry."""
        for joint, (parent_raw, child_raw) in enumerate(zip(model.joint_parent.numpy(), model.joint_child.numpy())):
            parent = int(parent_raw)
            child = int(child_raw)
            if parent < 0:
                continue
            parent_owner = body_owners[parent]
            child_owner = body_owners[child]
            if (parent_owner is None) != (child_owner is None):
                raise ValueError(
                    f"CouplerCfg joint {joint} has only one owned endpoint: parent body {parent} is owned by "
                    f"{parent_owner!r}, child body {child} is owned by {child_owner!r}. Assign both articulation "
                    "bodies or leave both unassigned."
                )

    @classmethod
    def _register_builder_attributes(cls, builder: ModelBuilder) -> None:
        """Register custom attributes required by nested coupled entries."""
        super()._register_builder_attributes(builder)
        solver_cfg = getattr(PhysicsManager._cfg, "solver_cfg", None)
        if any(isinstance(entry.solver_cfg, MPMSolverCfg) for entry in getattr(solver_cfg, "entries", ())):
            NewtonMPMManager._register_builder_attributes(builder)

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: ModelBuilder) -> None:
        """Normalize kinematic colliders when a coupled entry uses implicit MPM."""
        super()._prepare_builder_for_finalize(builder)
        solver_cfg = getattr(PhysicsManager._cfg, "solver_cfg", None)
        if any(isinstance(entry.solver_cfg, MPMSolverCfg) for entry in getattr(solver_cfg, "entries", ())):
            NewtonMPMManager._prepare_builder_for_finalize(builder)

    @classmethod
    def _initialize_contacts(cls) -> None:
        """Initialize contacts and entry-local buffers before CUDA graph capture."""
        super()._initialize_contacts()
        if cls._contacts is not None and hasattr(NewtonManager._solver, "prepare_contacts"):
            NewtonManager._solver.prepare_contacts(cls._contacts)

    @classmethod
    def _supports_cuda_graph_capture(cls) -> bool:
        """Reject graph capture when a nested MPM entry uses a dynamic grid."""
        solver_cfg = getattr(PhysicsManager._cfg, "solver_cfg", None)
        return all(
            not isinstance(entry.solver_cfg, MPMSolverCfg) or entry.solver_cfg.grid_type == "fixed"
            for entry in getattr(solver_cfg, "entries", ())
        )

    @classmethod
    def _resolve_entry(
        cls,
        model: Model,
        entry_cfg: CouplerEntryCfg,
        scene_cfg: InteractiveSceneCfg | None,
    ) -> _ResolvedEntry:
        """Resolve one entry's selectors and derived ownership."""
        bodies = cls._resolve_entities_to_body_ids(model, entry_cfg.bodies, scene_cfg, f"entry {entry_cfg.name!r}")

        particles = list(dict.fromkeys(map(int, entry_cfg.particles)))
        if entry_cfg.all_particles:
            particles = list(dict.fromkeys([*particles, *range(int(model.particle_count))]))

        joints: list[int] = []
        if entry_cfg.include_child_joints and int(model.joint_count):
            body_set = set(bodies)
            parents = model.joint_parent.numpy()
            joints = [
                joint
                for joint, child in enumerate(model.joint_child.numpy())
                if int(child) in body_set and (int(parents[joint]) < 0 or int(parents[joint]) in body_set)
            ]

        shapes: list[int] = []
        if entry_cfg.include_body_shapes or entry_cfg.include_static_shapes:
            body_set = set(bodies)
            for shape, body_raw in enumerate(model.shape_body.numpy()):
                body = int(body_raw)
                if (entry_cfg.include_body_shapes and body in body_set) or (
                    entry_cfg.include_static_shapes and body < 0
                ):
                    shapes.append(shape)
        if entry_cfg.shape_label_patterns:
            labels = list(getattr(model, "shape_label", ()) or ())
            labeled_shapes = [(index, label) for index, label in enumerate(labels) if label is not None]
            try:
                matched_shapes, _ = resolve_matching_names(
                    entry_cfg.shape_label_patterns, [label for _, label in labeled_shapes]
                )
            except ValueError as error:
                raise ValueError(
                    f"CouplerEntryCfg {entry_cfg.name!r}: failed to resolve shape-label patterns."
                ) from error
            shapes.extend(labeled_shapes[index][0] for index in matched_shapes)

        return cls._ResolvedEntry(
            config=entry_cfg,
            bodies=bodies,
            particles=particles,
            joints=list(dict.fromkeys(joints)),
            shapes=list(dict.fromkeys(shapes)),
        )

    @classmethod
    def _resolve_proxy(
        cls,
        model: Model,
        proxy_cfg: CouplerProxyMappingCfg,
        scene_cfg: InteractiveSceneCfg | None,
    ) -> CouplerProxyMappingCfg:
        """Resolve a proxy mapping's selectors to collidable body ids, writing them into the config in place."""
        selected = cls._resolve_entities_to_body_ids(
            model, proxy_cfg.bodies, scene_cfg, f"proxy {proxy_cfg.source!r}->{proxy_cfg.destination!r}"
        )
        collide_flag = int(ShapeFlags.COLLIDE_SHAPES)
        collide_bodies = {
            int(body)
            for body, flags in zip(model.shape_body.numpy(), model.shape_flags.numpy())
            if int(body) >= 0 and int(flags) & collide_flag
        }
        bodies = [body for body in selected if body in collide_bodies]
        if proxy_cfg.bodies and not bodies:
            raise ValueError(
                f"CouplerProxyMappingCfg {proxy_cfg.source!r}->{proxy_cfg.destination!r} selected no bodies "
                "with ShapeFlags.COLLIDE_SHAPES."
            )
        proxy_cfg.bodies = bodies
        proxy_cfg.particles = list(dict.fromkeys(map(int, proxy_cfg.particles)))
        return proxy_cfg

    @classmethod
    def _resolve_entities_to_body_ids(
        cls,
        model: Model,
        specs: list[SceneEntityCfg | str | int],
        scene_cfg: InteractiveSceneCfg | None,
        field: str,
    ) -> list[int]:
        """Resolve scene entities, body-label regexes, or raw body ids to unique, order-preserving body ids."""
        labels = list(model.body_label)
        body_ids: list[int] = []
        for spec in specs:
            if isinstance(spec, int):
                if not 0 <= spec < len(labels):
                    raise ValueError(f"CouplerCfg {field}: body id {spec} is out of range [0, {len(labels)}).")
                body_ids.append(spec)
                continue
            if isinstance(spec, str):
                matched, _ = resolve_matching_names(f"(?:{spec})(?:/.*)?", labels, raise_when_no_match=False)
                if not matched:
                    raise ValueError(f"CouplerCfg {field}: body-label regex {spec!r} matched no Newton bodies.")
                body_ids.extend(matched)
                continue

            if not isinstance(spec, SceneEntityCfg):
                raise TypeError(
                    f"CouplerCfg {field}: expected a SceneEntityCfg, full-label regex string, or raw body id; "
                    f"got {type(spec).__name__}."
                )
            if scene_cfg is None:
                raise ValueError(
                    f"CouplerCfg {field}: scene_cfg is unset; assign the environment scene cfg before using "
                    f"SceneEntityCfg({spec.name!r}) selectors."
                )
            unsupported_fields = cls._unsupported_scene_entity_fields(spec)
            if unsupported_fields:
                raise ValueError(
                    f"CouplerCfg {field}: SceneEntityCfg({spec.name!r}) sets unsupported fields "
                    f"{unsupported_fields}; coupler selectors use only name, body_names, and preserve_order."
                )

            asset_cfg = getattr(scene_cfg, spec.name, None)
            if asset_cfg is None or not hasattr(asset_cfg, "prim_path"):
                raise ValueError(f"CouplerCfg {field}: scene entity {spec.name!r} is not on the attached scene cfg.")
            asset_body_ids, _ = resolve_matching_names(
                f"(?:{asset_cfg.prim_path})(?:/.*)?", labels, raise_when_no_match=False
            )
            if not asset_body_ids:
                raise ValueError(f"CouplerCfg {field}: scene entity {spec.name!r} matched no Newton bodies.")
            if spec.body_names is None:
                body_ids.extend(asset_body_ids)
                continue

            body_patterns = [spec.body_names] if isinstance(spec.body_names, str) else spec.body_names
            short_names = [labels[index].rsplit("/", 1)[-1] for index in asset_body_ids]
            try:
                local_body_ids, _ = resolve_matching_names(
                    body_patterns, short_names, preserve_order=spec.preserve_order
                )
            except ValueError as error:
                raise ValueError(
                    f"CouplerCfg {field}: scene entity {spec.name!r} could not match body patterns {body_patterns}."
                ) from error
            body_ids.extend(asset_body_ids[index] for index in local_body_ids)

        return list(dict.fromkeys(body_ids))

    @staticmethod
    def _unsupported_scene_entity_fields(spec: SceneEntityCfg) -> list[str]:
        """Return selector fields whose asset-local semantics cannot map safely to Newton labels."""
        unsupported = []
        defaults = {
            "joint_names": None,
            "joint_ids": slice(None),
            "fixed_tendon_names": None,
            "fixed_tendon_ids": slice(None),
            "body_ids": slice(None),
            "object_collection_names": None,
            "object_collection_ids": slice(None),
        }
        for name, default in defaults.items():
            if getattr(spec, name) != default:
                unsupported.append(name)
        return unsupported

    @classmethod
    def _build_entry(cls, entry: _ResolvedEntry) -> SolverCoupled.Entry:
        entry_cfg = entry.config

        def solver_factory(model_view):
            return entry_cfg.solver_cfg.class_type._create_solver(model_view, entry_cfg.solver_cfg)

        return SolverCoupled.Entry(
            name=entry_cfg.name,
            solver=solver_factory,
            bodies=entry.bodies,
            particles=entry.particles,
            joints=entry.joints,
            shapes=entry.shapes,
            substeps=entry_cfg.substeps,
            in_place=entry_cfg.in_place,
        )

    @classmethod
    def _build_proxy_coupled_solver(
        cls,
        model: Model,
        entries: list[SolverCoupled.Entry],
        proxy_cfgs: list[CouplerProxyMappingCfg],
        solver_cfg: CouplerProxyCfg,
    ) -> SolverCoupledProxy:
        proxies = []
        checked_factories: list[tuple[str, str, object, object]] = []
        for proxy_cfg in proxy_cfgs:
            values = cls._checked_config_values(SolverCoupledProxy.Proxy, proxy_cfg)
            factory = values.get("collision_pipeline")
            if factory is not None:
                checked_factory = next(
                    (
                        checked
                        for source, destination, original, checked in checked_factories
                        if source == proxy_cfg.source and destination == proxy_cfg.destination and original is factory
                    ),
                    None,
                )
                if checked_factory is None:
                    checked_factory = cls._checked_collision_pipeline_factory(
                        factory, proxy_cfg.source, proxy_cfg.destination
                    )
                    checked_factories.append((proxy_cfg.source, proxy_cfg.destination, factory, checked_factory))
                values["collision_pipeline"] = checked_factory
            proxies.append(SolverCoupledProxy.Proxy(**values))

        coupling_values = cls._checked_config_values(
            SolverCoupledProxy.Config,
            solver_cfg,
            handled_fields={"class_type", "solver_type", "model_cfg", "entries", "scene_cfg", "proxies"},
        )
        coupling_values["proxies"] = proxies
        coupling = SolverCoupledProxy.Config(**coupling_values)
        return SolverCoupledProxy(model=model, entries=entries, coupling=coupling)

    @classmethod
    def _build_admm_coupled_solver(
        cls,
        model: Model,
        entries: list[SolverCoupled.Entry],
        solver_cfg: CouplerAdmmCfg,
    ) -> SolverCoupledADMM:
        values = cls._checked_config_values(
            SolverCoupledADMM.Config,
            solver_cfg,
            handled_fields={
                "class_type",
                "solver_type",
                "model_cfg",
                "entries",
                "scene_cfg",
                "contact_pairs",
            },
        )
        if solver_cfg.contact_pairs is None:
            values["contact_pairs"] = SolverCoupledADMM.auto_detect_contact_pairs(entries)
        else:
            values["contact_pairs"] = [
                SolverCoupledADMM.ContactPair(source=source, destination=destination)
                for source, destination in solver_cfg.contact_pairs
            ]
        coupling = SolverCoupledADMM.Config(**values)
        return SolverCoupledADMM(model=model, entries=entries, coupling=coupling)

    @staticmethod
    def _checked_config_values(target_type: type, config, *, handled_fields: set[str] | None = None) -> dict:
        """Forward every config field or fail loudly when the Newton schema drifts."""
        handled_fields = handled_fields or set()
        target_fields = {field.name for field in fields(target_type)}
        config_fields = {field.name for field in fields(config)}
        unhandled = config_fields - target_fields - handled_fields
        if unhandled:
            raise TypeError(
                f"Cannot forward {type(config).__name__} to {target_type.__qualname__}; "
                f"unhandled fields: {sorted(unhandled)}."
            )
        return {name: getattr(config, name) for name in config_fields & target_fields if name not in handled_fields}

    @staticmethod
    def _checked_collision_pipeline_factory(factory, source: str, destination: str):
        """Wrap a proxy-local collision factory so a silent outer-contact fallback is impossible."""

        @wraps(factory)
        def checked_factory(model_view):
            pipeline = factory(model_view)
            if pipeline is None:
                raise TypeError(
                    "CouplerProxyMappingCfg collision_pipeline factory for "
                    f"{source!r}->{destination!r} returned None. Set collision_pipeline=None explicitly "
                    "to use the shared outer contacts."
                )
            return pipeline

        return checked_factory

    @staticmethod
    def _validate_no_cross_entry_proxy_joints(model: Model, entries: dict[str, _ResolvedEntry]) -> None:
        body_owner = {int(body): name for name, entry in entries.items() for body in entry.bodies}
        for joint, (parent_raw, child_raw) in enumerate(zip(model.joint_parent.numpy(), model.joint_child.numpy())):
            parent = int(parent_raw)
            child = int(child_raw)
            parent_owner = body_owner.get(parent)
            child_owner = body_owner.get(child)
            if parent >= 0 and parent_owner is not None and child_owner is not None and parent_owner != child_owner:
                raise ValueError(
                    f"CouplerProxyCfg does not support cross-entry joint {joint} between "
                    f"{parent_owner!r} and {child_owner!r}; keep the articulation in one entry "
                    "or use ADMM coupling."
                )
