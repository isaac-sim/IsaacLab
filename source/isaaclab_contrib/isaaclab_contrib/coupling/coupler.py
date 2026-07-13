# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton coupler for named solver configurations."""

from __future__ import annotations

from dataclasses import dataclass
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

    @staticmethod
    def _requires_external_contacts(solver_cfg: NewtonSolverCfg) -> bool:
        """Return whether a sub-solver expects contacts from Newton's collision pipeline."""
        if isinstance(solver_cfg, MJWarpSolverCfg):
            return not solver_cfg.use_mujoco_contacts
        if isinstance(solver_cfg, KaminoSolverCfg):
            return not solver_cfg.use_collision_detector
        return True

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CouplerCfg) -> None:
        """Resolve ownership and construct the selected coupled solver."""
        scene_cfg = solver_cfg.scene_cfg

        resolved_entries = [cls._resolve_entry(model, entry, scene_cfg) for entry in solver_cfg.entries]
        entries = [cls._build_entry(entry) for entry in resolved_entries]

        if isinstance(solver_cfg, CouplerProxyCfg):
            proxies = [cls._resolve_proxy(model, proxy, scene_cfg) for proxy in solver_cfg.proxies]
            cls._validate_no_cross_entry_proxy_joints(model, {entry.config.name: entry for entry in resolved_entries})
            NewtonManager._solver = cls._build_proxy_coupled_solver(model, entries, proxies, solver_cfg)
            proxy_destinations = {proxy.destination for proxy in proxies}
            needs_collision_pipeline = any(
                entry.config.name not in proxy_destinations and cls._requires_external_contacts(entry.config.solver_cfg)
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
        if NewtonManager._report_contacts:
            raise NotImplementedError(
                "Newton contact sensors are not yet supported by coupled solvers because contact forces live "
                "in per-entry buffers. Remove the contact sensor."
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

            asset_cfg = getattr(scene_cfg, spec.name, None) if scene_cfg is not None else None
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
                local_body_ids, _ = resolve_matching_names(body_patterns, short_names)
            except ValueError as error:
                raise ValueError(
                    f"CouplerCfg {field}: scene entity {spec.name!r} could not match body patterns {body_patterns}."
                ) from error
            body_ids.extend(asset_body_ids[index] for index in local_body_ids)

        return list(dict.fromkeys(body_ids))

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
        proxies = [SolverCoupledProxy.Proxy(**vars(proxy_cfg)) for proxy_cfg in proxy_cfgs]
        coupling = SolverCoupledProxy.Config(proxies=proxies, iterations=solver_cfg.iterations)
        return SolverCoupledProxy(model=model, entries=entries, coupling=coupling)

    @classmethod
    def _build_admm_coupled_solver(
        cls,
        model: Model,
        entries: list[SolverCoupled.Entry],
        solver_cfg: CouplerAdmmCfg,
    ) -> SolverCoupledADMM:
        values = cls._filter_solver_kwargs(SolverCoupledADMM.Config, solver_cfg)
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
