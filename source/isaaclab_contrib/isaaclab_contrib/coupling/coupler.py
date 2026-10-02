# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton coupler for named solver configurations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import warp as wp
from isaaclab_newton.physics import (
    KaminoDVISolverCfg,
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCollisionPipelineCfg,
    NewtonSolverCfg,
)
from isaaclab_newton.physics.mpm_manager import implicit_mpm_solvers, mpm_supports_graph_capture
from isaaclab_newton.physics.newton_manager import NewtonManager
from isaaclab_newton.physics.solver_binding import NewtonSolverBinding
from newton import CollisionPipeline, Contacts, Model, ModelBuilder, ShapeFlags, State
from newton.solvers import SolverBase
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM, SolverCoupledProxy

from isaaclab.physics import PhysicsManager
from isaaclab.utils.string import resolve_matching_names

from .coupler_cfg import (
    CouplerAdmmCfg,
    CouplerCfg,
    CouplerEntryCfg,
    CouplerProxyCfg,
    CouplerProxyMappingCfg,
)


class CouplerSolverBinding(NewtonSolverBinding):
    """Couple named Newton solver entries through proxy or ADMM interfaces."""

    @dataclass
    class _ResolvedEntry:
        """Entry configuration with model selectors resolved to indices."""

        config: CouplerEntryCfg
        bodies: list[int]
        particles: list[int]
        joints: list[int]
        shapes: list[int]

    def __init__(self, model: Model, solver_cfg: CouplerCfg, deterministic_mode: wp.DeterministicMode):
        self._needs_collision_pipeline = True
        super().__init__(model, solver_cfg, deterministic_mode)
        self.needs_collision_pipeline = self._needs_collision_pipeline
        self.supports_contact_sensors = False
        self.implicit_mpm_solvers = implicit_mpm_solvers(self.solver)
        self._has_mpm_entry = any(isinstance(entry.solver_cfg, MPMSolverCfg) for entry in solver_cfg.entries)

    @classmethod
    def create(
        cls,
        model: Model,
        solver_cfg: CouplerCfg,
        deterministic_mode: wp.DeterministicMode = wp.DeterministicMode.NOT_GUARANTEED,
    ) -> SolverBase:
        """Reject recursive use as a nested coupled-solver entry."""
        del model, solver_cfg, deterministic_mode
        raise NotImplementedError("Nested Newton couplers are not supported.")

    @staticmethod
    def _requires_external_contacts(solver_cfg: NewtonSolverCfg) -> bool:
        """Return whether a sub-solver expects contacts from Newton's collision pipeline.

        Unknown solver configs conservatively opt in to external contacts.
        """
        if isinstance(solver_cfg, MJWarpSolverCfg):
            return not solver_cfg.use_mujoco_contacts
        if isinstance(solver_cfg, (KaminoPADMMSolverCfg, KaminoDVISolverCfg)):
            return not solver_cfg.use_collision_detector
        if isinstance(solver_cfg, MPMSolverCfg):
            return False
        return True

    def construct(self) -> SolverBase:
        """Resolve ownership and construct the selected coupled solver."""
        model, solver_cfg = self.model, self.cfg
        self._validate_config(solver_cfg)
        resolved_entries = [self._resolve_entry(model, entry) for entry in solver_cfg.entries]
        proxies: list[CouplerProxyMappingCfg] = []
        active_proxy_destinations: set[str] = set()
        if isinstance(solver_cfg, CouplerProxyCfg):
            proxies = [self._resolve_proxy(model, proxy) for proxy in solver_cfg.proxies]
            active_proxy_destinations = {proxy.destination for proxy in proxies if proxy.bodies or proxy.particles}
        self._validate_resolved_entries(model, resolved_entries, solver_cfg, active_proxy_destinations)
        entries = [self._build_entry(entry) for entry in resolved_entries]

        if not isinstance(solver_cfg, CouplerProxyCfg):
            return self._build_admm_coupled_solver(model, entries, solver_cfg)
        solver = self._build_proxy_coupled_solver(model, entries, proxies, solver_cfg)
        directions = {(proxy.source, proxy.destination) for proxy in proxies}
        proxy_destinations = {destination for _, destination in directions}
        outer_contact_entries = {
            entry.config.name for entry in resolved_entries if entry.config.name not in proxy_destinations
        }
        outer_contact_entries.update(source for source, _ in directions)
        outer_contact_entries.update(
            destination for source, destination in directions if solver.get_proxy_contacts(source, destination) is None
        )
        self._needs_collision_pipeline = any(
            entry.config.name in outer_contact_entries and self._requires_external_contacts(entry.config.solver_cfg)
            for entry in resolved_entries
        )
        return solver

    @classmethod
    def _validate_config(cls, solver_cfg: CouplerCfg) -> None:
        """Validate adapter-specific nested-solver constraints before construction."""
        if not isinstance(solver_cfg, (CouplerProxyCfg, CouplerAdmmCfg)):
            raise TypeError(
                f"CouplerCfg subclass {type(solver_cfg).__name__!r} is not supported; "
                "use CouplerProxyCfg or CouplerAdmmCfg."
            )
        if not solver_cfg.entries:
            raise ValueError("CouplerCfg.entries must contain at least one solver entry.")

        if isinstance(solver_cfg, CouplerAdmmCfg):
            capacity = solver_cfg.contact_max_triangle_pairs
            if capacity is not None and (type(capacity) is not int or capacity <= 0):
                raise ValueError("CouplerAdmmCfg.contact_max_triangle_pairs must be a positive integer or None.")
            if capacity is not None and capacity >= 2**20 and solver_cfg.rigid_contact_matching in ("latest", "sticky"):
                raise ValueError(
                    "CouplerAdmmCfg.contact_max_triangle_pairs must be less than 2**20 when "
                    "rigid_contact_matching is 'latest' or 'sticky'."
                )
            factor = solver_cfg.contact_reduction_hashtable_size_factor
            if factor is not None and (not math.isfinite(factor) or factor <= 0.0):
                raise ValueError(
                    "CouplerAdmmCfg.contact_reduction_hashtable_size_factor must be finite and positive or None."
                )

        if any(not isinstance(entry.name, str) or not entry.name for entry in solver_cfg.entries):
            raise ValueError("CouplerCfg entry names must be non-empty strings.")

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
            binding = getattr(nested_cfg.class_type, "solver_binding", None)
            if binding is None or binding.create.__func__ is NewtonSolverBinding.create.__func__:
                raise TypeError(
                    f"CouplerEntryCfg {entry.name!r} uses {type(nested_cfg).__name__}, whose manager "
                    "does not implement nested solver construction."
                )
            if isinstance(nested_cfg, (KaminoPADMMSolverCfg, KaminoDVISolverCfg)):
                raise NotImplementedError(
                    f"CouplerEntryCfg {entry.name!r} uses a Kamino solver config, whose manager-specific FK/reset "
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

    @classmethod
    def _validate_resolved_entries(
        cls,
        model: Model,
        entries: list[_ResolvedEntry],
        solver_cfg: CouplerCfg,
        active_proxy_destinations: set[str] | None = None,
    ) -> None:
        """Reject entries that neither own nor receive model elements."""
        active_proxy_destinations = active_proxy_destinations or set()
        for entry in entries:
            owns_any = any((entry.bodies, entry.particles, entry.joints, entry.shapes))
            if not owns_any and entry.config.name not in active_proxy_destinations:
                raise ValueError(f"CouplerEntryCfg {entry.config.name!r} neither owns nor receives any model elements.")

        if isinstance(solver_cfg, CouplerProxyCfg):
            cls._validate_no_cross_entry_proxy_joints(model, {entry.config.name: entry for entry in entries})

    @staticmethod
    def _entry_bindings() -> list[type[NewtonSolverBinding]]:
        """Return the solver binding of each configured entry, without duplicates."""
        entries = PhysicsManager._cfg.solver_cfg.entries
        return list(dict.fromkeys(entry.solver_cfg.class_type.solver_binding for entry in entries))

    @classmethod
    def register_builder_attributes(cls, builder: ModelBuilder) -> None:
        """Register custom attributes required by nested coupled entries."""
        for binding in cls._entry_bindings():
            binding.register_builder_attributes(builder)

    @classmethod
    def registers_builder_attributes_from(cls, solver_cls: type[SolverBase]) -> bool:
        """Return whether a configured entry registers ``solver_cls`` attributes."""
        return any(binding.registers_builder_attributes_from(solver_cls) for binding in cls._entry_bindings())

    @classmethod
    def prepare_builder(cls, builder: ModelBuilder) -> None:
        """Prepare the shared builder once per selected solver binding."""
        for binding in cls._entry_bindings():
            binding.prepare_builder(builder)

    def prepare_contacts(self, contacts: Contacts, collision_pipeline: CollisionPipeline | None) -> None:
        """Initialize entry-local contact buffers before CUDA graph capture."""
        if hasattr(self.solver, "prepare_contacts"):
            self.solver.prepare_contacts(contacts)

    @property
    def supports_graph_capture(self) -> bool:
        """Reject capture when a nested MPM solver has dynamic storage."""
        return all(mpm_supports_graph_capture(solver) for solver in self.implicit_mpm_solvers)

    def check_status(self, captured: bool) -> None:
        """Raise asynchronous failures from nested implicit-MPM solvers."""
        if captured:
            for solver in self.implicit_mpm_solvers:
                solver.check_sparse_grid_rebuild_status()

    def reset(self, state: State, world_mask: wp.array) -> None:
        """Promote a selected single MPM world to the solver's full-reset path."""
        if self.model.world_count == 1 and self._has_mpm_entry:
            selected = world_mask.numpy()
            if not selected.any():
                return
            if selected[0] and not selected[-1]:
                self.solver.reset(state, world_mask=None, flags=0)
                return
        super().reset(state, world_mask)

    @classmethod
    def _resolve_entry(
        cls,
        model: Model,
        entry_cfg: CouplerEntryCfg,
    ) -> _ResolvedEntry:
        """Resolve one entry's selectors and derived ownership."""
        bodies = cls._resolve_entities_to_body_ids(model, entry_cfg.bodies, f"entry {entry_cfg.name!r}")

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
    ) -> CouplerProxyMappingCfg:
        """Resolve a proxy mapping's selectors to collidable body ids, writing them into the config in place."""
        selected = cls._resolve_entities_to_body_ids(
            model, proxy_cfg.bodies, f"proxy {proxy_cfg.source!r}->{proxy_cfg.destination!r}"
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
        specs: list[str | int],
        field: str,
    ) -> list[int]:
        """Resolve body-label regexes or raw body ids to unique, order-preserving body ids."""
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

            raise TypeError(
                f"CouplerCfg {field}: expected a full-label regex string or raw body id; got {type(spec).__name__}."
            )

        return list(dict.fromkeys(body_ids))

    @classmethod
    def _build_entry(cls, entry: _ResolvedEntry) -> SolverCoupled.Entry:
        entry_cfg = entry.config

        def solver_factory(model_view):
            return entry_cfg.solver_cfg.class_type.solver_binding.create(model_view, entry_cfg.solver_cfg)

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

    def _build_proxy_coupled_solver(
        self,
        model: Model,
        entries: list[SolverCoupled.Entry],
        proxy_cfgs: list[CouplerProxyMappingCfg],
        solver_cfg: CouplerProxyCfg,
    ) -> SolverCoupledProxy:
        proxies = []
        for proxy_cfg in proxy_cfgs:
            values = vars(proxy_cfg).copy()
            if isinstance(values["collision_pipeline"], NewtonCollisionPipelineCfg):
                collision_cfg = values["collision_pipeline"]
                values["collision_pipeline"] = partial(CollisionPipeline, **collision_cfg.to_pipeline_args())
            proxies.append(SolverCoupledProxy.Proxy(**values))
        coupling_values = self.filter_kwargs(SolverCoupledProxy.Config, solver_cfg, self.deterministic_mode)
        coupling_values["proxies"] = proxies
        coupling = SolverCoupledProxy.Config(**coupling_values)
        return SolverCoupledProxy(model=model, entries=entries, coupling=coupling)

    def _build_admm_coupled_solver(
        self,
        model: Model,
        entries: list[SolverCoupled.Entry],
        solver_cfg: CouplerAdmmCfg,
    ) -> SolverCoupledADMM:
        values = self.filter_kwargs(SolverCoupledADMM.Config, solver_cfg, self.deterministic_mode)
        for name in ("contact_max_triangle_pairs", "contact_reduction_hashtable_size_factor"):
            if getattr(solver_cfg, name) is not None and name not in values:
                raise RuntimeError(f"The installed Newton version does not support {name}.")
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


class NewtonCouplerManager(NewtonManager):
    """:class:`NewtonManager` running named Newton solver entries coupled through proxy or ADMM interfaces."""

    solver_binding = CouplerSolverBinding
