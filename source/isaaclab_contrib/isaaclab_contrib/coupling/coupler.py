# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton coupler for named solver configurations.

:class:`CouplerSolver` builds each entry's solver from the entry's configuration on its model view and couples
the entries with Newton's experimental coupled solvers; each entry's solver class answers for it (contacts, capture,
status).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import partial

import warp as wp
from isaaclab_newton.physics import MPMSolverCfg, NewtonCollisionPipelineCfg, NewtonSolver, NewtonSolverCfg
from isaaclab_newton.physics.newton_solver import solver_kwargs
from newton import CollisionPipeline, Model, ModelBuilder, ShapeFlags, State
from newton.solvers import SolverBase
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM, SolverCoupledProxy

from isaaclab.cloner.cloner_cfg import DEFAULT_ENV_TEMPLATE, expand_env_regex_ns
from isaaclab.utils.string import resolve_matching_names

from .coupler_cfg import (
    CouplerAdmmCfg,
    CouplerCfg,
    CouplerEntryCfg,
    CouplerProxyCfg,
    CouplerProxyMappingCfg,
)


@dataclass
class _ResolvedEntry:
    """Entry configuration with model selectors resolved to indices."""

    config: CouplerEntryCfg
    bodies: list[int]
    particles: list[int]
    joints: list[int]
    shapes: list[int]


# ----- Construction --------------------------------------------------------------------


def _validate_cfg(cfg: CouplerCfg) -> None:
    """Validate the coupler configuration and ask each entry's configuration whether it can run coupled."""
    if not isinstance(cfg, (CouplerProxyCfg, CouplerAdmmCfg)):
        raise TypeError(
            f"CouplerCfg subclass {type(cfg).__name__!r} is not supported; use CouplerProxyCfg or CouplerAdmmCfg."
        )
    if not cfg.entries:
        raise ValueError("CouplerCfg.entries must contain at least one solver entry.")

    if isinstance(cfg, CouplerAdmmCfg):
        capacity = cfg.contact_max_triangle_pairs
        if capacity is not None and (type(capacity) is not int or capacity <= 0):
            raise ValueError("CouplerAdmmCfg.contact_max_triangle_pairs must be a positive integer or None.")
        if capacity is not None and capacity >= 2**20 and cfg.rigid_contact_matching in ("latest", "sticky"):
            raise ValueError(
                "CouplerAdmmCfg.contact_max_triangle_pairs must be less than 2**20 when "
                "rigid_contact_matching is 'latest' or 'sticky'."
            )
        factor = cfg.contact_reduction_hashtable_size_factor
        if factor is not None and (not math.isfinite(factor) or factor <= 0.0):
            raise ValueError(
                "CouplerAdmmCfg.contact_reduction_hashtable_size_factor must be finite and positive or None."
            )

    if any(not isinstance(entry.name, str) or not entry.name for entry in cfg.entries):
        raise ValueError("CouplerCfg entry names must be non-empty strings.")
    names = {entry.name for entry in cfg.entries}
    if cfg.outer_contact_entry is not None and cfg.outer_contact_entry not in names:
        raise ValueError(f"CouplerCfg.outer_contact_entry {cfg.outer_contact_entry!r} names no entry.")

    for entry in cfg.entries:
        nested_cfg = entry.solver_cfg
        if not isinstance(nested_cfg, NewtonSolverCfg):
            raise TypeError(
                f"CouplerEntryCfg {entry.name!r} solver_cfg must be a NewtonSolverCfg, got {type(nested_cfg).__name__}."
            )
        if isinstance(nested_cfg, CouplerCfg):
            raise ValueError(
                f"CouplerEntryCfg {entry.name!r} contains a nested CouplerCfg; nested couplers are not supported."
            )
        error = nested_cfg.class_type.coupling_error(nested_cfg, entry.in_place)
        if error is not None:
            raise NotImplementedError(f"CouplerEntryCfg {entry.name!r} {error}.")


def _build_proxy_coupled_solver(
    model: Model,
    entries: list[SolverCoupled.Entry],
    proxy_cfgs: list[CouplerProxyMappingCfg],
    cfg: CouplerProxyCfg,
    deterministic_mode: wp.DeterministicMode,
) -> SolverCoupledProxy:
    proxies = []
    for proxy_cfg in proxy_cfgs:
        values = vars(proxy_cfg).copy()
        if isinstance(values["collision_pipeline"], NewtonCollisionPipelineCfg):
            collision_cfg = values["collision_pipeline"]
            values["collision_pipeline"] = partial(CollisionPipeline, **collision_cfg.to_pipeline_args())
        proxies.append(SolverCoupledProxy.Proxy(**values))
    coupling_values = solver_kwargs(SolverCoupledProxy.Config, cfg, deterministic_mode)
    coupling_values["proxies"] = proxies
    coupling = SolverCoupledProxy.Config(**coupling_values)
    return SolverCoupledProxy(model=model, entries=entries, coupling=coupling)


def _build_admm_coupled_solver(
    model: Model,
    entries: list[SolverCoupled.Entry],
    cfg: CouplerAdmmCfg,
    deterministic_mode: wp.DeterministicMode,
) -> SolverCoupledADMM:
    values = solver_kwargs(SolverCoupledADMM.Config, cfg, deterministic_mode)
    for name in ("contact_max_triangle_pairs", "contact_reduction_hashtable_size_factor"):
        if getattr(cfg, name) is not None and name not in values:
            raise RuntimeError(f"The installed Newton version does not support {name}.")
    if cfg.contact_pairs is None:
        values["contact_pairs"] = SolverCoupledADMM.auto_detect_contact_pairs(entries)
    else:
        values["contact_pairs"] = [
            SolverCoupledADMM.ContactPair(source=source, destination=destination)
            for source, destination in cfg.contact_pairs
        ]
    coupling = SolverCoupledADMM.Config(**values)
    return SolverCoupledADMM(model=model, entries=entries, coupling=coupling)


# ----- Ownership resolution ----------------------------------------------------------


def _resolve_entry(model: Model, entry_cfg: CouplerEntryCfg, env_template: str) -> _ResolvedEntry:
    """Resolve one entry's selectors and derived ownership."""
    bodies = _resolve_entities_to_body_ids(model, entry_cfg.bodies, f"entry {entry_cfg.name!r}", env_template)

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
            if (entry_cfg.include_body_shapes and body in body_set) or (entry_cfg.include_static_shapes and body < 0):
                shapes.append(shape)
    if entry_cfg.shape_label_patterns:
        labels = list(getattr(model, "shape_label", ()) or ())
        labeled_shapes = [(index, label) for index, label in enumerate(labels) if label is not None]
        try:
            matched_shapes, _ = resolve_matching_names(
                [expand_env_regex_ns(pattern, env_template) for pattern in entry_cfg.shape_label_patterns],
                [label for _, label in labeled_shapes],
            )
        except ValueError as error:
            raise ValueError(f"CouplerEntryCfg {entry_cfg.name!r}: failed to resolve shape-label patterns.") from error
        shapes.extend(labeled_shapes[index][0] for index in matched_shapes)

    return _ResolvedEntry(
        config=entry_cfg,
        bodies=bodies,
        particles=particles,
        joints=list(dict.fromkeys(joints)),
        shapes=list(dict.fromkeys(shapes)),
    )


def _resolve_proxy(model: Model, proxy_cfg: CouplerProxyMappingCfg, env_template: str) -> CouplerProxyMappingCfg:
    """Resolve a proxy mapping to collidable body ids without changing its declaration."""
    selected = _resolve_entities_to_body_ids(
        model, proxy_cfg.bodies, f"proxy {proxy_cfg.source!r}->{proxy_cfg.destination!r}", env_template
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
    return replace(proxy_cfg, bodies=bodies, particles=list(dict.fromkeys(map(int, proxy_cfg.particles))))


def _resolve_entities_to_body_ids(model: Model, specs: list[str | int], field: str, env_template: str) -> list[int]:
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
            pattern = expand_env_regex_ns(spec, env_template)
            matched, _ = resolve_matching_names(f"(?:{pattern})(?:/.*)?", labels, raise_when_no_match=False)
            if not matched:
                raise ValueError(f"CouplerCfg {field}: body-label regex {spec!r} matched no Newton bodies.")
            body_ids.extend(matched)
            continue

        raise TypeError(
            f"CouplerCfg {field}: expected a full-label regex string or raw body id; got {type(spec).__name__}."
        )

    return list(dict.fromkeys(body_ids))


def _validate_resolved_entries(
    model: Model,
    entries: list[_ResolvedEntry],
    cfg: CouplerCfg,
    active_proxy_destinations: set[str] | None = None,
) -> None:
    """Reject entries that neither own nor receive model elements."""
    active_proxy_destinations = active_proxy_destinations or set()
    for entry in entries:
        owns_any = any((entry.bodies, entry.particles, entry.joints, entry.shapes))
        if not owns_any and entry.config.name not in active_proxy_destinations:
            raise ValueError(f"CouplerEntryCfg {entry.config.name!r} neither owns nor receives any model elements.")

    if isinstance(cfg, CouplerProxyCfg):
        _validate_no_cross_entry_proxy_joints(model, {entry.config.name: entry for entry in entries})


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


# ----- Solver ----------------------------------------------------------------------------


class CouplerSolver(NewtonSolver):
    """Couple named entries, each built from its own solver configuration on its model view."""

    contact_sensors = False

    @classmethod
    def solver_classes(cls, cfg: CouplerCfg) -> tuple[type[SolverBase], ...]:
        return tuple(
            solver_cls
            for entry in cfg.entries
            for solver_cls in entry.solver_cfg.class_type.solver_classes(entry.solver_cfg)
        )

    @classmethod
    def prepare_builder(cls, cfg: CouplerCfg, builder: ModelBuilder) -> None:
        """Prepare the shared builder once per distinct entry solver class."""
        for entry_cfg in {entry.solver_cfg.class_type: entry.solver_cfg for entry in cfg.entries}.values():
            entry_cfg.class_type.prepare_builder(entry_cfg, builder)

    @classmethod
    def build(
        cls, cfg: CouplerCfg, model: Model, mode: wp.DeterministicMode, env_template: str | None = None
    ) -> SolverCoupled:
        """Resolve entry ownership, build each entry's solver on its model view, and couple the entries.

        ``{ENV_REGEX_NS}`` in entry and proxy body selectors and shape patterns expands against ``env_template``; the
        configuration stays unchanged.
        """
        _validate_cfg(cfg)
        template = DEFAULT_ENV_TEMPLATE if env_template is None else env_template
        resolved_entries = [_resolve_entry(model, entry, template) for entry in cfg.entries]
        proxies: list[CouplerProxyMappingCfg] = []
        active_proxy_destinations: set[str] = set()
        if isinstance(cfg, CouplerProxyCfg):
            proxies = [_resolve_proxy(model, proxy, template) for proxy in cfg.proxies]
            active_proxy_destinations = {proxy.destination for proxy in proxies if proxy.bodies or proxy.particles}
        _validate_resolved_entries(model, resolved_entries, cfg, active_proxy_destinations)
        entries = [_coupled_entry(entry, mode, env_template) for entry in resolved_entries]
        return cls.build_coupled(cfg, model, entries, proxies, mode)

    @classmethod
    def build_coupled(
        cls,
        cfg: CouplerCfg,
        model: Model,
        entries: list[SolverCoupled.Entry],
        proxies: list[CouplerProxyMappingCfg],
        mode: wp.DeterministicMode,
    ) -> SolverCoupled:
        """Construct Newton's coupled solver from the built entries; tasks with a custom coupled solver override it.

        A proxy configuration without proxies steps its entries independently: Newton's proxy solver advances only
        connected proxy groups.
        """
        if isinstance(cfg, CouplerAdmmCfg):
            return _build_admm_coupled_solver(model, entries, cfg, mode)
        if not proxies:
            return SolverCoupled(model=model, entries=entries)
        return _build_proxy_coupled_solver(model, entries, proxies, cfg, mode)

    @classmethod
    def supports_capture(cls, cfg: CouplerCfg, solver: SolverCoupled) -> bool:
        return all(
            entry.solver_cfg.class_type.supports_capture(entry.solver_cfg, solver.solver(entry.name))
            for entry in cfg.entries
        )

    @classmethod
    def uses_collision_pipeline(cls, cfg: CouplerCfg, solver: SolverCoupled) -> bool:
        """Whether an entry that collides outside its proxies expects contacts from Newton's collision pipeline."""
        if not isinstance(cfg, CouplerProxyCfg):
            return True
        directions = {(proxy.source, proxy.destination) for proxy in cfg.proxies}
        proxy_destinations = {destination for _, destination in directions}
        outer_contact_entries = {entry.name for entry in cfg.entries if entry.name not in proxy_destinations}
        outer_contact_entries.update(source for source, _ in directions)
        outer_contact_entries.update(
            destination for source, destination in directions if solver.get_proxy_contacts(source, destination) is None
        )
        return any(
            entry.name in outer_contact_entries
            and entry.solver_cfg.class_type.uses_collision_pipeline(entry.solver_cfg, solver.solver(entry.name))
            for entry in cfg.entries
        )

    @classmethod
    def create_collision_pipeline(cls, cfg: CouplerCfg, solver: SolverCoupled, **kwargs) -> CollisionPipeline:
        """Collide the whole model, or only :attr:`~CouplerCfg.outer_contact_entry`'s shape pairs."""
        if cfg.outer_contact_entry is not None:
            # Shape IDs remain global: the full model's body bindings collide the shared state.
            kwargs["shape_pairs_filtered"] = solver.view(cfg.outer_contact_entry).shape_contact_pairs
        return super().create_collision_pipeline(cfg, solver, **kwargs)

    @classmethod
    def reset(cls, cfg: CouplerCfg, solver: SolverCoupled, state: State, world_mask: wp.array) -> None:
        """Reset every entry for masked worlds, promoting a selected single MPM world to the full-reset path."""
        has_mpm_entry = any(isinstance(entry.solver_cfg, MPMSolverCfg) for entry in cfg.entries)
        if solver.model.world_count == 1 and has_mpm_entry:
            selected = world_mask.numpy()
            if not selected.any():
                return
            if selected[0] and not selected[-1]:
                solver.reset(state, world_mask=None, flags=0)
                return
        solver.reset(state, world_mask=world_mask, flags=0)

    @classmethod
    def check_step(cls, cfg: CouplerCfg, solver: SolverCoupled, captured: bool, debug: bool) -> None:
        """Raise asynchronous failures, and log diagnostics, of every entry's solver."""
        for entry in cfg.entries:
            entry.solver_cfg.class_type.check_step(entry.solver_cfg, solver.solver(entry.name), captured, debug)


def _coupled_entry(entry: _ResolvedEntry, mode: wp.DeterministicMode, env_template: str | None) -> SolverCoupled.Entry:
    """Describe a resolved entry to Newton; its solver is built on the entry's model view."""
    entry_cfg = entry.config
    solver_cfg = entry_cfg.solver_cfg
    return SolverCoupled.Entry(
        name=entry_cfg.name,
        solver=lambda view: solver_cfg.class_type.build(solver_cfg, view, mode, env_template),
        bodies=entry.bodies,
        particles=entry.particles,
        joints=entry.joints,
        shapes=entry.shapes,
        substeps=entry_cfg.substeps,
        in_place=entry_cfg.in_place,
    )
