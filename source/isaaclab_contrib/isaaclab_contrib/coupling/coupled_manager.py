# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton manager for named coupled-solver configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from isaaclab_newton.physics import (
    FeatherstoneSolverCfg,
    KaminoSolverCfg,
    MJWarpSolverCfg,
    NewtonSolverCfg,
    XPBDSolverCfg,
)
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import CollisionPipeline, Model, ShapeFlags
from newton.solvers import SolverBase, SolverFeatherstone, SolverKamino, SolverMuJoCo, SolverVBD, SolverXPBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM, SolverCoupledProxy

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.string import resolve_matching_names

from ..deformable.newton_manager_cfg import VBDSolverCfg
from ..deformable.vbd_manager import NewtonVBDManager
from .coupled_manager_cfg import (
    CoupledAdmmSolverCfg,
    CoupledProxyCfg,
    CoupledProxySolverCfg,
    CoupledSolverCfg,
    CoupledSolverEntryCfg,
)

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveSceneCfg


class NewtonCoupledSolverManager(NewtonVBDManager):
    """Build and manage Newton proxy or ADMM coupling from named entries."""

    @dataclass
    class _ResolvedEntry:
        """Entry configuration with model selectors resolved to indices."""

        config: CoupledSolverEntryCfg
        bodies: list[int]
        particles: list[int]
        joints: list[int]
        shapes: list[int]

    @dataclass
    class _ResolvedProxy:
        """Proxy configuration with source selectors resolved to indices."""

        config: CoupledProxyCfg
        bodies: list[int]
        particles: list[int]

    _SOLVER_CLASS_BY_CFG_TYPE: ClassVar[dict[type[NewtonSolverCfg], type[SolverBase]]] = {
        MJWarpSolverCfg: SolverMuJoCo,
        VBDSolverCfg: SolverVBD,
        FeatherstoneSolverCfg: SolverFeatherstone,
        XPBDSolverCfg: SolverXPBD,
        KaminoSolverCfg: SolverKamino,
    }

    @classmethod
    def _resolve_solver_class(cls, sub_cfg: NewtonSolverCfg) -> type[SolverBase]:
        """Resolve a supported Isaac Lab solver config to its Newton solver class."""
        try:
            return cls._SOLVER_CLASS_BY_CFG_TYPE[type(sub_cfg)]
        except KeyError:
            known = ", ".join(sorted(t.__name__ for t in cls._SOLVER_CLASS_BY_CFG_TYPE))
            raise ValueError(
                f"No Newton solver registered for sub-cfg type {type(sub_cfg).__name__!r}. Known config types: {known}."
            ) from None

    @staticmethod
    def _requires_external_contacts(solver_cfg: NewtonSolverCfg) -> bool:
        """Return whether a sub-solver expects contacts from Newton's collision pipeline."""
        if isinstance(solver_cfg, MJWarpSolverCfg):
            return not solver_cfg.use_mujoco_contacts
        if isinstance(solver_cfg, KaminoSolverCfg):
            return not solver_cfg.use_collision_detector
        return True

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CoupledSolverCfg) -> None:
        """Resolve ownership and construct the selected coupled solver."""
        scene_cfg = solver_cfg.scene_cfg

        resolved_entries = [cls._resolve_entry(model, entry, scene_cfg) for entry in solver_cfg.entries]
        entries = [cls._build_entry(entry) for entry in resolved_entries]

        if isinstance(solver_cfg, CoupledProxySolverCfg):
            proxies = [cls._resolve_proxy(model, proxy, scene_cfg) for proxy in solver_cfg.proxies]
            cls._validate_solver_cfg(model, solver_cfg, resolved_entries, proxies)
            NewtonManager._solver = cls._build_proxy_coupled_solver(model, entries, proxies, solver_cfg)
            proxy_destinations = {proxy.config.destination for proxy in proxies}
            needs_collision_pipeline = any(
                entry.config.name not in proxy_destinations and cls._requires_external_contacts(entry.config.solver_cfg)
                for entry in resolved_entries
            )
        elif isinstance(solver_cfg, CoupledAdmmSolverCfg):
            cls._validate_solver_cfg(model, solver_cfg, resolved_entries)
            NewtonManager._solver = cls._build_admm_coupled_solver(model, entries, solver_cfg)
            needs_collision_pipeline = True
        else:
            raise TypeError(
                f"CoupledSolverCfg subclass {type(solver_cfg).__name__!r} is not supported; "
                "use CoupledProxySolverCfg or CoupledAdmmSolverCfg."
            )

        NewtonManager._use_single_state = False
        NewtonManager._supports_contact_sensors = False
        NewtonManager._needs_collision_pipeline = needs_collision_pipeline
        if NewtonManager._report_contacts:
            raise NotImplementedError(
                "Newton contact sensors are not yet supported by coupled solvers because contact forces live "
                "in per-entry buffers. Remove the contact sensor."
            )

    @classmethod
    def _resolve_entry(
        cls,
        model: Model,
        entry_cfg: CoupledSolverEntryCfg,
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
                    f"CoupledSolverEntryCfg {entry_cfg.name!r}: failed to resolve shape-label patterns."
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
        proxy_cfg: CoupledProxyCfg,
        scene_cfg: InteractiveSceneCfg | None,
    ) -> _ResolvedProxy:
        """Resolve one proxy mapping's body selectors to collidable body ids."""
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
                f"CoupledProxyCfg {proxy_cfg.source!r}->{proxy_cfg.destination!r} selected no bodies "
                "with ShapeFlags.COLLIDE_SHAPES."
            )
        return cls._ResolvedProxy(
            config=proxy_cfg,
            bodies=bodies,
            particles=list(dict.fromkeys(map(int, proxy_cfg.particles))),
        )

    @classmethod
    def _resolve_entities_to_body_ids(
        cls,
        model: Model,
        specs: list[SceneEntityCfg | str],
        scene_cfg: InteractiveSceneCfg | None,
        field: str,
    ) -> list[int]:
        """Resolve scene entities or body-label regexes to unique, order-preserving body ids."""
        labels = list(model.body_label)
        body_ids: list[int] = []
        for spec in specs:
            if isinstance(spec, str):
                matched, _ = resolve_matching_names(f"(?:{spec})(?:/.*)?", labels, raise_when_no_match=False)
                if not matched:
                    raise ValueError(f"CoupledSolverCfg {field}: body-label regex {spec!r} matched no Newton bodies.")
                body_ids.extend(matched)
                continue

            asset_cfg = getattr(scene_cfg, spec.name, None) if scene_cfg is not None else None
            if asset_cfg is None or not hasattr(asset_cfg, "prim_path"):
                raise ValueError(
                    f"CoupledSolverCfg {field}: scene entity {spec.name!r} is not on the attached scene cfg."
                )
            asset_body_ids, _ = resolve_matching_names(
                f"(?:{asset_cfg.prim_path})(?:/.*)?", labels, raise_when_no_match=False
            )
            if not asset_body_ids:
                raise ValueError(f"CoupledSolverCfg {field}: scene entity {spec.name!r} matched no Newton bodies.")
            if spec.body_names is None:
                body_ids.extend(asset_body_ids)
                continue

            body_patterns = [spec.body_names] if isinstance(spec.body_names, str) else spec.body_names
            short_names = [labels[index].rsplit("/", 1)[-1] for index in asset_body_ids]
            try:
                local_body_ids, _ = resolve_matching_names(body_patterns, short_names)
            except ValueError as error:
                raise ValueError(
                    f"CoupledSolverCfg {field}: scene entity {spec.name!r} could not match body patterns"
                    f" {body_patterns}."
                ) from error
            body_ids.extend(asset_body_ids[index] for index in local_body_ids)

        return list(dict.fromkeys(body_ids))

    @classmethod
    def _build_entry(cls, entry: _ResolvedEntry) -> SolverCoupled.Entry:
        entry_cfg = entry.config
        solver_cls = cls._resolve_solver_class(entry_cfg.solver_cfg)
        solver_kwargs = cls._filter_solver_kwargs(solver_cls, entry_cfg.solver_cfg)

        return SolverCoupled.Entry(
            name=entry_cfg.name,
            solver=lambda v: solver_cls(model=v, **solver_kwargs),
            bodies=entry.bodies,
            particles=entry.particles,
            joints=entry.joints,
            shapes=entry.shapes,
        )

    @classmethod
    def _build_proxy_coupled_solver(
        cls,
        model: Model,
        entries: list[SolverCoupled.Entry],
        proxies: list[SolverCoupledProxy.Proxy],
        solver_cfg: CoupledProxySolverCfg,
    ) -> SolverCoupledProxy:
        cls._apply_proxy_shape_overrides(model, proxies)
        proxy_mappings = [
            SolverCoupledProxy.Proxy(
                source=proxy.config.source,
                destination=proxy.config.destination,
                bodies=proxy.bodies,
                particles=proxy.particles,
                mode=proxy.config.mode,
                mass_scale=float(proxy.config.mass_scale),
                collision_pipeline=proxy.config.collision_pipeline_factory
                or (lambda model_view: CollisionPipeline(model_view, broad_phase="explicit")),
                collide_interval=proxy.config.collide_interval,
            )
            for proxy in proxies
        ]
        return SolverCoupledProxy(
            model=model,
            entries=entries,
            coupling=SolverCoupledProxy.Config(proxies=proxy_mappings, iterations=int(solver_cfg.iterations)),
        )

    @classmethod
    def _build_admm_coupled_solver(
        cls,
        model: Model,
        entries: list[SolverCoupled.Entry],
        solver_cfg: CoupledAdmmSolverCfg,
    ) -> SolverCoupledADMM:
        contact_pairs = (
            SolverCoupledADMM.auto_detect_contact_pairs(entries)
            if solver_cfg.contact_pairs is None
            else [
                SolverCoupledADMM.ContactPair(source=source, destination=destination)
                for source, destination in solver_cfg.contact_pairs
            ]
        )
        coupling = SolverCoupledADMM.Config(
            iterations=int(solver_cfg.iterations),
            rho=float(solver_cfg.rho),
            gamma=float(solver_cfg.gamma),
            baumgarte=float(solver_cfg.baumgarte),
            joint_stiffness=float(solver_cfg.joint_stiffness),
            joint_damping=float(solver_cfg.joint_damping),
            joint_angular_stiffness=float(solver_cfg.joint_angular_stiffness),
            joint_angular_damping=float(solver_cfg.joint_angular_damping),
            joint_proximal_bodies=bool(solver_cfg.joint_proximal_bodies),
            joint_proximal_destination_entries=solver_cfg.joint_proximal_destination_entries,
            joint_proximal_mass_scale=float(solver_cfg.joint_proximal_mass_scale),
            rigid_contact_matching=solver_cfg.rigid_contact_matching,
            contact_matching_pos_threshold=solver_cfg.contact_matching_pos_threshold,
            contact_matching_normal_dot_threshold=solver_cfg.contact_matching_normal_dot_threshold,
            contact_matching_force_scale=float(solver_cfg.contact_matching_force_scale),
            contact_pairs=contact_pairs,
        )
        return SolverCoupledADMM(model=model, entries=entries, coupling=coupling)

    @classmethod
    def _validate_solver_cfg(
        cls,
        model: Model,
        solver_cfg: CoupledSolverCfg,
        entries: list[_ResolvedEntry],
        proxies: list[_ResolvedProxy] | None = None,
    ) -> None:
        if len(entries) < 2:
            raise ValueError("A coupled solver requires at least two named entries.")
        names = [entry.config.name for entry in entries]
        if any(not name for name in names):
            raise ValueError("CoupledSolverEntryCfg.name must be non-empty.")
        if len(set(names)) != len(names):
            raise ValueError(f"Coupled solver entry names must be unique, got {names!r}.")

        cls._validate_ownership(model, entries, "bodies", int(model.body_count), require_complete=True)
        cls._validate_ownership(model, entries, "particles", int(model.particle_count), require_complete=True)
        cls._validate_ownership(model, entries, "joints", int(model.joint_count))
        cls._validate_ownership(model, entries, "shapes", int(model.shape_count))

        if isinstance(solver_cfg, CoupledProxySolverCfg):
            if len(entries) > 2:
                raise ValueError("Newton proxy coupling currently supports at most two solver entries.")
            if solver_cfg.iterations < 1:
                raise ValueError("CoupledProxySolverCfg.iterations must be >= 1.")
            if not proxies:
                raise ValueError("CoupledProxySolverCfg requires at least one proxy mapping.")
            entries_by_name = {entry.config.name: entry for entry in entries}
            cls._validate_no_cross_entry_proxy_joints(model, entries_by_name)
            for proxy in proxies:
                cls._validate_proxy(proxy, entries_by_name)
        elif isinstance(solver_cfg, CoupledAdmmSolverCfg):
            if solver_cfg.iterations < 1:
                raise ValueError("CoupledAdmmSolverCfg.iterations must be >= 1.")
            seen_pairs: set[frozenset[str]] = set()
            for source, destination in solver_cfg.contact_pairs or []:
                if source not in names or destination not in names:
                    raise ValueError(
                        f"ADMM contact-pair endpoints {source!r}->{destination!r} must name coupled entries."
                    )
                if source == destination:
                    raise ValueError("ADMM contact-pair source and destination must differ.")
                pair = frozenset((source, destination))
                if pair in seen_pairs:
                    raise ValueError(f"Duplicate symmetric ADMM contact pair {source!r}, {destination!r}.")
                seen_pairs.add(pair)

    @staticmethod
    def _validate_ownership(
        model: Model,
        entries: list[_ResolvedEntry],
        field: str,
        count: int,
        *,
        require_complete: bool = False,
    ) -> None:
        owners: dict[int, str] = {}
        for entry in entries:
            for index in getattr(entry, field, ()):
                if index < 0 or index >= count:
                    raise ValueError(f"Coupled entry {entry.config.name!r} owns out-of-range {field} index {index}.")
                if index in owners:
                    raise ValueError(
                        f"{field} index {index} is owned by both {owners[index]!r} and {entry.config.name!r}."
                    )
                owners[index] = entry.config.name
        if require_complete and (unclaimed := [index for index in range(count) if index not in owners]):
            labels = getattr(model, "body_label", None) if field == "bodies" else None
            preview = [labels[index] if labels is not None else index for index in unclaimed[:5]]
            raise ValueError(f"Coupled solver has {len(unclaimed)} unclaimed {field} (first few: {preview!r}).")

    @staticmethod
    def _validate_proxy(proxy: _ResolvedProxy, entries: dict[str, _ResolvedEntry]) -> None:
        proxy_cfg = proxy.config
        if proxy_cfg.source not in entries or proxy_cfg.destination not in entries:
            raise ValueError(
                f"CoupledProxyCfg endpoints {proxy_cfg.source!r}->{proxy_cfg.destination!r} must name coupled entries."
            )
        if proxy_cfg.source == proxy_cfg.destination:
            raise ValueError("CoupledProxyCfg source and destination must differ.")
        if not proxy.bodies and not proxy.particles:
            raise ValueError("CoupledProxyCfg must map at least one body or particle.")
        if not set(proxy.bodies).issubset(entries[proxy_cfg.source].bodies):
            raise ValueError("CoupledProxyCfg bodies must be owned by its source entry.")
        if not set(proxy.particles).issubset(entries[proxy_cfg.source].particles):
            raise ValueError("CoupledProxyCfg particles must be owned by its source entry.")
        if proxy_cfg.mass_scale <= 0.0:
            raise ValueError("CoupledProxyCfg.mass_scale must be > 0.")
        if proxy_cfg.collide_interval is not None and proxy_cfg.collide_interval < 1:
            raise ValueError("CoupledProxyCfg.collide_interval must be >= 1.")
        if proxy_cfg.mode not in ("lagged", "staggered"):
            raise ValueError("CoupledProxyCfg.mode must be 'lagged' or 'staggered'.")

    @staticmethod
    def _validate_no_cross_entry_proxy_joints(model: Model, entries: dict[str, _ResolvedEntry]) -> None:
        body_owner = {int(body): name for name, entry in entries.items() for body in entry.bodies}
        for joint, (parent_raw, child_raw) in enumerate(zip(model.joint_parent.numpy(), model.joint_child.numpy())):
            parent = int(parent_raw)
            child = int(child_raw)
            if parent >= 0 and body_owner[parent] != body_owner[child]:
                raise ValueError(
                    f"CoupledProxySolverCfg does not support cross-entry joint {joint} between "
                    f"{body_owner[parent]!r} and {body_owner[child]!r}; keep the articulation in one entry "
                    "or use ADMM coupling."
                )

    @classmethod
    def _apply_proxy_shape_overrides(cls, model: Model, proxies: list[_ResolvedProxy]) -> None:
        shape_bodies = model.shape_body.numpy()
        for proxy in proxies:
            body_set = set(proxy.bodies)
            shape_ids = [shape for shape, body in enumerate(shape_bodies) if int(body) in body_set]
            for name in ("shape_material_ke", "shape_material_kd", "shape_material_mu", "shape_margin", "shape_gap"):
                value = getattr(proxy.config, name)
                array = getattr(model, name, None)
                if value is not None and shape_ids and array is not None:
                    values = array.numpy()
                    values[np.asarray(shape_ids, dtype=np.int32)] = float(value)
                    array.assign(values)
