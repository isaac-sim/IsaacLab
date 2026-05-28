# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton manager wrapping the experimental coupled solvers.

Dispatches on the config subclass to instantiate either
:class:`newton.solvers.experimental.coupled.SolverCoupledProxy` (when given a
:class:`~isaaclab_contrib.coupling.coupled_manager_cfg.CoupledProxySolverCfg`)
or :class:`newton.solvers.experimental.coupled.SolverCoupledAdmm` (when given a
:class:`~isaaclab_contrib.coupling.coupled_manager_cfg.CoupledAdmmSolverCfg`).
Sub-solver classes are resolved from their configs via
:attr:`NewtonCoupledSolverManager._SOLVER_CLASS_BY_CFG_TYPE`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

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
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledAdmm, SolverCoupledProxy

from isaaclab.managers import SceneEntityCfg
from isaaclab.physics import PhysicsManager

from ..deformable.newton_manager_cfg import CoupledNewtonCfg, VBDSolverCfg
from ..deformable.vbd_manager import NewtonVBDManager
from .coupled_manager_cfg import CoupledAdmmSolverCfg, CoupledProxySolverCfg, CoupledSolverCfg

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveSceneCfg


class NewtonCoupledSolverManager(NewtonVBDManager):
    """Newton manager wrapping the experimental coupled solvers.

    The source/destination entries are built generically from the
    ``src_solver_cfg`` / ``dst_solver_cfg`` types via
    :attr:`_SOLVER_CLASS_BY_CFG_TYPE`; the coupling algorithm is selected from
    the :class:`~isaaclab_contrib.coupling.coupled_manager_cfg.CoupledSolverCfg`
    subclass passed in.
    """

    _SOLVER_CLASS_BY_CFG_TYPE: ClassVar[dict[type[NewtonSolverCfg], type[SolverBase]]] = {
        MJWarpSolverCfg: SolverMuJoCo,
        VBDSolverCfg: SolverVBD,
        FeatherstoneSolverCfg: SolverFeatherstone,
        XPBDSolverCfg: SolverXPBD,
        KaminoSolverCfg: SolverKamino,
    }
    """Registry of Newton solver-cfg classes to their concrete solver classes."""

    @classmethod
    def _resolve_solver_class(cls, sub_cfg: NewtonSolverCfg) -> type[SolverBase]:
        """Look ``sub_cfg``'s concrete solver class up in :attr:`_SOLVER_CLASS_BY_CFG_TYPE`."""
        try:
            return cls._SOLVER_CLASS_BY_CFG_TYPE[type(sub_cfg)]
        except KeyError:
            known = ", ".join(sorted(t.__name__ for t in cls._SOLVER_CLASS_BY_CFG_TYPE))
            raise ValueError(
                f"No Newton solver registered for sub-cfg type {type(sub_cfg).__name__!r}. "
                f"Register it in `NewtonCoupledSolverManager._SOLVER_CLASS_BY_CFG_TYPE` (known: {known})."
            ) from None

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CoupledSolverCfg) -> None:
        src_solver_cls = cls._resolve_solver_class(solver_cfg.src_solver_cfg)
        dst_solver_cls = cls._resolve_solver_class(solver_cfg.dst_solver_cfg)
        src_kw = cls._filter_solver_kwargs(src_solver_cls, solver_cfg.src_solver_cfg)
        dst_kw = cls._filter_solver_kwargs(dst_solver_cls, solver_cfg.dst_solver_cfg)

        outer_cfg = PhysicsManager._cfg
        scene_cfg = outer_cfg.scene_cfg if isinstance(outer_cfg, CoupledNewtonCfg) else None

        src_bodies, dst_bodies, src_joints, dst_joints, src_shapes, dst_shapes = cls._partition_model_by_entities(
            model,
            solver_cfg.src_bodies,
            solver_cfg.dst_bodies,
            scene_cfg,
        )
        dst_particles = list(range(model.particle_count))

        entries = [
            SolverCoupled.Entry(
                name="src",
                solver=lambda v, _cls=src_solver_cls, _kw=src_kw: _cls(model=v, **_kw),
                bodies=src_bodies,
                joints=src_joints,
                shapes=src_shapes,
            ),
            SolverCoupled.Entry(
                name="dst",
                solver=lambda v, _cls=dst_solver_cls, _kw=dst_kw: _cls(model=v, **_kw),
                bodies=dst_bodies,
                joints=dst_joints,
                particles=dst_particles,
                shapes=dst_shapes,
            ),
        ]

        if isinstance(solver_cfg, CoupledProxySolverCfg):
            NewtonManager._solver = cls._build_proxy_coupled_solver(model, entries, solver_cfg, scene_cfg)
            NewtonManager._use_single_state = False
            NewtonManager._needs_collision_pipeline = False
        elif isinstance(solver_cfg, CoupledAdmmSolverCfg):
            NewtonManager._solver = cls._build_admm_coupled_solver(model, entries, solver_cfg)
            NewtonManager._use_single_state = False
            NewtonManager._needs_collision_pipeline = True
        else:
            raise TypeError(
                f"CoupledSolverCfg subclass {type(solver_cfg).__name__!r} is not supported by "
                "`NewtonCoupledSolverManager`. Use `CoupledProxySolverCfg` or `CoupledAdmmSolverCfg`."
            )

    @classmethod
    def _build_proxy_coupled_solver(
        cls,
        model: Model,
        entries: list[SolverCoupled.Entry],
        solver_cfg: CoupledProxySolverCfg,
        scene_cfg: InteractiveSceneCfg | None,
    ) -> SolverCoupledProxy:
        proxy_body_ids = cls._select_proxy_bodies(model, solver_cfg.proxy_bodies, scene_cfg)
        if solver_cfg.proxy_bodies and not proxy_body_ids:
            raise ValueError(
                f"CoupledProxySolverCfg.proxy_bodies={solver_cfg.proxy_bodies!r} resolved to "
                "zero bodies after filtering for `ShapeFlags.COLLIDE_SHAPES`. Source bodies would not "
                "be visible to the destination solver; check that the selected bodies own at least "
                "one collidable shape."
            )

        proxies: list[SolverCoupledProxy.Proxy] = []
        if proxy_body_ids:
            proxies.append(
                SolverCoupledProxy.Proxy(
                    source="src",
                    destination="dst",
                    bodies=proxy_body_ids,
                    mode=solver_cfg.proxy_mode,
                    mass_scale=float(solver_cfg.proxy_mass_scale),
                    collision_pipeline=lambda destination_model: CollisionPipeline(
                        destination_model,
                        broad_phase="explicit",
                    ),
                    collide_interval=int(solver_cfg.proxy_collide_interval),
                )
            )

        return SolverCoupledProxy(
            model=model,
            entries=entries,
            coupling=SolverCoupledProxy.Config(
                proxies=proxies,
                iterations=int(solver_cfg.proxy_iterations),
            ),
        )

    @classmethod
    def _build_admm_coupled_solver(
        cls,
        model: Model,
        entries: list[SolverCoupled.Entry],
        solver_cfg: CoupledAdmmSolverCfg,
    ) -> SolverCoupledAdmm:
        contact_pairs: list[SolverCoupledAdmm.ContactPair] = []
        if solver_cfg.enable_contacts:
            contact_pairs.append(
                SolverCoupledAdmm.ContactPair(
                    source="src",
                    destination="dst",
                    contact_distance=solver_cfg.contact_distance,
                    detection_margin=solver_cfg.detection_margin,
                )
            )
        return SolverCoupledAdmm(
            model=model,
            entries=entries,
            coupling=SolverCoupledAdmm.Config(
                iterations=int(solver_cfg.iterations),
                rho=float(solver_cfg.rho),
                gamma=float(solver_cfg.gamma),
                baumgarte=float(solver_cfg.baumgarte),
                joint_stiffness=float(solver_cfg.joint_stiffness),
                joint_damping=float(solver_cfg.joint_damping),
                joint_angular_stiffness=float(solver_cfg.joint_angular_stiffness),
                joint_angular_damping=float(solver_cfg.joint_angular_damping),
                contact_pairs=contact_pairs,
            ),
        )

    @classmethod
    def _resolve_entity_to_body_ids(
        cls,
        model: Model,
        spec: SceneEntityCfg | str,
        scene_cfg: InteractiveSceneCfg | None,
        field: str,
    ) -> list[int]:
        """Resolve one selector to ``model.body_label`` indices.

        Strings are matched directly via ``^<string>(/|$)``. :class:`SceneEntityCfg`
        looks up the asset's ``prim_path`` on ``scene_cfg`` and (optionally)
        full-matches ``body_names`` regexes against the body short name.

        Raises:
            ValueError: Asset missing on ``scene_cfg``; ``body_names`` pattern
                with zero matches; or a string with zero matches.
        """
        if isinstance(spec, str):
            prim_path, patterns, spec_repr = spec, None, f"prim-path regex {spec!r}"
        else:
            asset_cfg = getattr(scene_cfg, spec.name, None) if scene_cfg is not None else None
            if asset_cfg is None or not hasattr(asset_cfg, "prim_path"):
                raise ValueError(
                    f"CoupledSolverCfg.{field}: scene entity {spec.name!r} is not on the "
                    "attached scene cfg (or lacks `prim_path`)."
                )
            prim_path = asset_cfg.prim_path
            patterns = [spec.body_names] if isinstance(spec.body_names, str) else spec.body_names
            spec_repr = f"asset {spec.name!r}"

        asset_re = re.compile(rf"^{prim_path}(/|$)")
        # Treat patterns=None as ".*" so the loop is uniform across both branches.
        compiled = [re.compile(p) for p in (patterns if patterns is not None else [r".*"])]
        matched = [False] * len(compiled)
        body_ids: list[int] = []
        for b in range(int(model.body_count)):
            lbl = model.body_label[b]
            if not asset_re.match(lbl):
                continue
            short = lbl.rsplit("/", 1)[-1]
            hit = next((i for i, rx in enumerate(compiled) if rx.fullmatch(short)), None)
            if hit is None:
                continue
            matched[hit] = True
            body_ids.append(b)

        if patterns is not None:
            unmatched = [p for p, ok in zip(patterns, matched) if not ok]
            if unmatched:
                raise ValueError(f"CoupledSolverCfg.{field}: {spec_repr} has no bodies matching {unmatched}.")
        elif isinstance(spec, str) and not body_ids:
            # Strings have no asset-cfg safety net — zero matches is almost always a typo.
            raise ValueError(
                f"CoupledSolverCfg.{field}: {spec_repr} matched no bodies in "
                "`model.body_label` (labels are full post-clone prim paths)."
            )
        return body_ids

    @classmethod
    def _partition_model_by_entities(
        cls,
        model: Model,
        src_bodies: list[SceneEntityCfg | str],
        dst_bodies: list[SceneEntityCfg | str],
        scene_cfg: InteractiveSceneCfg | None,
    ) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
        """Split bodies/joints/shapes between the source and destination entries.

        Joints/shapes inherit their (child) body's owner. Static shapes
        (``body == -1``) always go to the destination entry so its proxy
        collision pipeline tests source proxies against the world.

        Raises:
            ValueError: A body matches both partitions or neither.
        """
        src_owned: set[int] = set()
        for spec in src_bodies:
            src_owned.update(cls._resolve_entity_to_body_ids(model, spec, scene_cfg, "src_bodies"))
        dst_owned: set[int] = set()
        for spec in dst_bodies:
            dst_owned.update(cls._resolve_entity_to_body_ids(model, spec, scene_cfg, "dst_bodies"))

        def _preview(ids: list[int]) -> str:
            return ", ".join(f"{b}:{model.body_label[b]!r}" for b in ids[:5])

        if overlap := sorted(src_owned & dst_owned):
            raise ValueError(
                f"CoupledSolverCfg: {len(overlap)} bodies match both `src_bodies` and `dst_bodies` "
                f"(first few: {_preview(overlap)})."
            )
        unclaimed = [b for b in range(int(model.body_count)) if b not in src_owned and b not in dst_owned]
        if unclaimed:
            raise ValueError(
                f"CoupledSolverCfg: {len(unclaimed)} bodies unclaimed by `src_bodies`/`dst_bodies` "
                f"(first few: {_preview(unclaimed)})."
            )

        src_joints: list[int] = []
        dst_joints: list[int] = []
        if int(model.joint_count):
            for j, c in enumerate(model.joint_child.numpy()):
                child = int(c)
                if child in src_owned:
                    src_joints.append(j)
                elif child in dst_owned:
                    dst_joints.append(j)

        src_shapes: list[int] = []
        dst_shapes: list[int] = []
        if int(model.shape_count):
            for s, b in enumerate(model.shape_body.numpy()):
                body = int(b)
                if body < 0 or body in dst_owned:
                    dst_shapes.append(s)
                elif body in src_owned:
                    src_shapes.append(s)

        return sorted(src_owned), sorted(dst_owned), src_joints, dst_joints, src_shapes, dst_shapes

    @classmethod
    def _select_proxy_bodies(
        cls,
        model: Model,
        proxy_bodies: list[SceneEntityCfg | str],
        scene_cfg: InteractiveSceneCfg | None,
    ) -> list[int]:
        """Resolve proxy bodies, filtered to those owning a ``COLLIDE_SHAPES`` shape.

        Raises:
            ValueError: A :class:`SceneEntityCfg` entry has ``body_names=None``
                (proxies must be a subset, not the whole asset).
        """
        if not proxy_bodies:
            return []

        shape_count = int(model.shape_count)
        collide_flag = int(ShapeFlags.COLLIDE_SHAPES)
        collide_bodies: set[int] = set()
        if shape_count:
            shape_body_np = model.shape_body.numpy()
            shape_flags_np = model.shape_flags.numpy()
            collide_bodies = {
                int(shape_body_np[s])
                for s in range(shape_count)
                if int(shape_body_np[s]) >= 0 and int(shape_flags_np[s]) & collide_flag
            }

        proxy_ids: list[int] = []
        seen: set[int] = set()
        for spec in proxy_bodies:
            if isinstance(spec, SceneEntityCfg) and spec.body_names is None:
                raise ValueError(
                    f"CoupledProxySolverCfg.proxy_bodies entry {spec.name!r} requires `body_names` "
                    "(proxies must be a subset of the asset)."
                )
            for body_id in cls._resolve_entity_to_body_ids(model, spec, scene_cfg, "proxy_bodies"):
                if body_id in collide_bodies and body_id not in seen:
                    seen.add(body_id)
                    proxy_ids.append(body_id)

        return proxy_ids
