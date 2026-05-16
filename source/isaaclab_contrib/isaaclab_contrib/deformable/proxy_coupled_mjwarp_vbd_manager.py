# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Proxy-coupled MJWarp + VBD Newton manager.

Wraps :class:`newton.solvers.SolverProxyCoupled` with MuJoCo Warp as the rigid
sub-solver and VBD as the soft sub-solver, exposing selected MuJoCo bodies as
proxies in the VBD view.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import CollisionPipeline, Model, ShapeFlags
from newton.solvers import SolverMuJoCo, SolverProxyCoupled, SolverVBD

from isaaclab.managers import SceneEntityCfg
from isaaclab.physics import PhysicsManager

from .newton_manager_cfg import CoupledNewtonCfg, ProxyCoupledMJWarpVBDSolverCfg
from .vbd_manager import NewtonVBDManager

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveSceneCfg

logger = logging.getLogger(__name__)


class NewtonProxyCoupledMJWarpVBDManager(NewtonVBDManager):
    """:class:`NewtonManager` specialization for proxy-coupled MJWarp + VBD.

    Extends :class:`NewtonVBDManager` and partitions bodies/joints/shapes
    between an ``"mjc"`` MuJoCo entry and a ``"vbd"`` VBD entry, wrapped in
    :class:`newton.solvers.SolverProxyCoupled`. Proxy bodies are resolved from
    :class:`~isaaclab.managers.SceneEntityCfg` specs in
    :attr:`ProxyCoupledMJWarpVBDSolverCfg.proxy_bodies`.
    """

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: ProxyCoupledMJWarpVBDSolverCfg) -> None:
        """Construct :class:`SolverProxyCoupled` and populate base-class slots.

        Partitions the model via :meth:`_partition_model_by_entities` using
        :attr:`solver_cfg.mjwarp_bodies` and :attr:`solver_cfg.vbd_bodies`, and
        resolves proxies via :meth:`_select_proxy_bodies` from
        :attr:`solver_cfg.proxy_bodies`.
        """
        mjc_kw = cls._filter_solver_kwargs(SolverMuJoCo, solver_cfg.mjwarp_cfg)
        vbd_kw = cls._filter_solver_kwargs(SolverVBD, solver_cfg.vbd_cfg)

        outer_cfg = PhysicsManager._cfg
        scene_cfg = outer_cfg.scene_cfg if isinstance(outer_cfg, CoupledNewtonCfg) else None
        if (solver_cfg.mjwarp_bodies or solver_cfg.vbd_bodies or solver_cfg.proxy_bodies) and scene_cfg is None:
            raise ValueError(
                "ProxyCoupledMJWarpVBDSolverCfg requires the outer physics cfg to be a "
                "`CoupledNewtonCfg` with `scene_cfg=self.scene` set (e.g. "
                "`self.sim.physics = CoupledNewtonCfg(solver_cfg=..., scene_cfg=self.scene)`)."
            )

        mjc_bodies, vbd_bodies, mjc_joints, vbd_joints, mjc_shapes, vbd_shapes = cls._partition_model_by_entities(
            model,
            solver_cfg.mjwarp_bodies,
            solver_cfg.vbd_bodies,
            scene_cfg,
        )
        vbd_particles = list(range(model.particle_count))

        proxy_body_ids = cls._select_proxy_bodies(model, solver_cfg.proxy_bodies, scene_cfg)
        if solver_cfg.proxy_bodies and not proxy_body_ids:
            logger.warning(
                "ProxyCoupledMJWarpVBDSolverCfg.proxy_bodies=%s matched no bodies with COLLIDE_SHAPES. "
                "Rigid bodies will not be visible to VBD.",
                solver_cfg.proxy_bodies,
            )

        entries = [
            SolverProxyCoupled.Entry(
                name="mjc",
                solver=lambda v, _kw=mjc_kw: SolverMuJoCo(model=v, **_kw),
                bodies=mjc_bodies,
                joints=mjc_joints,
                shapes=mjc_shapes,
            ),
            SolverProxyCoupled.Entry(
                name="vbd",
                solver=lambda v, _kw=vbd_kw: SolverVBD(model=v, **_kw),
                bodies=vbd_bodies,
                joints=vbd_joints,
                particles=vbd_particles,
                shapes=vbd_shapes,
            ),
        ]

        proxies: list[SolverProxyCoupled.Proxy] = []
        if proxy_body_ids:
            proxies.append(
                SolverProxyCoupled.Proxy(
                    source="mjc",
                    destination="vbd",
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

        NewtonManager._solver = SolverProxyCoupled(
            model=model,
            entries=entries,
            coupling=SolverProxyCoupled.Config(
                proxies=proxies,
                iterations=int(solver_cfg.proxy_iterations),
            ),
        )
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = False

    @classmethod
    def _resolve_entity_to_body_ids(
        cls,
        model: Model,
        entity_cfg: SceneEntityCfg,
        scene_cfg: InteractiveSceneCfg,
        field: str,
    ) -> list[int]:
        """Resolve one :class:`SceneEntityCfg` to ``model.body_label`` indices.

        Scopes the match by the asset's :attr:`prim_path` template (looked up
        on ``scene_cfg`` by :attr:`SceneEntityCfg.name`). If
        :attr:`SceneEntityCfg.body_names` is set, each pattern is full-matched
        against the body's short name (segment after the last ``/``); if
        ``None``, every body under the asset is matched.

        Args:
            field: Cfg attribute name used in error messages.

        Raises:
            ValueError: If the asset is not on ``scene_cfg``, or if any
                ``body_names`` pattern matches zero bodies.
        """
        asset_cfg = getattr(scene_cfg, entity_cfg.name, None)
        if asset_cfg is None or not hasattr(asset_cfg, "prim_path"):
            raise ValueError(
                f"ProxyCoupledMJWarpVBDSolverCfg.{field} references scene entity "
                f"{entity_cfg.name!r}, which is not on the attached scene cfg (or lacks `prim_path`)."
            )
        # `.*` in the template stays a regex wildcard; trailing `(/|$)` keeps
        # `/Cable` from matching `/CableBag`.
        asset_regex = re.compile(rf"^{asset_cfg.prim_path}(/|$)")
        patterns = entity_cfg.body_names
        if isinstance(patterns, str):
            patterns = [patterns]

        if patterns is None:
            return [b for b in range(int(model.body_count)) if asset_regex.match(model.body_label[b])]

        compiled = [re.compile(p) for p in patterns]
        matched_flags: list[bool] = [False] * len(compiled)
        body_ids: list[int] = []
        for body_id in range(int(model.body_count)):
            lbl = model.body_label[body_id]
            if not asset_regex.match(lbl):
                continue
            short = lbl.rsplit("/", 1)[-1]
            hit_index = next((i for i, rx in enumerate(compiled) if rx.fullmatch(short)), None)
            if hit_index is None:
                continue
            matched_flags[hit_index] = True
            body_ids.append(body_id)

        unmatched = [p for p, ok in zip(patterns, matched_flags) if not ok]
        if unmatched:
            raise ValueError(
                f"ProxyCoupledMJWarpVBDSolverCfg.{field}: asset {entity_cfg.name!r} has no bodies "
                f"matching {unmatched}. Check the regex against the asset's body short names."
            )
        return body_ids

    @classmethod
    def _partition_model_by_entities(
        cls,
        model: Model,
        mjwarp_bodies: list[SceneEntityCfg],
        vbd_bodies: list[SceneEntityCfg],
        scene_cfg: InteractiveSceneCfg | None,
    ) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
        """Split bodies/joints/shapes between MuJoCo and VBD entries.

        Body ownership is resolved via :meth:`_resolve_entity_to_body_ids`.
        Joints inherit their child body's owner; shapes inherit their body's
        owner, except static shapes (``body == -1``) always go to VBD.

        Raises:
            ValueError: If any body matches both or neither partition.
        """
        body_count = int(model.body_count)

        mjc_owned: set[int] = set()
        for spec in mjwarp_bodies:
            mjc_owned.update(cls._resolve_entity_to_body_ids(model, spec, scene_cfg, "mjwarp_bodies"))
        vbd_owned: set[int] = set()
        for spec in vbd_bodies:
            vbd_owned.update(cls._resolve_entity_to_body_ids(model, spec, scene_cfg, "vbd_bodies"))

        overlapping_ids = sorted(mjc_owned & vbd_owned)
        if overlapping_ids:
            previews = ", ".join(f"{b}:{model.body_label[b]!r}" for b in overlapping_ids[:5])
            raise ValueError(
                f"ProxyCoupledMJWarpVBDSolverCfg: {len(overlapping_ids)} bodies match both "
                f"`mjwarp_bodies` and `vbd_bodies`. First few: {previews}. Make sure each "
                f"scene entity is declared in at most one partition list."
            )
        unclaimed_ids = [b for b in range(body_count) if b not in mjc_owned and b not in vbd_owned]
        if unclaimed_ids:
            previews = ", ".join(f"{b}:{model.body_label[b]!r}" for b in unclaimed_ids[:5])
            raise ValueError(
                f"ProxyCoupledMJWarpVBDSolverCfg: {len(unclaimed_ids)} bodies are not claimed by "
                f"any entity in `mjwarp_bodies` or `vbd_bodies`. First few: {previews}. Add their "
                f"scene entities to one of the partition lists."
            )

        mjc_joints: list[int] = []
        vbd_joints: list[int] = []
        if int(model.joint_count):
            for j, c in enumerate(model.joint_child.numpy()):
                child = int(c)
                if child in mjc_owned:
                    mjc_joints.append(j)
                elif child in vbd_owned:
                    vbd_joints.append(j)

        # Static shapes (body == -1) go to VBD: its proxy collision pipeline
        # tests rigid proxies against static colliders.
        mjc_shapes: list[int] = []
        vbd_shapes: list[int] = []
        if int(model.shape_count):
            for s, b in enumerate(model.shape_body.numpy()):
                body = int(b)
                if body < 0 or body in vbd_owned:
                    vbd_shapes.append(s)
                elif body in mjc_owned:
                    mjc_shapes.append(s)

        return sorted(mjc_owned), sorted(vbd_owned), mjc_joints, vbd_joints, mjc_shapes, vbd_shapes

    @classmethod
    def _select_proxy_bodies(
        cls,
        model: Model,
        proxy_bodies: list[SceneEntityCfg],
        scene_cfg: InteractiveSceneCfg | None,
    ) -> list[int]:
        """Resolve proxy bodies from per-asset :class:`SceneEntityCfg` specs.

        Delegates to :meth:`_resolve_entity_to_body_ids`, then filters to
        bodies with at least one ``COLLIDE_SHAPES``-flagged shape.
        :attr:`SceneEntityCfg.body_names` is required: proxies are a subset.

        Raises:
            ValueError: If any entry has ``body_names=None``.
        """
        if not proxy_bodies:
            return []

        shape_count = int(model.shape_count)
        shape_body_np = model.shape_body.numpy() if shape_count else None
        shape_flags_np = model.shape_flags.numpy() if shape_count else None
        collide_flag = int(ShapeFlags.COLLIDE_SHAPES)
        collide_bodies: set[int] = {
            int(shape_body_np[s])
            for s in range(shape_count)
            if int(shape_body_np[s]) >= 0 and int(shape_flags_np[s]) & collide_flag
        }

        proxy_ids: list[int] = []
        seen: set[int] = set()
        for spec in proxy_bodies:
            if spec.body_names is None:
                raise ValueError(
                    f"ProxyCoupledMJWarpVBDSolverCfg.proxy_bodies entry for {spec.name!r} has "
                    f"body_names=None. Proxies are a subset of an asset's bodies, so body_names "
                    f"must be a list of regex patterns."
                )
            for body_id in cls._resolve_entity_to_body_ids(model, spec, scene_cfg, "proxy_bodies"):
                if body_id not in collide_bodies or body_id in seen:
                    continue
                seen.add(body_id)
                proxy_ids.append(body_id)

        return proxy_ids
