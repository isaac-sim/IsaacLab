# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Proxy-coupled MJWarp + VBD Newton manager.

Wraps :class:`newton.solvers.SolverCoupledProxy` with MuJoCo Warp as the rigid
sub-solver and VBD as the soft sub-solver, exposing selected MuJoCo bodies as
proxies in the VBD view.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import CollisionPipeline, Model, ShapeFlags
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.solvers.experimental.coupled import SolverCoupledProxy

from isaaclab.managers import SceneEntityCfg
from isaaclab.physics import PhysicsManager

from .newton_manager_cfg import CoupledNewtonCfg, ProxyCoupledMJWarpVBDSolverCfg
from .vbd_manager import NewtonVBDManager

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveSceneCfg

logger = logging.getLogger(__name__)


class NewtonProxyCoupledMJWarpVBDManager(NewtonVBDManager):
    """Newton manager wrapping :class:`newton.solvers.SolverCoupledProxy` with an MJWarp+VBD split.

    Bodies/joints/shapes are partitioned between the two entries; all particles
    are solved by VBD.
    """

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: ProxyCoupledMJWarpVBDSolverCfg) -> None:
        mjc_kw = cls._filter_solver_kwargs(SolverMuJoCo, solver_cfg.mjwarp_cfg)
        vbd_kw = cls._filter_solver_kwargs(SolverVBD, solver_cfg.vbd_cfg)

        outer_cfg = PhysicsManager._cfg
        scene_cfg = outer_cfg.scene_cfg if isinstance(outer_cfg, CoupledNewtonCfg) else None

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
            SolverCoupledProxy.Entry(
                name="mjc",
                solver=lambda v, _kw=mjc_kw: SolverMuJoCo(model=v, **_kw),
                bodies=mjc_bodies,
                joints=mjc_joints,
                shapes=mjc_shapes,
            ),
            SolverCoupledProxy.Entry(
                name="vbd",
                solver=lambda v, _kw=vbd_kw: SolverVBD(model=v, **_kw),
                bodies=vbd_bodies,
                joints=vbd_joints,
                particles=vbd_particles,
                shapes=vbd_shapes,
            ),
        ]

        proxies: list[SolverCoupledProxy.Proxy] = []
        if proxy_body_ids:
            proxies.append(
                SolverCoupledProxy.Proxy(
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

        NewtonManager._solver = SolverCoupledProxy(
            model=model,
            entries=entries,
            coupling=SolverCoupledProxy.Config(
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
                    f"ProxyCoupledMJWarpVBDSolverCfg.{field}: scene entity {spec.name!r} "
                    f"is not on the attached scene cfg (or lacks `prim_path`)."
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
                raise ValueError(
                    f"ProxyCoupledMJWarpVBDSolverCfg.{field}: {spec_repr} has no bodies matching {unmatched}."
                )
        elif isinstance(spec, str) and not body_ids:
            # Strings have no asset-cfg safety net — zero matches is almost always a typo.
            raise ValueError(
                f"ProxyCoupledMJWarpVBDSolverCfg.{field}: {spec_repr} matched no bodies in "
                f"`model.body_label` (labels are full post-clone prim paths)."
            )
        return body_ids

    @classmethod
    def _partition_model_by_entities(
        cls,
        model: Model,
        mjwarp_bodies: list[SceneEntityCfg | str],
        vbd_bodies: list[SceneEntityCfg | str],
        scene_cfg: InteractiveSceneCfg | None,
    ) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
        """Split bodies/joints/shapes between the MJWarp and VBD entries.

        Joints/shapes inherit their (child) body's owner. Static shapes
        (``body == -1``) always go to VBD so its proxy collision pipeline
        tests rigid proxies against the world.

        Raises:
            ValueError: A body matches both partitions or neither.
        """
        mjc_owned: set[int] = set()
        for spec in mjwarp_bodies:
            mjc_owned.update(cls._resolve_entity_to_body_ids(model, spec, scene_cfg, "mjwarp_bodies"))
        vbd_owned: set[int] = set()
        for spec in vbd_bodies:
            vbd_owned.update(cls._resolve_entity_to_body_ids(model, spec, scene_cfg, "vbd_bodies"))

        def _preview(ids: list[int]) -> str:
            return ", ".join(f"{b}:{model.body_label[b]!r}" for b in ids[:5])

        if overlap := sorted(mjc_owned & vbd_owned):
            raise ValueError(
                f"ProxyCoupledMJWarpVBDSolverCfg: {len(overlap)} bodies match both "
                f"`mjwarp_bodies` and `vbd_bodies` (first few: {_preview(overlap)})."
            )
        unclaimed = [b for b in range(int(model.body_count)) if b not in mjc_owned and b not in vbd_owned]
        if unclaimed:
            raise ValueError(
                f"ProxyCoupledMJWarpVBDSolverCfg: {len(unclaimed)} bodies unclaimed by "
                f"`mjwarp_bodies`/`vbd_bodies` (first few: {_preview(unclaimed)})."
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
                    f"ProxyCoupledMJWarpVBDSolverCfg.proxy_bodies entry {spec.name!r} requires "
                    f"`body_names` (proxies must be a subset of the asset)."
                )
            for body_id in cls._resolve_entity_to_body_ids(model, spec, scene_cfg, "proxy_bodies"):
                if body_id in collide_bodies and body_id not in seen:
                    seen.add(body_id)
                    proxy_ids.append(body_id)

        return proxy_ids
