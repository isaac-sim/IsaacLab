# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Proxy-coupled MJWarp + VBD Newton manager.

Wraps :class:`newton.solvers.SolverProxyCoupled` with MuJoCo Warp as the rigid
sub-solver and VBD as the soft sub-solver. Selected MuJoCo bodies are exposed
as proxy bodies in the VBD view via lagged-impulse virtual-proxy coupling
(see Newton's ``example_cable_robot_proxy_coupled_solver.py``).
"""

from __future__ import annotations

import inspect
import logging
import re
from typing import TYPE_CHECKING

import newton
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import Model, ShapeFlags
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

    Subclasses :class:`NewtonVBDManager` to inherit the cable builder hooks,
    the cable-aware :meth:`forward` override, and the per-world USD env
    handling. Overrides :meth:`_build_solver` to partition bodies/joints/shapes
    between an ``"mjc"`` MuJoCo entry and a ``"vbd"`` VBD entry, then wraps
    them in :class:`newton.solvers.SolverProxyCoupled` with proxy bodies
    resolved from per-asset :class:`~isaaclab.managers.SceneEntityCfg` specs in
    :attr:`ProxyCoupledMJWarpVBDSolverCfg.proxy_bodies`.
    """

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: ProxyCoupledMJWarpVBDSolverCfg) -> None:
        """Construct :class:`SolverProxyCoupled` and populate base-class slots.

        Partitions bodies/joints/shapes between the two entries by resolving
        :attr:`solver_cfg.mjwarp_bodies` and :attr:`solver_cfg.vbd_bodies`
        (both ``list[SceneEntityCfg]``) against ``newton.Model.body_label``,
        using each entry's asset :attr:`prim_path` template (read from
        :attr:`CoupledNewtonCfg.scene_cfg`) plus an optional
        :attr:`~isaaclab.managers.SceneEntityCfg.body_names` regex filter:

        - Body labels matching :attr:`solver_cfg.mjwarp_bodies` → ``mjc``
          entry.
        - Body labels matching :attr:`solver_cfg.vbd_bodies` → ``vbd`` entry.
        - Body labels matching both → :class:`ValueError` (overlapping
          partition is not allowed).
        - Body labels matching neither → :class:`ValueError` (every body must
          be claimed by exactly one entry).
        - Static shapes (``shape_body == -1``, e.g. ground / table mesh) go
          to the ``vbd`` entry only. ``SolverCoupled`` enforces disjoint shape
          ownership, and the VBD entry owns the proxy collision pipeline that
          tests the rigid proxy bodies against static colliders, so this is
          where static geometry needs to live. The pattern matches Newton's
          ``example_cable_robot_proxy_coupled_solver.py``.
        - Deformable particles (from the inherited registry) all go to the
          ``vbd`` entry.

        Proxy bodies are resolved separately via :meth:`_select_proxy_bodies`,
        which uses the same :class:`SceneEntityCfg` machinery on
        :attr:`solver_cfg.proxy_bodies` plus a ``COLLIDE_SHAPES`` filter.
        """
        # Filter sub-solver cfg dicts down to the kwargs accepted by each solver.
        mj_valid = set(inspect.signature(SolverMuJoCo.__init__).parameters) - {"self", "model"}
        mjc_kw = {k: v for k, v in solver_cfg.mjwarp_cfg.to_dict().items() if k in mj_valid}
        vbd_valid = set(inspect.signature(SolverVBD.__init__).parameters) - {"self", "model"}
        vbd_kw = {k: v for k, v in solver_cfg.vbd_cfg.to_dict().items() if k in vbd_valid}

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
                    collision_pipeline=lambda destination_model: newton.examples.create_collision_pipeline(
                        destination_model, args=None
                    ),
                    collide_interval=int(solver_cfg.proxy_collide_interval),
                )
            )

        coupled = SolverProxyCoupled(
            model=model,
            entries=entries,
            coupling=SolverProxyCoupled.Config(
                proxies=proxies,
                iterations=int(solver_cfg.proxy_iterations),
            ),
        )

        NewtonManager._solver = coupled
        NewtonManager._use_single_state = False
        # SolverProxyCoupled owns the per-proxy collision pipeline; the outer
        # NewtonManager does not run its own.
        NewtonManager._needs_collision_pipeline = False

        logger.info(
            "Proxy-coupled MJWarp+VBD: mjc bodies=%d joints=%d shapes=%d | "
            "vbd bodies=%d joints=%d shapes=%d particles=%d | proxies=%d",
            len(mjc_bodies),
            len(mjc_joints),
            len(mjc_shapes),
            len(vbd_bodies),
            len(vbd_joints),
            len(vbd_shapes),
            len(vbd_particles),
            len(proxy_body_ids),
        )

    @staticmethod
    def _prim_path_template_to_regex(prim_path_template: str) -> re.Pattern[str]:
        """Compile a scene-entity prim-path template into a body-label regex.

        IsaacLab prim-path templates already use ``.*`` as a regex wildcard
        (typically inside ``env_.*``) and ``{ENV_REGEX_NS}`` as a shorthand
        for ``/World/envs/env_.*``. ``newton.Model.body_label`` mixes labels
        with that wildcard expanded (``env_0`` for USD-imported bodies)
        and labels where the wildcard is preserved verbatim (``env_.*`` for
        bodies added by builder hooks such as cables) — so the regex must
        match both. Solution: keep ``.*`` as a regex wildcard, ``re.escape``
        everything else.

        Anchors at the start of the body label and requires either end-of-
        string or a path separator immediately after the template — that
        prevents an entity pattern from spuriously matching a sibling whose
        name has the same prefix (e.g. ``/Cable`` must not match
        ``/CableBag``).
        """
        expanded = prim_path_template.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
        # Split on `.*` so each segment can be re.escape'd literally and the
        # wildcards re-joined as regex `.*`. Matches `env_0` (via `.*` → `0`)
        # AND literal `env_.*` (via `.*` → the string `.*`).
        parts = expanded.split(".*")
        pattern = ".*".join(re.escape(p) for p in parts)
        return re.compile(rf"^{pattern}(/|$)")

    @classmethod
    def _resolve_entity_to_body_ids(
        cls,
        model: Model,
        entity_cfg: SceneEntityCfg,
        scene_cfg: InteractiveSceneCfg,
        field: str,
    ) -> list[int]:
        """Resolve one :class:`SceneEntityCfg` to a list of ``model.body_label`` indices.

        The asset's :attr:`prim_path` template (looked up on ``scene_cfg`` by
        :attr:`SceneEntityCfg.name`) scopes the body-label match. If
        :attr:`SceneEntityCfg.body_names` is set, each pattern is full-matched
        against the body's short name (segment after the last ``/``) — same
        convention as :func:`isaaclab.utils.string.resolve_matching_names`. If
        :attr:`body_names` is ``None``, every body under the asset's
        :attr:`prim_path` is matched.

        Args:
            field: Cfg attribute name (e.g. ``"mjwarp_bodies"``) used in error
                messages so the user sees which field is misconfigured.

        Raises:
            ValueError: If the asset is not on ``scene_cfg``, or if any
                ``body_names`` pattern matches zero bodies on the asset.
        """
        asset_cfg = getattr(scene_cfg, entity_cfg.name, None)
        if asset_cfg is None or not hasattr(asset_cfg, "prim_path"):
            raise ValueError(
                f"ProxyCoupledMJWarpVBDSolverCfg.{field} references scene entity "
                f"{entity_cfg.name!r}, which is not on the attached scene cfg (or lacks `prim_path`)."
            )
        asset_regex = cls._prim_path_template_to_regex(asset_cfg.prim_path)
        patterns = entity_cfg.body_names
        if isinstance(patterns, str):
            patterns = [patterns]

        if patterns is None:
            return [
                body_id
                for body_id in range(int(model.body_count))
                if body_id < len(model.body_label) and asset_regex.match(model.body_label[body_id])
            ]

        compiled = [re.compile(p) for p in patterns]
        matched_per_pattern: list[list[str]] = [[] for _ in compiled]
        body_ids: list[int] = []
        for body_id in range(int(model.body_count)):
            if body_id >= len(model.body_label):
                continue
            lbl = model.body_label[body_id]
            if not asset_regex.match(lbl):
                continue
            short = lbl.rsplit("/", 1)[-1] if "/" in lbl else lbl
            hit_index = next((i for i, rx in enumerate(compiled) if rx.fullmatch(short)), None)
            if hit_index is None:
                continue
            matched_per_pattern[hit_index].append(short)
            body_ids.append(body_id)

        unmatched = [p for p, m in zip(patterns, matched_per_pattern) if not m]
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

        Each body's owner is decided by resolving the
        :class:`SceneEntityCfg` entries in
        :attr:`ProxyCoupledMJWarpVBDSolverCfg.mjwarp_bodies` and
        :attr:`vbd_bodies` via :meth:`_resolve_entity_to_body_ids`. Joints
        inherit their child body's owner. Shapes inherit their body's owner;
        static shapes (``body == -1``) always go to the VBD entry.

        Raises:
            ValueError: If ``scene_cfg`` is missing (and either partition is
                non-empty), if any body matches both partition lists
                (overlap), or if any body matches neither (unclaimed).
        """
        if scene_cfg is None and (mjwarp_bodies or vbd_bodies):
            raise ValueError(
                "ProxyCoupledMJWarpVBDSolverCfg requires the outer physics cfg to be a "
                "`CoupledNewtonCfg` with `scene_cfg=self.scene` set so `mjwarp_bodies` / "
                "`vbd_bodies` SceneEntityCfg specs can be resolved."
            )

        body_count = int(model.body_count)
        joint_count = int(model.joint_count)
        shape_count = int(model.shape_count)

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
        unclaimed_ids = [
            b for b in range(body_count) if b < len(model.body_label) and b not in mjc_owned and b not in vbd_owned
        ]
        if unclaimed_ids:
            previews = ", ".join(f"{b}:{model.body_label[b]!r}" for b in unclaimed_ids[:5])
            raise ValueError(
                f"ProxyCoupledMJWarpVBDSolverCfg: {len(unclaimed_ids)} bodies are not claimed by "
                f"any entity in `mjwarp_bodies` or `vbd_bodies`. First few: {previews}. Add their "
                f"scene entities to one of the partition lists."
            )

        body_owner: list[str | None] = [None] * body_count
        for b in mjc_owned:
            body_owner[b] = "mjc"
        for b in vbd_owned:
            body_owner[b] = "vbd"

        mjc_bodies_out = sorted(mjc_owned)
        vbd_bodies_out = sorted(vbd_owned)

        # Joints follow their child body's owner. Joints with no child (or
        # whose child wasn't classified — only possible for body indices past
        # the label array) are dropped from both buckets; they would have
        # surfaced as 'unclaimed' above if they referenced real bodies.
        joint_child_np = model.joint_child.numpy() if joint_count else None
        mjc_joints: list[int] = []
        vbd_joints: list[int] = []
        for j in range(joint_count):
            child = int(joint_child_np[j])
            if 0 <= child < body_count:
                owner = body_owner[child]
                if owner == "mjc":
                    mjc_joints.append(j)
                elif owner == "vbd":
                    vbd_joints.append(j)

        # Shapes inherit their body's owner; static shapes go to the VBD entry
        # only. ``SolverCoupled`` enforces disjoint shape ownership, and the
        # VBD entry's proxy collision pipeline is what tests rigid proxy bodies
        # against static colliders.
        shape_body_np = model.shape_body.numpy() if shape_count else None
        mjc_shapes: list[int] = []
        vbd_shapes: list[int] = []
        for s in range(shape_count):
            body = int(shape_body_np[s])
            if body < 0:
                vbd_shapes.append(s)
                continue
            owner = body_owner[body] if 0 <= body < body_count else None
            if owner == "mjc":
                mjc_shapes.append(s)
            elif owner == "vbd":
                vbd_shapes.append(s)

        return mjc_bodies_out, vbd_bodies_out, mjc_joints, vbd_joints, mjc_shapes, vbd_shapes

    @classmethod
    def _select_proxy_bodies(
        cls,
        model: Model,
        proxy_bodies: list[SceneEntityCfg],
        scene_cfg: InteractiveSceneCfg | None,
    ) -> list[int]:
        """Resolve proxy bodies from per-asset :class:`SceneEntityCfg` specs.

        Calls :meth:`_resolve_entity_to_body_ids` per entry (so the
        asset-scoped + ``body_names``-regex semantics match
        :attr:`mjwarp_bodies` / :attr:`vbd_bodies`), then filters to bodies
        that own at least one shape flagged ``COLLIDE_SHAPES``.
        :attr:`SceneEntityCfg.body_names` is **required** here — proxies are
        a subset, not "every body under the asset".

        Raises:
            ValueError: If :attr:`proxy_bodies` is non-empty but ``scene_cfg``
                is missing, if any entry has ``body_names=None``, or if any
                body-name regex matches zero bodies on its asset.
        """
        if not proxy_bodies:
            return []
        if scene_cfg is None:
            raise ValueError(
                "ProxyCoupledMJWarpVBDSolverCfg.proxy_bodies requires the outer physics cfg to be a "
                "`CoupledNewtonCfg` with `scene_cfg=self.scene` set (e.g. "
                "`self.sim.physics = CoupledNewtonCfg(solver_cfg=..., scene_cfg=self.scene)`)."
            )

        shape_count = int(model.shape_count)
        shape_body_np = model.shape_body.numpy() if shape_count else None
        shape_flags_np = model.shape_flags.numpy() if shape_count else None
        collide_flag = int(ShapeFlags.COLLIDE_SHAPES)
        body_has_collide_shape: dict[int, bool] = {}
        for s in range(shape_count):
            body = int(shape_body_np[s])
            if body < 0:
                continue
            if int(shape_flags_np[s]) & collide_flag:
                body_has_collide_shape[body] = True

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
                if not body_has_collide_shape.get(body_id, False):
                    continue
                if body_id in seen:
                    continue
                seen.add(body_id)
                proxy_ids.append(body_id)

        return proxy_ids
