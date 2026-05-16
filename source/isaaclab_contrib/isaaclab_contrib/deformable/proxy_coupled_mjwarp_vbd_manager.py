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

from .newton_manager_cfg import ProxyCoupledMJWarpVBDSolverCfg
from .vbd_manager import NewtonVBDManager

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class NewtonProxyCoupledMJWarpVBDManager(NewtonVBDManager):
    """:class:`NewtonManager` specialization for proxy-coupled MJWarp + VBD.

    Subclasses :class:`NewtonVBDManager` to inherit the cable builder hooks,
    the cable-aware :meth:`forward` override, and the per-world USD env
    handling. Overrides :meth:`_build_solver` to partition bodies/joints/shapes
    between an ``"mjc"`` MuJoCo entry and a ``"vbd"`` VBD entry, then wraps
    them in :class:`newton.solvers.SolverProxyCoupled` with proxy bodies
    resolved from regex patterns in
    :attr:`ProxyCoupledMJWarpVBDSolverCfg.proxy_body_label_patterns`.
    """

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: ProxyCoupledMJWarpVBDSolverCfg) -> None:
        """Construct :class:`SolverProxyCoupled` and populate base-class slots.

        Partitions bodies/joints/shapes between the two entries by resolving
        :attr:`solver_cfg.mjwarp_entities` and :attr:`solver_cfg.vbd_entities`
        against the env's scene cfg (attached as ``solver_cfg._scene_cfg`` by
        the env's ``__post_init__``), then grep-matching each entity's
        ``prim_path`` template against ``newton.Model.body_label``:

        - Body labels matching the patterns from ``mjwarp_entities`` →
          ``mjc`` entry.
        - Body labels matching the patterns from ``vbd_entities`` → ``vbd`` entry.
        - Body labels matching both → :class:`ValueError` (overlapping
          partition is not allowed).
        - Body labels matching neither → :class:`ValueError` (every body must
          be claimed by exactly one entry).
        - Static shapes (``shape_body == -1``, e.g. ground / table mesh) are
          appended to **both** entries' shape lists; they have no body and so
          don't need an explicit declaration.
        - Deformable particles (from the inherited registry) all go to the
          ``vbd`` entry.
        """
        # Filter sub-solver cfg dicts down to the kwargs accepted by each solver.
        mj_valid = set(inspect.signature(SolverMuJoCo.__init__).parameters) - {"self", "model"}
        mjc_kw = {k: v for k, v in solver_cfg.mjwarp_cfg.to_dict().items() if k in mj_valid}
        vbd_valid = set(inspect.signature(SolverVBD.__init__).parameters) - {"self", "model"}
        vbd_kw = {k: v for k, v in solver_cfg.vbd_cfg.to_dict().items() if k in vbd_valid}

        mjc_bodies, vbd_bodies, mjc_joints, vbd_joints, mjc_shapes, vbd_shapes = cls._partition_model_by_prim_paths(
            model,
            solver_cfg.mjwarp_prim_paths,
            solver_cfg.vbd_prim_paths,
        )
        vbd_particles = list(range(model.particle_count))

        proxy_body_ids = cls._select_proxy_bodies_by_label(model, solver_cfg.proxy_body_label_patterns)
        if solver_cfg.proxy_body_label_patterns and not proxy_body_ids:
            logger.warning(
                "ProxyCoupledMJWarpVBDSolverCfg.proxy_body_label_patterns=%s matched no bodies with COLLIDE_SHAPES. "
                "Rigid bodies will not be visible to VBD.",
                solver_cfg.proxy_body_label_patterns,
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

        Handles IsaacLab's two prim-path placeholder conventions:

        - ``{ENV_REGEX_NS}`` is expanded to ``/World/envs/env_.*``.
        - ``env_.*`` is tightened to ``env_\\d+`` (the cloner only emits
          numeric env indices, so matching a digit run is safer).

        Anchors at the start of the body label and requires either end-of-
        string or a path separator immediately after the template — that
        prevents an entity pattern from spuriously matching a sibling whose
        name has the same prefix (e.g. ``/Cable`` must not match
        ``/CableBag``).
        """
        expanded = prim_path_template.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
        sentinel = "\x00ENV\x00"
        expanded = expanded.replace("env_.*", sentinel)
        escaped = re.escape(expanded).replace(sentinel, r"env_\d+")
        return re.compile(rf"^{escaped}(/|$)")

    @classmethod
    def _partition_model_by_prim_paths(
        cls,
        model: Model,
        mjwarp_prim_paths: list[str],
        vbd_prim_paths: list[str],
    ) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
        """Split bodies/joints/shapes between MuJoCo and VBD entries by prim path.

        Each body's owner is decided by matching its ``model.body_label`` (the
        full USD prim path post-cloning) against the prim-path templates from
        :attr:`ProxyCoupledMJWarpVBDSolverCfg.mjwarp_prim_paths` and
        :attr:`vbd_prim_paths`. Joints inherit their child body's owner.
        Shapes inherit their body's owner; static shapes (``body == -1``) are
        duplicated into both buckets.

        Raises:
            ValueError: If any body matches both partition lists (overlap), or
                if any body matches neither (unclaimed).
        """
        body_count = int(model.body_count)
        joint_count = int(model.joint_count)
        shape_count = int(model.shape_count)

        mjwarp_regexes = [cls._prim_path_template_to_regex(p) for p in mjwarp_prim_paths]
        vbd_regexes = [cls._prim_path_template_to_regex(p) for p in vbd_prim_paths]

        # Classify each body.
        body_owner: list[str | None] = [None] * body_count
        overlapping: list[tuple[int, str]] = []
        unclaimed: list[tuple[int, str]] = []
        for b in range(body_count):
            if b >= len(model.body_label):
                continue
            lbl = model.body_label[b]
            in_mj = any(rx.match(lbl) for rx in mjwarp_regexes)
            in_vbd = any(rx.match(lbl) for rx in vbd_regexes)
            if in_mj and in_vbd:
                overlapping.append((b, lbl))
                continue
            if in_mj:
                body_owner[b] = "mjc"
            elif in_vbd:
                body_owner[b] = "vbd"
            else:
                unclaimed.append((b, lbl))

        if overlapping:
            previews = ", ".join(f"{b}:{lbl!r}" for b, lbl in overlapping[:5])
            raise ValueError(
                f"ProxyCoupledMJWarpVBDSolverCfg: {len(overlapping)} bodies match both "
                f"mjwarp_entities and vbd_entities. First few: {previews}. Make sure each "
                f"scene entity is declared in at most one partition list."
            )
        if unclaimed:
            previews = ", ".join(f"{b}:{lbl!r}" for b, lbl in unclaimed[:5])
            raise ValueError(
                f"ProxyCoupledMJWarpVBDSolverCfg: {len(unclaimed)} bodies are not claimed by "
                f"any entity in mjwarp_entities or vbd_entities. First few: {previews}. Add "
                f"their scene entities to one of the partition lists."
            )

        mjc_bodies = [b for b in range(body_count) if body_owner[b] == "mjc"]
        vbd_bodies = [b for b in range(body_count) if body_owner[b] == "vbd"]

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

        # Shapes inherit their body's owner; static shapes go to both.
        shape_body_np = model.shape_body.numpy() if shape_count else None
        mjc_shapes: list[int] = []
        vbd_shapes: list[int] = []
        for s in range(shape_count):
            body = int(shape_body_np[s])
            if body < 0:
                mjc_shapes.append(s)
                vbd_shapes.append(s)
                continue
            owner = body_owner[body] if 0 <= body < body_count else None
            if owner == "mjc":
                mjc_shapes.append(s)
            elif owner == "vbd":
                vbd_shapes.append(s)

        return mjc_bodies, vbd_bodies, mjc_joints, vbd_joints, mjc_shapes, vbd_shapes

    @staticmethod
    def _select_proxy_bodies_by_label(model: Model, patterns: list[str]) -> list[int]:
        """Resolve proxy bodies from regex patterns on ``model.body_label``.

        Matches each pattern against the short-name component (segment after
        the last ``/``) of each body label, then filters to bodies that own at
        least one shape flagged ``COLLIDE_SHAPES``. Mirrors the selection
        strategy in Newton's ``example_cable_robot_proxy_coupled_solver.py``.
        """
        if not patterns:
            return []

        compiled = [re.compile(p) for p in patterns]
        shape_count = int(model.shape_count)
        shape_body_np = model.shape_body.numpy() if shape_count else None
        shape_flags_np = model.shape_flags.numpy() if shape_count else None
        collide_flag = int(ShapeFlags.COLLIDE_SHAPES)

        # Group shape ids by body for O(num_bodies + num_shapes) instead of O(num_bodies * num_shapes).
        body_has_collide_shape: dict[int, bool] = {}
        for s in range(shape_count):
            body = int(shape_body_np[s])
            if body < 0:
                continue
            if int(shape_flags_np[s]) & collide_flag:
                body_has_collide_shape[body] = True

        proxy_ids: list[int] = []
        for body_id in range(int(model.body_count)):
            if body_id >= len(model.body_label):
                continue
            lbl = model.body_label[body_id]
            short = lbl.rsplit("/", 1)[-1] if "/" in lbl else lbl
            if not any(p.search(short) for p in compiled):
                continue
            if not body_has_collide_shape.get(body_id, False):
                continue
            proxy_ids.append(body_id)

        return proxy_ids
