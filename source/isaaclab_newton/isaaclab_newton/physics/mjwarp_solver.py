# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Warp solver implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from mujoco_warp._src.types import MJ_MINMU, vec5
from newton import Contacts, Model, State
from newton.solvers import SolverMuJoCo

from .newton_solver import NewtonSolver, solver_kwargs

if TYPE_CHECKING:
    from .mjwarp_solver_cfg import MJWarpSolverCfg
    from .mjwarp_tendon_control import MjWarpTendonControl

logger = logging.getLogger(__name__)

_SENSORS_BY_STATE_ATTRIBUTE = {"body_qdd": "the IMU or PVA sensor", "body_parent_f": "the joint-wrench sensor"}


@wp.kernel
def _multiply_contact_friction(
    count: wp.array[int],
    contact_indices: wp.array[int],
    contact_scale: wp.array[float],
    contact_geom: wp.array[wp.vec2i],
    contact_world: wp.array[int],
    contact_friction: wp.array[vec5],
    geom_shape: wp.array2d[int],
    shape_mode: wp.array[int],
    geom_priority: wp.array[int],
    geom_friction: wp.array2d[wp.vec3],
    explicit_pairs: wp.array[wp.int64],
):
    tid = wp.tid()
    if tid >= count[0]:
        return
    cid = tid
    if contact_indices:
        cid = contact_indices[tid]
    if cid < 0 or cid >= contact_friction.shape[0]:
        return
    geoms = contact_geom[cid]
    g1, g2 = geoms[0], geoms[1]
    # Flex contacts and unequal-priority geoms keep their native parameters.
    if g1 < 0 or g2 < 0:
        return
    if geom_priority[g1] != geom_priority[g2]:
        return
    world = contact_world[cid]
    s1 = geom_shape[world, g1]
    s2 = geom_shape[world, g2]
    if s1 < 0 or s2 < 0:
        return
    if wp.max(shape_mode[s1], shape_mode[s2]) != 2:
        return
    # Explicit MuJoCo pairs own their (potentially anisotropic) friction.
    if explicit_pairs:
        key = wp.int64(wp.min(g1, g2)) * wp.int64(geom_shape.shape[1]) + wp.int64(wp.max(g1, g2))
        index = wp.lower_bound(explicit_pairs, key)
        if index < explicit_pairs.shape[0] and explicit_pairs[index] == key:
            return
    friction_world = world % geom_friction.shape[0]
    mu = wp.max(MJ_MINMU, geom_friction[friction_world, g1][0] * geom_friction[friction_world, g2][0])
    if contact_scale and contact_scale[tid] > 0.0:
        mu *= contact_scale[tid]
    friction = contact_friction[cid]
    friction[0] = mu
    friction[1] = mu
    contact_friction[cid] = friction


class MaterialSolverMuJoCo(SolverMuJoCo):
    """Honor explicit material multiplication without changing MuJoCo defaults.

    Only sliding friction is adapted. Other combine modes retain native behavior;
    an explicitly authored ``max`` takes precedence over ``multiply``. MuJoCo geom
    priority, explicit contact pairs, and torsional/rolling friction remain native.
    """

    def __init__(self, model: Model, **kwargs):
        modes = getattr(getattr(model, "isaaclab", None), "friction_combine_mode", None)
        self._multiply_friction = modes is not None and bool(np.any(modes.numpy() == 2))
        if self._multiply_friction and kwargs.get("use_mujoco_cpu", False):
            raise ValueError("Material friction_combine_mode='multiply' requires MuJoCo Warp (use_mujoco_cpu=False).")
        super().__init__(model, **kwargs)
        if self._multiply_friction:
            pairs = np.sort(np.stack((self.mj_model.pair_geom1, self.mj_model.pair_geom2)), axis=0)
            keys = np.sort(pairs[0].astype(np.int64) * self.mj_model.ngeom + pairs[1])
            self._explicit_friction_pairs = wp.array(keys, dtype=wp.int64, device=model.device)
            self.mjw_model.callback.contactfilter = self._update_contact_friction

    def _update_contact_friction(self, model, data, contacts: Contacts | None = None) -> None:
        """Apply the selected rule after collision, before constraint construction."""
        wp.launch(
            _multiply_contact_friction,
            dim=min(contacts.rigid_contact_max, data.naconmax) if contacts is not None else data.naconmax,
            inputs=[
                contacts.rigid_contact_count if contacts is not None else data.nacon,
                self._contact_tid_to_cid if contacts is not None else None,
                contacts.rigid_contact_friction if contacts is not None and contacts.rigid_contact_stiffness else None,
                data.contact.geom,
                data.contact.worldid,
                data.contact.friction,
                self.mjc_geom_to_newton_shape,
                self.model.isaaclab.friction_combine_mode,
                model.geom_priority,
                model.geom_friction,
                self._explicit_friction_pairs if contacts is None else None,
            ],
            device=self.model.device,
        )

    def _convert_contacts_to_mjwarp(self, model: Model, state_in: State, contacts: Contacts):
        super()._convert_contacts_to_mjwarp(model, state_in, contacts)
        if self._multiply_friction:
            # Recompute from geom coefficients so cached contacts do not compound
            # the product across substeps, and runtime material DR stays effective.
            self._update_contact_friction(self.mjw_model, self.mjw_data, contacts)


class MJWarpSolver(NewtonSolver):
    """Newton's MuJoCo Warp solver, which steps in place on one state."""

    solver_class = SolverMuJoCo
    single_state = True
    deterministic = True

    @classmethod
    def applied_deterministic_mode(cls, cfg: MJWarpSolverCfg, mode: wp.DeterministicMode) -> wp.DeterministicMode:
        """MuJoCo on the CPU is reproducible on its own and Warp's mode does not reach it, so none applies."""
        if not cfg.use_mujoco_cpu:
            return mode
        if mode != wp.DeterministicMode.NOT_GUARANTEED:
            logger.info("MuJoCo CPU backend is already reproducible; Newton's deterministic mode is not applied.")
        return wp.DeterministicMode.NOT_GUARANTEED

    @classmethod
    def coupling_error(cls, cfg: MJWarpSolverCfg, in_place: bool) -> str | None:
        """MuJoCo on the CPU resets one global state, which a coupled entry cannot mask per world."""
        if cfg.use_mujoco_cpu:
            return (
                "enables MJWarpSolverCfg.use_mujoco_cpu, whose global reset state cannot yet preserve the per-world"
                " reset-mask lifecycle inside a coupled entry"
            )
        return None

    @classmethod
    def build(
        cls, cfg: MJWarpSolverCfg, model: Model, mode: wp.DeterministicMode, env_template: str | None = None
    ) -> SolverMuJoCo:
        """Construct MuJoCo Warp; a determinism guarantee disables its sensor stage, which cannot honor it."""
        deterministic = mode != wp.DeterministicMode.NOT_GUARANTEED
        blocked = set(model.get_requested_state_attributes()) & _SENSORS_BY_STATE_ATTRIBUTE.keys()
        if deterministic and blocked:
            sensors = sorted({_SENSORS_BY_STATE_ATTRIBUTE[attr] for attr in blocked})
            raise ValueError(
                f"This task does not support deterministic physics: it uses {' and '.join(sensors)},"
                f" reading {sorted(blocked)}. Those attributes come from MuJoCo's post-constraint pass,"
                " which runs inside the sensor stage that a determinism guarantee must disable, so the"
                " values would never be refreshed. Remove the sensors, or drop the determinism request"
                f" (deterministic_mode={mode.name})."
            )
        modes = getattr(getattr(model, "isaaclab", None), "friction_combine_mode", None)
        solver_class = MaterialSolverMuJoCo if modes is not None and np.any(modes.numpy() == 2) else cls.solver_class
        kwargs = solver_kwargs(solver_class, cfg, mode)
        # ls_parallel is deprecated in newton; forwarding it (even as False) emits a warning.
        kwargs.pop("ls_parallel", None)
        if deterministic:
            kwargs["disable_sensors"] = True
        return solver_class(model, **kwargs)

    @classmethod
    def supports_capture(cls, cfg: MJWarpSolverCfg, solver: SolverMuJoCo) -> bool:
        return not cfg.use_mujoco_cpu

    @classmethod
    def uses_collision_pipeline(cls, cfg: MJWarpSolverCfg, solver: SolverMuJoCo) -> bool:
        return not cfg.use_mujoco_contacts

    @classmethod
    def create_contacts(cls, cfg: MJWarpSolverCfg, solver: SolverMuJoCo) -> Contacts:
        """Allocate contacts sized to MuJoCo's contact buffer; ``update_contacts`` fills them for sensors."""
        model = solver.model
        return Contacts(
            rigid_contact_max=solver.get_max_contact_count(),
            soft_contact_max=0,
            device=model.device,
            requested_attributes=model.get_requested_contact_attributes(),
        )

    @classmethod
    def reset(cls, cfg: MJWarpSolverCfg, solver: SolverMuJoCo, state: State, world_mask: wp.array) -> None:
        """Clear MuJoCo Warp warm-start and applied-force buffers for masked worlds.

        With ``flags=0`` MuJoCo zeroes only solver-owned buffers that persist across steps, leaving the authored joint
        state untouched; otherwise a NaN from one solve persists across resets through the warm start
        (https://github.com/newton-physics/newton/issues/1266). The CPU backend's one global ``MjData`` resets every
        world, so it resets only when a world is flagged.
        """
        if solver.use_mujoco_cpu and not world_mask.numpy().any():
            return
        solver.reset(state, world_mask=world_mask, flags=0)

    @classmethod
    def check_step(cls, cfg: MJWarpSolverCfg, solver: SolverMuJoCo, captured: bool, debug: bool) -> None:
        """Log MuJoCo solver convergence statistics in debug mode."""
        if not debug:
            return
        niter = solver.mjw_data.solver_niter.numpy()
        data = {"max": np.max(niter), "mean": np.mean(niter), "min": np.min(niter), "std": np.std(niter)}
        logger.info(f"Solver convergence data: {data}")
        if np.max(niter) == solver.mjw_model.opt.iterations:
            logger.warning(f"Solver didn't converge! max_iter={np.max(niter)}")

    @classmethod
    def fixed_tendon_control(cls, cfg: MJWarpSolverCfg, articulation, model: Model) -> MjWarpTendonControl | None:
        """Build the MuJoCo tendon adapter, or ``None`` when no MuJoCo actuator drives the articulation's tendons."""
        from .mjwarp_tendon_control import create_mjwarp_tendon_control

        return create_mjwarp_tendon_control(articulation, model)
