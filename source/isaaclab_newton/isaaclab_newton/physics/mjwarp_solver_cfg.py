# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton physics manager."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, ClassVar

from isaaclab.utils import configclass
from isaaclab.utils.string import string_to_callable

from .newton_manager_cfg import NewtonSolverCfg, solver_kwargs

if TYPE_CHECKING:
    import warp as wp
    from newton import Contacts, Model, State
    from newton.solvers import SolverMuJoCo

    from .mjwarp_tendon_control import MjWarpTendonControl

logger = logging.getLogger(__name__)

_SENSORS_BY_STATE_ATTRIBUTE = {"body_qdd": "the IMU or PVA sensor", "body_parent_f": "the joint-wrench sensor"}


@configclass
class MJWarpSolverCfg(NewtonSolverCfg):
    """Configuration for MuJoCo Warp solver-related parameters.

    These parameters are used to configure the MuJoCo Warp solver. For more information, see the
    `MuJoCo Warp documentation`_.

    .. _MuJoCo Warp documentation: https://github.com/google-deepmind/mujoco_warp
    """

    solver_class: str = "newton.solvers:SolverMuJoCo"

    single_state: ClassVar[bool] = True
    deterministic: ClassVar[bool] = True

    solver_type: str = "mujoco_warp"
    """Solver type. Can be "mujoco_warp"."""

    njmax: int = 300
    """Number of constraints per environment (world)."""

    nconmax: int | None = None
    """Number of contact points per environment (world)."""

    iterations: int = 100
    """Number of solver iterations."""

    ls_iterations: int = 50
    """Number of line search iterations for the solver."""

    solver: str = "newton"
    """Solver type. Can be "cg" or "newton", or their corresponding MuJoCo integer constants."""

    integrator: str = "euler"
    """Integrator type. Can be "euler", "rk4", or "implicitfast", or their corresponding MuJoCo integer constants."""

    use_mujoco_cpu: bool = False
    """Whether to use the pure MuJoCo backend instead of `mujoco_warp`."""

    disable_contacts: bool = False
    """Whether to disable contact computation in MuJoCo."""

    disable_sensors: bool = False
    """Whether to disable MuJoCo Warp's internal sensor computation.

    A determinism guarantee (:attr:`NewtonCfg.deterministic_mode`) disables it automatically. Isaac Lab sensors use
    Newton state directly and do not depend on MuJoCo Warp's internal sensor data.
    """

    default_actuator_gear: float | None = None
    """Default gear ratio for all actuators."""

    actuator_gears: dict[str, float] | None = None
    """Dictionary mapping joint names to specific gear ratios, overriding the `default_actuator_gear`."""

    update_data_interval: int = 1
    """Frequency (in simulation steps) at which to update the MuJoCo Data object from the Newton state.

    If 0, Data is never updated after initialization.
    """

    save_to_mjcf: str | None = None
    """Optional path to save the generated MJCF model file.

    If None, the MJCF model is not saved.
    """

    impratio: float = 1.0
    """Frictional-to-normal constraint impedance ratio."""

    cone: str = "pyramidal"
    """The type of contact friction cone. Can be "pyramidal" or "elliptic"."""

    ccd_iterations: int = 35
    """Maximum iterations for convex collision detection (GJK/EPA).

    Increase this if you see warnings about ``opt.ccd_iterations`` needing to be increased,
    which typically occurs with complex collision geometries (e.g. multi-finger hands).
    """

    enable_multiccd: bool = False
    """Whether to enable multiple-contact convex collision detection. Defaults to False.

    With :attr:`use_mujoco_contacts` enabled, supported convex geometry pairs can generate
    a contact manifold (several points across a touching surface) instead of a single point.
    This can improve stability for flat mesh contacts in stacking and grasping. It does not
    enable continuous collision detection or prevent tunneling between simulation steps.

    MuJoCo Warp's supported pairs and contact-margin restrictions differ from MuJoCo CPU.
    Primitive colliders can already produce multiple contacts with this option disabled.
    Additional contacts can increase solver work and the required :attr:`nconmax` and
    :attr:`njmax` capacities. See MuJoCo's `multiple-contact documentation
    <https://mujoco.readthedocs.io/en/stable/computation/index.html#multiple-contacts>`__.
    """

    ls_parallel: bool = False
    """Deprecated parallel line search option.

    Setting this to ``True`` emits a :class:`DeprecationWarning` and is ignored.
    MuJoCo Warp is dropping support for parallel line search; Isaac Lab uses
    iterative line search for performance.
    """

    use_mujoco_contacts: bool = True
    """Whether to use MuJoCo's internal contact solver.

    If ``True`` (default), MuJoCo handles collision detection and contact resolution internally.
    If ``False``, Newton's :class:`CollisionPipeline` is used instead.  A default pipeline
    (``broad_phase="explicit"``) is created automatically when :attr:`NewtonCfg.collision_cfg`
    is ``None``.  Set :attr:`NewtonCfg.collision_cfg` to a :class:`NewtonCollisionPipelineCfg`
    to customize pipeline parameters (broad phase, contact limits, hydroelastic, etc.).

    .. note::
        Setting ``collision_cfg`` while ``use_mujoco_contacts=True`` raises
        :class:`ValueError` because the two collision modes are mutually exclusive.
    """

    tolerance: float = 1e-6
    """Solver convergence tolerance for the constraint residual.

    The solver iterates until the residual drops below this threshold or
    ``iterations`` is reached.  Lower values give more precise constraint
    satisfaction at the cost of more iterations.  MuJoCo default is ``1e-8``;
    Newton default is ``1e-6``.
    """

    def __post_init__(self):
        if self.ls_parallel:
            warnings.warn(
                "MJWarpSolverCfg.ls_parallel is deprecated and ignored. "
                "Isaac Lab uses iterative line search for performance because "
                "MuJoCo Warp is dropping parallel line search support. Tune "
                "MJWarpSolverCfg.ls_iterations instead.",
                DeprecationWarning,
                stacklevel=5,
            )
            self.ls_parallel = False

    def build(self, model: Model, mode: wp.DeterministicMode, env_template: str | None = None) -> SolverMuJoCo:
        """Construct MuJoCo Warp; a determinism guarantee disables its sensor stage, which cannot honor it."""
        import warp as wp

        solver_cls = string_to_callable(self.solver_class)
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
        kwargs = solver_kwargs(solver_cls, self, mode)
        # ls_parallel is deprecated in newton; forwarding it (even as False) emits a warning.
        kwargs.pop("ls_parallel", None)
        if deterministic:
            kwargs["disable_sensors"] = True
        return solver_cls(model, **kwargs)

    def applied_deterministic_mode(self, mode: wp.DeterministicMode) -> wp.DeterministicMode:
        """MuJoCo on the CPU is reproducible on its own and Warp's mode does not reach it, so none applies."""
        import warp as wp

        if not self.use_mujoco_cpu:
            return mode
        if mode != wp.DeterministicMode.NOT_GUARANTEED:
            logger.info("MuJoCo CPU backend is already reproducible; Newton's deterministic mode is not applied.")
        return wp.DeterministicMode.NOT_GUARANTEED

    def supports_capture(self, solver: SolverMuJoCo) -> bool:
        return not self.use_mujoco_cpu

    def coupling_error(self, in_place: bool) -> str | None:
        """MuJoCo on the CPU resets one global state, which a coupled entry cannot mask per world."""
        if self.use_mujoco_cpu:
            return (
                "enables MJWarpSolverCfg.use_mujoco_cpu, whose global reset state cannot yet preserve the per-world"
                " reset-mask lifecycle inside a coupled entry"
            )
        return None

    def uses_collision_pipeline(self, solver: SolverMuJoCo) -> bool:
        return not self.use_mujoco_contacts

    def create_contacts(self, solver: SolverMuJoCo) -> Contacts:
        """Allocate contacts sized to MuJoCo's contact buffer; ``solver.update_contacts`` fills them for sensors."""
        from newton import Contacts

        model = solver.model
        return Contacts(
            rigid_contact_max=solver.get_max_contact_count(),
            soft_contact_max=0,
            device=model.device,
            requested_attributes=model.get_requested_contact_attributes(),
        )

    def reset(self, solver: SolverMuJoCo, state: State, world_mask: wp.array) -> None:
        """Clear MuJoCo Warp warm-start and applied-force buffers for masked worlds.

        With ``flags=0`` MuJoCo zeroes only solver-owned buffers that persist across steps, leaving the authored joint
        state untouched; otherwise a NaN from one solve persists across resets through the warm start
        (https://github.com/newton-physics/newton/issues/1266). The CPU backend's one global ``MjData`` resets every
        world, so it resets only when a world is flagged.
        """
        if solver.use_mujoco_cpu and not world_mask.numpy().any():
            return
        solver.reset(state, world_mask=world_mask, flags=0)

    def check_step(self, solver: SolverMuJoCo, captured: bool, debug: bool) -> None:
        """Log MuJoCo solver convergence statistics in debug mode."""
        if not debug:
            return
        import numpy as np

        niter = solver.mjw_data.solver_niter.numpy()
        data = {"max": np.max(niter), "mean": np.mean(niter), "min": np.min(niter), "std": np.std(niter)}
        logger.info(f"Solver convergence data: {data}")
        if np.max(niter) == solver.mjw_model.opt.iterations:
            logger.warning(f"Solver didn't converge! max_iter={np.max(niter)}")

    def fixed_tendon_control(self, articulation, model: Model) -> MjWarpTendonControl | None:
        """Build the MuJoCo tendon adapter, or ``None`` when no MuJoCo actuator drives the articulation's tendons."""
        from .mjwarp_tendon_control import create_mjwarp_tendon_control

        return create_mjwarp_tendon_control(articulation, model)
