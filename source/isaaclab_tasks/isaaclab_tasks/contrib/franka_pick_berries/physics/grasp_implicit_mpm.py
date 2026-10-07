# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton's implicit MPM with clamped finger grasping and a tissue damage model.

:class:`SolverGraspImplicitMPM` extends :class:`~newton.solvers.SolverImplicitMPM` for gripping soft tissue:

* **Clamped grasping.** In Newton's implicit MPM, the stress with which a pinched solid pushes back on two fingers
  drains away while the fingers hold it, even though its elastic deformation is kept: a berry held only by friction
  slips out as soon as the hand carries it. The solver holds the berry the way Newton drives solids through
  prescribed particles (see ``example_mpm_beam_twist``): while grasping is on, tissue particles that the solve presses
  against a gripping body are clamped to it. A clamped particle has zero density, which Newton treats as a kinematic
  boundary condition, and its position, velocity and velocity gradient follow the body's rigid motion. The rest of
  the tissue hangs from them and deforms as MPM. Turning grasping off releases every clamp.
* **Damage.** Newton's implicit MPM has plasticity but no damage model. After each step, the solver accumulates a
  plastic strain history from the strain rate while a particle yields, and from it damage and tearing, plus a bruise
  dose from sustained compression; damage and tearing weaken the particle's shear and tensile yield limits. Stress is
  evaluated from the stored elastic deformation with the tissue's elastic law, not read from the solver's stress
  output: at a constant deformation, that output drains away over the steps by an amount that depends on the step
  size, while the elastic deformation is kept.

Clamping needs a particle collider basis (``"pic"`` or ``"picN"``), where every tissue particle is a collider node.
The tissue's yield limits and damage parameters are set per body of tissue with
:meth:`SolverGraspImplicitMPM.set_tissue`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import newton
import numpy as np
import warp as wp
import warp.fem as fem
from isaaclab_newton.physics import MPMSolverCfg, NewtonMPMManager
from newton.solvers import SolverImplicitMPM

from isaaclab.utils import configclass

from .materials import YOUNG_MODULUS

_FREE = -1


# --------------------------------------------------------------------------------------------------------------------
# Clamped grasping
# --------------------------------------------------------------------------------------------------------------------


@wp.kernel
def drive_clamped_particles(
    grasping: wp.array[int],
    clamp_body: wp.array[int],
    anchor: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_volume: wp.array[float],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_qd_grad: wp.array[wp.mat33],
    particle_density: wp.array[float],
):
    """Move clamped particles rigidly with their finger, as kinematic particles; release them all when not grasping."""
    particle = wp.tid()
    body = clamp_body[particle]
    if body == _FREE:
        return
    if grasping[0] == 0:
        clamp_body[particle] = _FREE
        particle_density[particle] = particle_mass[particle] / particle_volume[particle]
        return
    pose = body_q[body]
    position = wp.transform_point(pose, anchor[particle])
    angular = wp.spatial_bottom(body_qd[body])
    particle_q[particle] = position
    particle_qd[particle] = wp.spatial_top(body_qd[body]) + wp.cross(
        angular, position - wp.transform_point(pose, body_com[body])
    )
    particle_qd_grad[particle] = wp.skew(angular)
    particle_density[particle] = 0.0


@wp.kernel
def release_clamps(
    particle_mass: wp.array[float],
    particle_volume: wp.array[float],
    clamp_body: wp.array[int],
    particle_density: wp.array[float],
):
    """Release every clamped particle and restore its density."""
    particle = wp.tid()
    if clamp_body[particle] != _FREE:
        clamp_body[particle] = _FREE
        particle_density[particle] = particle_mass[particle] / particle_volume[particle]


@wp.kernel
def clamp_pressed_particles(
    grasping: wp.array[int],
    space_nodes: wp.array[int],
    cell_particles: wp.array[int],
    node_positions: wp.array[wp.vec3],
    node_normals: wp.array[wp.vec3],
    node_colliders: wp.array[int],
    node_impulse: wp.array[wp.vec3],
    collider_body: wp.array[int],
    gripping_body: wp.array[int],
    body_q: wp.array[wp.transform],
    clamp_body: wp.array[int],
    anchor: wp.array[wp.vec3],
):
    """Clamp particles that the solve pressed against a gripping body, at their current place on it."""
    node = wp.tid()
    if grasping[0] == 0 or node_positions[node][0] == fem.OUTSIDE:
        # Not grasping, or an unused capacity slot that does not map to a particle.
        return
    collider = node_colliders[node]
    if collider < 0:
        return
    body = collider_body[collider]
    if body < 0 or gripping_body[body] == 0:
        return
    if wp.dot(node_impulse[node], node_normals[node]) <= 0.0:
        return
    # A particle collider basis: the node is an evaluation point of the particle quadrature.
    particle = cell_particles[space_nodes[node]]
    if clamp_body[particle] == _FREE:
        clamp_body[particle] = body
        anchor[particle] = wp.transform_point(wp.transform_inverse(body_q[body]), node_positions[node])


# --------------------------------------------------------------------------------------------------------------------
# Damage
# --------------------------------------------------------------------------------------------------------------------


@wp.func
def principal_stress(elastic: wp.mat33, young: float, poisson: float):
    """Principal stresses [Pa] of an elastic deformation gradient, linear elasticity in its principal stretches."""
    _u, stretch, _v = wp.svd3(elastic)
    strain = stretch - wp.vec3(1.0)
    lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    shear = young / (2.0 * (1.0 + poisson))
    return wp.vec3(lame * (strain[0] + strain[1] + strain[2])) + 2.0 * shear * strain


@wp.func
def von_mises(principal: wp.vec3):
    a = principal[0] - principal[1]
    b = principal[1] - principal[2]
    c = principal[2] - principal[0]
    return wp.sqrt(0.5 * (a * a + b * b + c * c))


@wp.struct
class DamageParameters:
    yielding_fraction: float
    strain_rate_scale: float
    onset: wp.vec2
    interval: wp.vec2
    softening: float
    tensile_ratio: float
    bruise_stress: float
    bruise_rate: float
    max_rate: float


@wp.kernel
def update_damage(
    velocity_gradient: wp.array[wp.mat33],
    elastic_strain: wp.array[wp.mat33],
    young: wp.array[float],
    poisson: wp.array[float],
    interface: wp.array[int],
    base_shear: wp.array[float],
    tear_onset: wp.array[float],
    tear_end: wp.array[float],
    clamp_body: wp.array[int],
    dt: float,
    config: DamageParameters,
    history: wp.array[float],
    damage: wp.array[float],
    tear: wp.array[float],
    dose: wp.array[float],
    yield_stress: wp.array[float],
    tensile_ratio: wp.array[float],
):
    """Advance damage, tearing and bruising by ``dt`` [s] and weaken the particle's yield limits."""
    p = wp.tid()
    if clamp_body[p] != _FREE:
        # A clamped particle's motion is prescribed by its body, not by the tissue's deformation.
        return
    principal = principal_stress(elastic_strain[p], young[p], poisson[p])

    # Strain history, standing in for accumulated plastic strain, which Newton does not store: once a particle sits on
    # its yield surface, all further deviatoric strain is plastic, so integrate the strain rate while yielding.
    sym = 0.5 * (velocity_gradient[p] + wp.transpose(velocity_gradient[p]))
    rate = sym - (wp.trace(sym) / 3.0) * wp.identity(3, dtype=float)
    if von_mises(principal) >= config.yielding_fraction * yield_stress[p]:
        history[p] += config.strain_rate_scale * dt * wp.sqrt(2.0 / 3.0 * wp.ddot(rate, rate))

    # Bruising: a dose accumulates while the largest principal compression exceeds the threshold.
    compression = wp.max(-principal[0], wp.max(-principal[1], -principal[2]))
    excess = wp.max(compression / config.bruise_stress - 1.0, 0.0)
    dose[p] += dt * config.bruise_rate * excess * excess

    # Damage and tearing tend to their strain-history and dose targets at a bounded rate (delay damage), which keeps
    # the softening they cause from running away within a few steps.
    side = wp.where(interface[p] != 0, 1, 0)
    target = wp.clamp((history[p] - config.onset[side]) / config.interval[side], 0.0, 1.0)
    target = wp.max(target, 1.0 - wp.exp(-dose[p]))
    damage[p] = wp.max(damage[p], wp.min(target, damage[p] + config.max_rate * dt))
    if tear_end[p] > tear_onset[p]:
        failure = wp.clamp((history[p] - tear_onset[p]) / (tear_end[p] - tear_onset[p]), 0.0, 1.0)
        tear[p] = wp.max(tear[p], wp.min(failure * failure * (3.0 - 2.0 * failure), tear[p] + config.max_rate * dt))

    # Damaged tissue is weaker in shear; torn tissue loses tensile capacity but keeps supporting compression.
    yield_stress[p] = base_shear[p] * (1.0 - config.softening * damage[p])
    tensile_ratio[p] = config.tensile_ratio * (1.0 - 0.98 * tear[p])


@wp.kernel
def restore_material(
    base_shear: wp.array[float],
    tensile: float,
    yield_stress: wp.array[float],
    tensile_ratio: wp.array[float],
):
    p = wp.tid()
    yield_stress[p] = base_shear[p]
    tensile_ratio[p] = tensile


@wp.kernel
def set_range(
    start: int,
    interface: wp.array[int],
    shear: float,
    pressure: float,
    tensile: float,
    tear_onset_value: float,
    tear_end_value: float,
    interface_out: wp.array[int],
    base_shear: wp.array[float],
    tear_onset: wp.array[float],
    tear_end: wp.array[float],
    yield_stress: wp.array[float],
    yield_pressure: wp.array[float],
    tensile_ratio: wp.array[float],
):
    i = wp.tid()
    p = start + i
    interface_out[p] = interface[i]
    base_shear[p] = shear
    tear_onset[p] = tear_onset_value
    tear_end[p] = tear_end_value
    yield_stress[p] = shear
    yield_pressure[p] = pressure
    tensile_ratio[p] = tensile


@dataclass
class DamageConfig:
    """Damage model parameters shared by all tissue."""

    yielding_fraction: float = 0.9
    """Von Mises stress, relative to the current shear yield stress, above which a particle counts as yielding."""
    strain_rate_scale: float = 0.25
    """Scale of the integrated plastic strain rate, calibrated so that a full squash reaches a mean damage comparable
    to the explicit solver's."""
    onset: tuple[float, float] = (0.5, 0.08)
    """Strain history at which damage starts, in bulk tissue and at interfaces."""
    interval: tuple[float, float] = (1.0, 0.22)
    """Further history over which damage reaches 1, in bulk tissue and at interfaces."""
    softening: float = 0.25
    """Fraction of the shear yield stress that full damage removes."""
    tensile_ratio: float = 1.0
    """Tensile yield ratio of intact tissue."""
    bruise_stress: float = 4500.0
    """Principal compression [Pa] above which a bruise dose accumulates; a gentle grasp peaks near 2 kPa."""
    bruise_rate: float = 8.0
    """Bruise dose rate [1/s] at twice the threshold compression."""
    max_rate: float = 1.0
    """Largest rate [1/s] at which damage and tearing grow (delay damage, after Allix and Deu 1997). Without it, the
    softening that damage causes feeds back into more yielding within a few steps and the solve diverges."""


@dataclass
class TissueMaterial:
    """Implicit-solver yield limits and tearing of one berry species."""

    shear_yield: float
    """Shear yield stress [Pa]."""
    pressure_yield: float
    """Compressive yield pressure [Pa]."""
    tear_onset: float
    tear_end: float


def tissue_material(profile: dict) -> TissueMaterial:
    """Return the implicit-solver material of a berry from its asset profile.

    Compression and shear stay elastic through handling loads; tearing applies to species whose asset enables it.
    """
    young = YOUNG_MODULUS[profile["berry"]]
    tears = profile["simulation"].get("tear_end", 0) > 0
    return TissueMaterial(
        shear_yield=0.5 * young,
        pressure_yield=5.0 * young,
        tear_onset=2.5 if tears else 0.0,
        tear_end=5.0 if tears else 0.0,
    )


# --------------------------------------------------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------------------------------------------------


class SolverGraspImplicitMPM(SolverImplicitMPM):
    """:class:`~newton.solvers.SolverImplicitMPM` with clamped grasping and damage (see the module docstring)."""

    def __init__(self, model: newton.Model, config: SolverImplicitMPM.Config, **kwargs):
        if not config.collider_basis.startswith("pic"):
            raise ValueError(f"SolverGraspImplicitMPM needs a particle collider basis, not {config.collider_basis!r}")
        super().__init__(model, config, **kwargs)
        device = model.device
        count = model.particle_count
        self.grasping = wp.zeros(1, dtype=int, device=device)
        """Whether clamping is on (1) or off (0); set with :meth:`set_grasping`."""
        self.clamp_body = wp.full(count, _FREE, dtype=int, device=device)
        """Body each particle is clamped to, or -1."""
        self.anchor = wp.zeros(count, dtype=wp.vec3, device=device)
        """Clamped position of each particle in its body's frame [m]."""
        self.gripping_body = wp.zeros(model.body_count, dtype=int, device=device)
        """Whether each body clamps tissue (1) or not (0); set with :meth:`set_gripping_bodies`."""
        self._pic = None
        self.damage_config = DamageConfig()
        """Damage model parameters; set before the first step."""
        with wp.ScopedDevice(device):
            self.interface = wp.zeros(count, dtype=int)
            self.base_shear = wp.zeros(count, dtype=float)
            self.tear_onset = wp.zeros(count, dtype=float)
            self.tear_end = wp.zeros(count, dtype=float)
            self.history = wp.zeros(count, dtype=float)
            self.damage = wp.zeros(count, dtype=float)
            self.tear = wp.zeros(count, dtype=float)
            self.dose = wp.zeros(count, dtype=float)

    def set_gripping_bodies(self, pattern: str) -> None:
        """Select the bodies that clamp tissue: those whose label fully matches the regular expression."""
        matches = np.array([re.fullmatch(pattern, label) is not None for label in self.model.body_label], np.int32)
        if not matches.any():
            raise ValueError(f"No body label matches {pattern!r}")
        self.gripping_body.assign(matches)

    def set_grasping(self, grasping: bool) -> None:
        """Turn clamping on or off; turning it off releases every clamped particle on the next step.

        Call outside CUDA graph capture; captured steps read the value when they run. :attr:`grasping` can also be
        written on the device.
        """
        self.grasping.fill_(int(grasping))

    def set_tissue(self, start: int, interface: np.ndarray, field: int, material: TissueMaterial) -> None:
        """Set the material of a body of tissue: the particles from ``start`` on, one per ``interface`` flag.

        Args:
            start: Index of the body's first particle in the model.
            interface: Nonzero for particles on a weak internal interface, where damage starts earlier.
            field: Unused: all tissue shares one velocity field.
            material: The body's yield limits and tearing.
        """
        del field
        mpm = self.model.mpm
        flags = wp.array(np.asarray(interface, np.int32), dtype=int, device=self.model.device)
        wp.launch(
            set_range,
            dim=len(flags),
            inputs=[
                start,
                flags,
                material.shear_yield,
                material.pressure_yield,
                self.damage_config.tensile_ratio,
                material.tear_onset,
                material.tear_end,
            ],
            outputs=[
                self.interface,
                self.base_shear,
                self.tear_onset,
                self.tear_end,
                mpm.yield_stress,
                mpm.yield_pressure,
                mpm.tensile_yield_ratio,
            ],
            device=self.model.device,
        )

    def damage_view(self, start: int, count: int) -> dict[str, wp.array]:
        """Damage arrays of ``count`` particles from ``start``: ``damage``, ``tear``, ``history`` and ``dose``."""
        span = slice(start, start + count)
        return {
            "damage": self.damage[span],
            "tear": self.tear[span],
            "history": self.history[span],
            "dose": self.dose[span],
        }

    def elastic_strain(self, state: newton.State, start: int, count: int) -> wp.array:
        """Elastic deformation gradients of ``count`` particles from ``start`` in ``state``."""
        return state.mpm.particle_elastic_strain[start : start + count]

    def reset(self, state, world_mask=None, flags=None):
        super().reset(state, world_mask=world_mask, flags=flags)
        # Clamps and damage are part of the particles' history.
        wp.launch(
            release_clamps,
            dim=self.model.particle_count,
            inputs=[self.model.particle_mass, self._mpm_model.particle_volume],
            outputs=[self.clamp_body, self._mpm_model.particle_density],
            device=self.model.device,
        )
        for array in (self.history, self.damage, self.tear, self.dose):
            array.zero_()
        wp.launch(
            restore_material,
            dim=self.model.particle_count,
            inputs=[self.base_shear, self.damage_config.tensile_ratio],
            outputs=[self.model.mpm.yield_stress, self.model.mpm.tensile_yield_ratio],
            device=self.model.device,
        )

    def step(self, state_in, state_out, control, contacts, dt):
        model = self.model
        wp.launch(
            drive_clamped_particles,
            dim=model.particle_count,
            inputs=[
                self.grasping,
                self.clamp_body,
                self.anchor,
                state_in.body_q,
                state_in.body_qd,
                model.body_com,
                model.particle_mass,
                self._mpm_model.particle_volume,
            ],
            outputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.mpm.particle_qd_grad,
                self._mpm_model.particle_density,
            ],
            device=model.device,
        )
        super().step(state_in, state_out, control, contacts, dt)
        config = self.damage_config
        parameters = DamageParameters()
        parameters.yielding_fraction = config.yielding_fraction
        parameters.strain_rate_scale = config.strain_rate_scale
        parameters.onset = wp.vec2(*config.onset)
        parameters.interval = wp.vec2(*config.interval)
        parameters.softening = config.softening
        parameters.tensile_ratio = config.tensile_ratio
        parameters.bruise_stress = config.bruise_stress
        parameters.bruise_rate = config.bruise_rate
        parameters.max_rate = config.max_rate
        # The yield limits written here are the solver's own material arrays, read directly by the next step.
        wp.launch(
            update_damage,
            dim=model.particle_count,
            inputs=[
                state_out.mpm.particle_qd_grad,
                state_out.mpm.particle_elastic_strain,
                model.mpm.young_modulus,
                model.mpm.poisson_ratio,
                self.interface,
                self.base_shear,
                self.tear_onset,
                self.tear_end,
                self.clamp_body,
                dt,
                parameters,
            ],
            outputs=[
                self.history,
                self.damage,
                self.tear,
                self.dose,
                model.mpm.yield_stress,
                model.mpm.tensile_yield_ratio,
            ],
            device=model.device,
        )

    def _step_impl(self, state_in, state_out, dt, pic, scratch):
        # The collider nodes are the particles of this step's quadrature.
        self._pic = pic
        super()._step_impl(state_in, state_out, dt, pic, scratch)

    def _update_particles(self, state_in, state_out, dt, pic, scratch):
        wp.launch(
            clamp_pressed_particles,
            dim=scratch.collider_node_count,
            inputs=[
                self.grasping,
                scratch.impulse_field.space_partition.space_node_indices(),
                self._pic.cell_particle_indices,
                scratch.collider_position_field.dof_values,
                scratch.collider_normal_field.dof_values,
                scratch.collider_ids,
                scratch.impulse_field.dof_values,
                self._mpm_model.collider.collider_body_index,
                self.gripping_body,
                state_in.body_q,
            ],
            outputs=[self.clamp_body, self.anchor],
        )
        super()._update_particles(state_in, state_out, dt, pic, scratch)


# --------------------------------------------------------------------------------------------------------------------
# Isaac Lab integration
# --------------------------------------------------------------------------------------------------------------------


class GraspImplicitMPMManager(NewtonMPMManager):
    """:class:`~isaaclab_newton.physics.NewtonMPMManager` that builds a :class:`SolverGraspImplicitMPM`."""

    solver_class = SolverGraspImplicitMPM

    @classmethod
    def _create_solver(cls, model, solver_cfg: GraspImplicitMPMSolverCfg) -> SolverGraspImplicitMPM:
        solver = super()._create_solver(model, solver_cfg)
        solver.set_gripping_bodies(solver_cfg.gripping_bodies)
        return solver


@configclass
class GraspImplicitMPMSolverCfg(MPMSolverCfg):
    """Configuration of :class:`SolverGraspImplicitMPM`, with defaults tuned for berry tissue."""

    class_type: type = GraspImplicitMPMManager

    gripping_bodies: str = ".*"
    """Regular expression fully matching the labels of the bodies that clamp tissue."""

    grid_type: str = "sparse"
    grid_padding: int = 0
    strain_basis: str = "P1d"
    """P1d strain with many iterations resolves the elastic response; P0 or few iterations leave the tissue far softer
    than its Young's modulus."""
    transfer_scheme: str = "apic"
    max_iterations: int = 100
    """Each solve is the dominant cost of a step. 50 iterations leave a carried berry visibly softer (it stretches
    out of shape)."""
    tolerance: float = 1.0e-5
    warmstart_mode: str = "auto"
    velocity_basis: str = "Q1"
    collider_basis: str = "pic27"
    """Collider basis; must be a particle basis, so that every particle is a collider node."""
    collider_velocity_mode: str = "forward"
    solver: str = "jacobi"
    separate_worlds: bool = True
