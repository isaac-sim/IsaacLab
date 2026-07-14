# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic retention helpers for the fixed native needle grasps.

These routines consume only pinned generator output and declared physical
constants. They never inspect a live trajectory, and therefore cannot turn the
reset into an adaptive or scripted grasp.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt

ArrayLike = npt.ArrayLike

FORCE_CLOSURE_CONE_FACETS = 8
"""Fixed facet count used by the deterministic static-friction proof."""

EXACT_POINT_CONTACT_FORCE_RESIDUAL_TOLERANCE_N = 1.0e-10
"""Fixed numerical force tolerance for exact point-contact feasibility."""

EXACT_POINT_CONTACT_MOMENT_ARM_TOLERANCE_M = 1.0e-10
"""Fixed numerical moment-arm tolerance for exact point-contact feasibility."""


def _vector3(value: ArrayLike, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite three-vector")
    return vector


def _unit(value: ArrayLike, name: str) -> np.ndarray:
    vector = _vector3(value, name)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


@dataclass(frozen=True, slots=True)
class RetentionLoad:
    """Analytical two-contact load required by gravity and commanded motion."""

    external_force_n: float
    normal_force_per_jaw_n: float
    friction_coefficient: float
    safety_factor: float


@dataclass(frozen=True, slots=True)
class ForceClosureProof:
    """Deterministic point-contact certificate for one required wrench.

    Force and torque residuals are reported separately because their units
    cannot be combined into one Euclidean norm.  The equivalent moment-arm
    residual divides the torque residual by the required-force norm.  It is
    infinite when a non-zero torque residual has no required force against
    which to normalise.
    """

    exact_point_contact_feasible: bool
    coefficients: tuple[float, ...]
    achieved_wrench: tuple[float, float, float, float, float, float]
    residual_wrench: tuple[float, float, float, float, float, float]
    required_force_norm_n: float
    force_residual_norm_n: float
    torque_residual_norm_n_m: float
    equivalent_moment_arm_residual_m: float
    force_residual_tolerance_n: float
    moment_arm_residual_tolerance_m: float
    active_generator_indices: tuple[int, ...]

    @property
    def feasible(self) -> bool:
        """Return the exact point-contact result for compatibility."""

        return self.exact_point_contact_feasible


@dataclass(frozen=True, slots=True)
class FiniteContactAcceptance:
    """Explicit finite-patch acceptance of an inexact point-contact proof.

    This result is a task-level soft-contact approximation, not exact force
    closure.  Both tolerances must be supplied by the caller.
    """

    accepted: bool
    force_within_tolerance: bool
    moment_arm_within_tolerance: bool
    force_residual_tolerance_n: float
    moment_arm_residual_tolerance_m: float


def required_retention_load(
    *,
    mass_kg: float,
    gravity_m_s2: float,
    maximum_commanded_acceleration_m_s2: float,
    friction_coefficient: float,
    safety_factor: float,
) -> RetentionLoad:
    """Return the per-jaw normal load for an opposed two-contact grasp."""

    values = (
        mass_kg,
        gravity_m_s2,
        maximum_commanded_acceleration_m_s2,
        friction_coefficient,
        safety_factor,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("retention-load inputs must be finite")
    if mass_kg <= 0.0 or gravity_m_s2 < 0.0 or maximum_commanded_acceleration_m_s2 < 0.0:
        raise ValueError("mass must be positive and accelerations non-negative")
    if friction_coefficient <= 0.0 or safety_factor < 1.0:
        raise ValueError("friction must be positive and safety_factor at least one")
    external_force = mass_kg * (gravity_m_s2 + maximum_commanded_acceleration_m_s2)
    normal_force = safety_factor * external_force / (2.0 * friction_coefficient)
    return RetentionLoad(external_force, normal_force, friction_coefficient, safety_factor)


def friction_cone_generators(
    inward_normal: ArrayLike,
    friction_coefficient: float,
    facets: int,
) -> np.ndarray:
    """Return a fixed polygonal approximation of one Coulomb friction cone."""

    normal = _unit(inward_normal, "inward normal")
    if not math.isfinite(friction_coefficient) or friction_coefficient <= 0.0:
        raise ValueError("friction_coefficient must be finite and positive")
    if facets < 4:
        raise ValueError("friction cone requires at least four facets")
    reference = np.array((1.0, 0.0, 0.0))
    if abs(float(reference @ normal)) > 0.9:
        reference = np.array((0.0, 1.0, 0.0))
    tangent_1 = _unit(np.cross(normal, reference), "friction tangent")
    tangent_2 = np.cross(normal, tangent_1)
    generators = []
    for index in range(facets):
        angle = 2.0 * math.pi * index / facets
        tangent = math.cos(angle) * tangent_1 + math.sin(angle) * tangent_2
        generators.append(normal + friction_coefficient * tangent)
    return np.stack(generators)


def two_contact_friction_wrench_generators(
    contact_points_m: ArrayLike,
    inward_normals: ArrayLike,
    static_friction_coefficient: float,
) -> np.ndarray:
    """Return 16 row-wise point-contact wrench generators for two contacts.

    Contact points are expressed relative to the wrench origin.  Each contact
    contributes eight edges of a polygonal Coulomb cone using the declared
    *static* friction coefficient.  A generator is ordered as ``[force,
    moment]``, with ``moment = point x force``.
    """

    points = np.asarray(contact_points_m, dtype=np.float64)
    normals = np.asarray(inward_normals, dtype=np.float64)
    if points.shape != (2, 3) or not np.isfinite(points).all():
        raise ValueError("contact_points_m must be a finite (2, 3) array")
    if normals.shape != (2, 3) or not np.isfinite(normals).all():
        raise ValueError("inward_normals must be a finite (2, 3) array")
    if not math.isfinite(static_friction_coefficient) or static_friction_coefficient <= 0.0:
        raise ValueError("static_friction_coefficient must be finite and positive")

    contact_grasp_matrix = grasp_matrix(points)
    wrenches = []
    for contact_index, normal in enumerate(normals):
        forces = friction_cone_generators(
            normal,
            static_friction_coefficient,
            FORCE_CLOSURE_CONE_FACETS,
        )
        for force in forces:
            padded_force = np.zeros(6, dtype=np.float64)
            padded_force[3 * contact_index : 3 * (contact_index + 1)] = force
            wrenches.append(contact_grasp_matrix @ padded_force)
    return np.stack(wrenches)


def prove_two_contact_force_closure(
    contact_points_m: ArrayLike,
    inward_normals: ArrayLike,
    static_friction_coefficient: float,
    required_wrench: ArrayLike,
) -> ForceClosureProof:
    """Prove whether two static-friction contacts can supply one 6D wrench.

    The proof enumerates generator subsets in stable lexicographic order and
    solves only unconstrained least-squares systems whose coefficients are
    nonnegative.  Conic Caratheodory bounds a certificate to the generator
    matrix rank, which is at most six here, so the search is finite and does
    not require SciPy or an adaptive optimiser.  Exact point-contact
    feasibility requires both the force residual and its equivalent
    moment-arm residual to meet their numerical tolerances.  The conservative
    fixed moment-arm tolerance is 0.1 nanometres, well below 10 micrometres;
    it cannot be loosened into a finite-contact acceptance allowance.  Use
    :func:`assess_finite_contact_acceptance` for that separate judgement.

    Infeasible candidates are ranked by the maximum of their two dimensionless
    tolerance ratios.  No force value is ever added to a torque value.
    """

    target = np.asarray(required_wrench, dtype=np.float64)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError("required_wrench must be a finite six-vector")
    force_residual_tolerance_n = EXACT_POINT_CONTACT_FORCE_RESIDUAL_TOLERANCE_N
    moment_arm_residual_tolerance_m = EXACT_POINT_CONTACT_MOMENT_ARM_TOLERANCE_M

    required_force_norm_n = float(np.linalg.norm(target[:3]))

    def residual_metrics(residual_wrench: np.ndarray) -> tuple[float, float, float]:
        force_residual_norm_n = float(np.linalg.norm(residual_wrench[:3]))
        torque_residual_norm_n_m = float(np.linalg.norm(residual_wrench[3:]))
        if required_force_norm_n > 0.0:
            moment_arm_residual_m = torque_residual_norm_n_m / required_force_norm_n
        elif torque_residual_norm_n_m == 0.0:
            moment_arm_residual_m = 0.0
        else:
            moment_arm_residual_m = math.inf
        return force_residual_norm_n, torque_residual_norm_n_m, moment_arm_residual_m

    def make_proof(
        candidate_coefficients: np.ndarray,
        candidate_achieved: np.ndarray,
        candidate_residual: np.ndarray,
        active_indices: tuple[int, ...],
    ) -> ForceClosureProof:
        force_residual_n, torque_residual_n_m, moment_arm_residual_m = residual_metrics(candidate_residual)
        exact = (
            force_residual_n <= force_residual_tolerance_n and moment_arm_residual_m <= moment_arm_residual_tolerance_m
        )
        return ForceClosureProof(
            exact_point_contact_feasible=exact,
            coefficients=tuple(float(value) for value in candidate_coefficients),
            achieved_wrench=tuple(float(value) for value in candidate_achieved),
            residual_wrench=tuple(float(value) for value in candidate_residual),
            required_force_norm_n=required_force_norm_n,
            force_residual_norm_n=force_residual_n,
            torque_residual_norm_n_m=torque_residual_n_m,
            equivalent_moment_arm_residual_m=moment_arm_residual_m,
            force_residual_tolerance_n=force_residual_tolerance_n,
            moment_arm_residual_tolerance_m=moment_arm_residual_tolerance_m,
            active_generator_indices=active_indices,
        )

    def residual_score(proof: ForceClosureProof) -> float:
        return max(
            proof.force_residual_norm_n / force_residual_tolerance_n,
            proof.equivalent_moment_arm_residual_m / moment_arm_residual_tolerance_m,
        )

    generators = two_contact_friction_wrench_generators(
        contact_points_m,
        inward_normals,
        static_friction_coefficient,
    )
    coefficients = np.zeros(generators.shape[0], dtype=np.float64)
    achieved = coefficients @ generators
    residual = target - achieved
    best = make_proof(coefficients, achieved, residual, ())
    best_score = residual_score(best)
    if best.feasible:
        return best

    generator_rank = int(np.linalg.matrix_rank(generators, tol=1.0e-12))
    maximum_subset_size = min(6, generator_rank)
    nonnegative_tolerance = 1.0e-12 * max(1.0, float(np.linalg.norm(target)))

    for subset_size in range(1, maximum_subset_size + 1):
        for indices in combinations(range(generators.shape[0]), subset_size):
            subset = generators[np.asarray(indices)].T
            subset_coefficients = np.linalg.lstsq(subset, target, rcond=1.0e-12)[0]
            if np.any(subset_coefficients < -nonnegative_tolerance):
                continue
            subset_coefficients = np.maximum(subset_coefficients, 0.0)
            candidate_coefficients = np.zeros_like(coefficients)
            candidate_coefficients[np.asarray(indices)] = subset_coefficients
            candidate_achieved = candidate_coefficients @ generators
            candidate_residual = target - candidate_achieved
            active = tuple(int(index) for index in np.flatnonzero(candidate_coefficients > nonnegative_tolerance))
            candidate = make_proof(candidate_coefficients, candidate_achieved, candidate_residual, active)
            candidate_score = residual_score(candidate)
            if candidate_score < best_score:
                best = candidate
                best_score = candidate_score
                if best.feasible:
                    return best
    return best


def assess_finite_contact_acceptance(
    proof: ForceClosureProof,
    *,
    force_residual_tolerance_n: float,
    moment_arm_residual_tolerance_m: float,
) -> FiniteContactAcceptance:
    """Apply an explicit soft-contact rule to an inexact point-contact proof.

    This helper does not change ``proof.exact_point_contact_feasible``.  It
    models a separately justified finite contact patch or torsional compliance
    and therefore requires both physical tolerances from the caller.  A
    conservative task should keep the moment-arm allowance at or below
    10 micrometres unless independent contact-patch evidence supports more.
    """

    for name, tolerance in (
        ("force_residual_tolerance_n", force_residual_tolerance_n),
        ("moment_arm_residual_tolerance_m", moment_arm_residual_tolerance_m),
    ):
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    force_within_tolerance = proof.force_residual_norm_n <= force_residual_tolerance_n
    moment_arm_within_tolerance = proof.equivalent_moment_arm_residual_m <= moment_arm_residual_tolerance_m
    return FiniteContactAcceptance(
        accepted=force_within_tolerance and moment_arm_within_tolerance,
        force_within_tolerance=force_within_tolerance,
        moment_arm_within_tolerance=moment_arm_within_tolerance,
        force_residual_tolerance_n=force_residual_tolerance_n,
        moment_arm_residual_tolerance_m=moment_arm_residual_tolerance_m,
    )


def grasp_matrix(contact_points_m: ArrayLike) -> np.ndarray:
    """Return the six-dimensional point-contact grasp matrix for two contacts."""

    points = np.asarray(contact_points_m, dtype=np.float64)
    if points.shape != (2, 3) or not np.isfinite(points).all():
        raise ValueError("contact_points_m must be a finite (2, 3) array")
    blocks = []
    for point in points:
        skew = np.array(
            ((0.0, -point[2], point[1]), (point[2], 0.0, -point[0]), (-point[1], point[0], 0.0)),
            dtype=np.float64,
        )
        blocks.append(np.vstack((np.eye(3), skew)))
    return np.hstack(blocks)


def impedance_gains(
    reflected_inertia_kg_m2: float,
    natural_frequency_rad_s: float,
    damping_ratio: float,
) -> tuple[float, float]:
    """Derive ``Kp`` and ``Kd`` from reflected inertia and pole placement."""

    inputs = (reflected_inertia_kg_m2, natural_frequency_rad_s, damping_ratio)
    if not all(math.isfinite(value) and value > 0.0 for value in inputs):
        raise ValueError("impedance inputs must be finite and positive")
    stiffness = reflected_inertia_kg_m2 * natural_frequency_rad_s**2
    damping = 2.0 * damping_ratio * reflected_inertia_kg_m2 * natural_frequency_rad_s
    return stiffness, damping


def solve_minimum_closing_target(
    normal_load_fn: Callable[[float], float],
    *,
    required_normal_load_n: float,
    lower_closedness: float,
    upper_closedness: float = 1.0,
    tolerance: float = 1.0e-6,
    max_iterations: int = 80,
) -> float:
    """Find the smallest bounded closedness whose measured model meets the load."""

    if not 0.0 <= lower_closedness <= upper_closedness <= 1.0:
        raise ValueError("closedness bounds must be ordered inside [0, 1]")
    if not math.isfinite(required_normal_load_n) or required_normal_load_n <= 0.0:
        raise ValueError("required_normal_load_n must be finite and positive")

    def evaluate(value: float) -> float:
        load = float(normal_load_fn(value))
        if not math.isfinite(load) or load < 0.0:
            raise ValueError("normal_load_fn must return a finite non-negative load")
        return load

    lower_load = evaluate(lower_closedness)
    upper_load = evaluate(upper_closedness)
    if upper_load < lower_load:
        raise ValueError("normal-load model must be non-decreasing")
    if upper_load < required_normal_load_n:
        raise RuntimeError("authored jaw drive cannot meet the required retaining load")
    if lower_load >= required_normal_load_n:
        return lower_closedness
    lower, upper = lower_closedness, upper_closedness
    for _ in range(max_iterations):
        if upper - lower <= tolerance:
            break
        midpoint = 0.5 * (lower + upper)
        if evaluate(midpoint) >= required_normal_load_n:
            upper = midpoint
        else:
            lower = midpoint
    return upper


__all__ = [
    "EXACT_POINT_CONTACT_FORCE_RESIDUAL_TOLERANCE_N",
    "EXACT_POINT_CONTACT_MOMENT_ARM_TOLERANCE_M",
    "FiniteContactAcceptance",
    "FORCE_CLOSURE_CONE_FACETS",
    "ForceClosureProof",
    "RetentionLoad",
    "assess_finite_contact_acceptance",
    "friction_cone_generators",
    "grasp_matrix",
    "impedance_gains",
    "prove_two_contact_force_closure",
    "required_retention_load",
    "solve_minimum_closing_target",
    "two_contact_friction_wrench_generators",
]
