# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton's implicit MPM solver."""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Literal

from isaaclab.utils import configclass

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from newton.solvers import SolverImplicitMPM

    from pxr import Usd


@configclass
class MPMSolverCfg(NewtonSolverCfg):
    """Configuration for Newton's implicit Material Point Method (MPM) solver.

    The implicit MPM solver advances particle materials and treats rigid geometry
    as colliders. It is not a rigid-body or articulation dynamics solver.
    """

    class_type: type | str = "{DIR}.mpm_solver:MPMSolver"

    solver_type: str = "implicit_mpm"
    """Solver type. Can be "implicit_mpm"."""

    # numerics
    max_iterations: int = 250
    """Maximum number of iterations for the rheology solver."""

    tolerance: float = 1.0e-4
    """Tolerance for the rheology solver."""

    solver: str | tuple[str, ...] = "auto"
    """Rheology solver, or an ordered warm-start sequence of solvers.

    Values use the canonical tokens defined by ``NewtonMPMSceneAPI``. ``"auto"``
    lets Newton pick the solver from the velocity basis. Pass a tuple such as
    ``("conjugate-residual", "gauss-seidel")`` to warm-start solvers left-to-right.
    """

    warmstart_mode: Literal["none", "auto", "particles", "grid", "smoothed"] = "auto"
    """Warm-start mode for the rheology solver."""

    collider_velocity_mode: Literal["forward", "backward", "instantaneous", "finite_difference"] = "forward"
    """Collider velocity computation mode.

    ``"instantaneous"`` is deprecated in favor of ``"forward"``, and
    ``"finite_difference"`` is deprecated in favor of ``"backward"``.
    """

    # grid
    voxel_size: float = 0.1
    """Size of the MPM grid voxels [m]."""

    grid_type: Literal["sparse", "dense", "fixed"] = "sparse"
    """Type of grid to use."""

    grid_padding: int = 0
    """Number of empty cells to add around particles when allocating the grid."""

    max_active_cell_count: int = -1
    """Maximum active grid-cell count shared by all worlds.

    A positive value reserves persistent capacity for rebuildable sparse grids and bounds active
    subsets of dense and fixed grids. ``-1`` keeps sparse allocation dynamic and uses exact active
    counts for the other grid types.
    """

    max_leaf_node_count: int = -1
    """Maximum sparse-grid leaf-node count shared by all worlds.

    ``-1`` lets Newton derive the capacity from :attr:`max_active_cell_count`.
    """

    max_lower_node_count: int = -1
    """Maximum sparse-grid lower internal-node count shared by all worlds.

    ``-1`` lets Newton derive the capacity from the initial topology.
    """

    max_upper_node_count: int = -1
    """Maximum sparse-grid upper internal-node count shared by all worlds.

    ``-1`` lets Newton derive the capacity from the initial topology.
    """

    separate_worlds: bool = False
    """Whether each Newton world uses an independent local MPM grid environment."""

    transfer_scheme: Literal["apic", "pic"] = "apic"
    """Particle-grid transfer scheme."""

    integration_scheme: Literal["pic", "gimp"] = "pic"
    """Integration scheme controlling shape-function support."""

    # material / background
    critical_fraction: float = 0.0
    """Dimensionless fraction under which the yield surface collapses."""

    air_drag: float = 1.0
    """Numerical drag for background air."""

    # experimental
    collider_normal_from_sdf_gradient: bool = False
    """Whether collider normals are computed from SDF gradients rather than closest points."""

    collider_basis: str = "S2"
    """Collider basis function, such as ``"S2"`` or ``"Q1"``."""

    strain_basis: str = "P0"
    """Strain basis function, such as ``"P0"``, ``"P1d"``, ``"Q1"``, or ``"Q1d"``."""

    velocity_basis: str = "Q1"
    """Velocity basis function, such as ``"Q1"``, ``"B2"``, or ``"B3"``."""

    project_outside_colliders: bool = False
    """Project particles out of collider interiors after each solver substep.

    Calls :meth:`SolverImplicitMPM.project_outside` after the implicit solve. This is an Isaac Lab stepping option,
    separate from ``SolverImplicitMPM.Config``.
    """

    def to_solver_config(self) -> SolverImplicitMPM.Config:
        """Return Newton's configuration without accessing or modifying a USD stage."""
        import warp as wp
        from newton.solvers import SolverImplicitMPM

        from .newton_solver import solver_kwargs

        values = solver_kwargs(SolverImplicitMPM.Config, self, wp.DeterministicMode.NOT_GUARANTEED)
        mode = self.collider_velocity_mode
        replacement = {"instantaneous": "forward", "finite_difference": "backward"}.get(mode)
        if replacement is not None:
            warnings.warn(
                f"collider_velocity_mode={mode!r} is deprecated; use {replacement!r} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            values["collider_velocity_mode"] = replacement
        return SolverImplicitMPM.Config(**values)

    def author_usd(self, scene_prim: Usd.Prim) -> None:
        """Author schema-representable solver settings on an explicit physics scene.

        Args:
            scene_prim: Physics scene prim receiving ``NewtonMPMSceneAPI`` and its settings.
        """
        from pxr import Sdf, Vt  # noqa: PLC0415

        if not scene_prim.AddAppliedSchema("NewtonMPMSceneAPI"):
            raise RuntimeError(f"Failed to apply NewtonMPMSceneAPI to '{scene_prim.GetPath()}'.")

        rheology_solvers = (self.solver,) if isinstance(self.solver, str) else self.solver

        attributes = {
            "newton:maxSolverIterations": self.max_iterations,
            "newton:mpm:tolerance": self.tolerance,
            "newton:mpm:rheologySolvers": Vt.TokenArray(rheology_solvers),
            "newton:mpm:voxelSize": self.voxel_size,
            "newton:mpm:gridType": self.grid_type,
            "newton:mpm:gridPadding": self.grid_padding,
            "newton:mpm:maxActiveCellCount": self.max_active_cell_count,
            "newton:mpm:transferScheme": self.transfer_scheme,
            "newton:mpm:integrationScheme": self.integration_scheme,
            "newton:mpm:criticalFraction": self.critical_fraction,
            "newton:mpm:airDrag": self.air_drag,
        }
        attributes.update(_basis_attributes("collider", self.collider_basis))
        attributes.update(_basis_attributes("strain", self.strain_basis))
        attributes.update(_basis_attributes("velocity", self.velocity_basis))
        for name, value in attributes.items():
            if name == "newton:mpm:rheologySolvers":
                type_name = Sdf.ValueTypeNames.TokenArray
            elif isinstance(value, bool):
                type_name = Sdf.ValueTypeNames.Bool
            elif isinstance(value, int):
                type_name = Sdf.ValueTypeNames.Int
            elif isinstance(value, str):
                type_name = Sdf.ValueTypeNames.Token
            else:
                type_name = Sdf.ValueTypeNames.Float
            scene_prim.CreateAttribute(name, type_name, custom=False, variability=Sdf.VariabilityUniform).Set(value)


def _basis_attributes(prefix: str, basis: str) -> dict[str, object]:
    """Expand a compact Newton basis name into newton-usd-schemas attributes."""
    if basis.startswith("pic"):
        if prefix == "velocity":
            raise ValueError("The MPM velocity basis cannot use a particle basis.")
        basis_type, order, discontinuous = "particle", 0, False
    else:
        matched = re.fullmatch(r"([PQBS])([0-9]+)(d?)", basis)
        if matched is None:
            raise ValueError(f"Unsupported MPM {prefix} basis {basis!r}.")
        basis_prefix = matched.group(1)
        order = int(matched.group(2))
        discontinuous = order == 0 or bool(matched.group(3))
        if basis_prefix == "P" and order > 0 and not discontinuous:
            raise ValueError(f"Unsupported MPM {prefix} basis {basis!r}: positive-order P bases must be discontinuous.")
        if (prefix == "strain" and basis_prefix in "BS") or (
            prefix == "velocity" and (basis_prefix not in "QB" or not 1 <= order <= 3)
        ):
            return {}
        if basis_prefix in "BS" and not 1 <= order <= 3:
            return {}
        basis_type = {
            "P": "linear",
            "Q": "trilinear",
            "B": "bspline",
            "S": "serendipity",
        }[basis_prefix]

    namespace = f"newton:mpm:{prefix}"
    attributes: dict[str, object] = {
        f"{namespace}BasisType": basis_type,
        f"{namespace}BasisOrder": order,
    }
    if prefix != "velocity":
        attributes[f"{namespace}DiscontinuousBasis"] = discontinuous
    return attributes
