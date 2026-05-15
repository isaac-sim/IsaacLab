# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bridge from Isaac Lab rod configuration to Newton ``SolverXPBDRod`` (GPU XPBD Cosserat rods).

This integrates the Newton physics library's direct block-tridiagonal XPBD rod solver
(see `newton.solvers.SolverXPBDRod`, `newton.solvers.xpbd_rod.add_elastic_rod`) with the
same :class:`RodConfig` / :class:`RodGeometryConfig` / :class:`RodMaterialConfig` /
:class:`RodSolverConfig` dataclasses used by :class:`RodSolver`.

**Dependency:** Your installed ``newton`` package must include ``SolverXPBDRod`` (not yet in
all PyPI releases). Until merged to ``newton-physics/newton`` main, install from the PR
branch, for example::

    pip uninstall -y newton
    pip install "newton @ git+https://github.com/newton-physics/newton.git@refs/pull/1981/head"

Or clone the fork that contains ``xpbd_rod`` and ``pip install -e .`` from that tree.

**Multi-env support:** When ``num_envs > 1``, one rod per environment is added to the same
Newton :class:`ModelBuilder` in a single ``SolverXPBDRod`` instance.  Newton automatically
activates ``_BatchedRodWorkspace`` (batched GPU kernels) when more than one rod is registered,
yielding near-linear scaling with environment count.

**Exposed API (beyond ``step`` / ``positions`` / ``time``):**

- :meth:`set_root_orientation` – set a rod's root frame orientation directly on GPU.
- :attr:`max_delta_lambda` – peak Lagrange-multiplier update from the last solver step
  (useful for convergence diagnostics).
- :attr:`max_correction` – peak positional correction from the last solver step.
- :meth:`apply_proximal_control` – translate and rotate a rod's root particle in-place
  via zero-copy ``wp.to_torch`` views (no CPU sync).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import warp as wp

from .rod_data import RodConfig


def orientations_xyzw_along_polyline(positions: torch.Tensor) -> torch.Tensor:
    """Build per-point quaternions (x, y, z, w) so local +X aligns with the polyline tangent.

    Used for capsule visualization when only particle positions are available (e.g. Newton
    ``SolverXPBDRod``). Tangents use central differences mid-chain and one-sided at ends.

    Args:
        positions: ``(N, 3)`` world positions.

    Returns:
        ``(N, 4)`` quaternions in **x, y, z, w** order (matches :class:`RodSolver` convention).
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be (N, 3)")
    n = positions.shape[0]
    device, dtype = positions.device, positions.dtype
    tangents = torch.zeros_like(positions)
    if n == 1:
        tangents[0] = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    else:
        tangents[0] = positions[1] - positions[0]
        tangents[-1] = positions[-1] - positions[-2]
        for i in range(1, n - 1):
            tangents[i] = positions[i + 1] - positions[i - 1]
        tangents = tangents / (tangents.norm(dim=-1, keepdim=True) + 1.0e-8)

    ex = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    quats = torch.zeros(n, 4, device=device, dtype=dtype)
    for i in range(n):
        t = tangents[i]
        d = torch.dot(ex, t).clamp(-1.0, 1.0)
        c = torch.cross(ex, t)
        if d < -0.999999:
            quats[i] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device, dtype=dtype)
            continue
        w = 1.0 + d
        q = torch.stack([c[0], c[1], c[2], w])
        quats[i] = q / (q.norm() + 1.0e-8)
    return quats


def _load_newton_xpbd():
    try:
        import newton
        from newton.solvers import SolverXPBDRod
        from newton.solvers import xpbd_rod as newton_xpbd_rod
    except ImportError as e:
        raise ImportError(
            "Could not import Newton XPBD rod solver. Install a Newton build that exports "
            "`newton.solvers.SolverXPBDRod` and `newton.solvers.xpbd_rod` (see PR #1981 on "
            "newton-physics/newton), e.g.:\n"
            '  pip install "newton @ git+https://github.com/newton-physics/newton.git@refs/pull/1981/head"\n'
            f"Original error: {e}"
        ) from e
    if not hasattr(newton.solvers, "SolverXPBDRod"):
        raise ImportError(
            "Installed `newton` does not expose SolverXPBDRod. Upgrade to a build that "
            "includes `newton/_src/solvers/xpbd_rod` (Newton PR #1981 or later)."
        )
    return newton, SolverXPBDRod, newton_xpbd_rod


def _particle_mass_from_rod_config(config: RodConfig) -> float:
    """Uniform per-particle mass [kg] from cylinder segment density."""
    r = float(config.geometry.radius) if not isinstance(config.geometry.radius, list) else float(
        config.geometry.radius[0]
    )
    L = float(config.geometry.segment_length)
    rho = float(config.material.density)
    vol = math.pi * r * r * L
    return max(vol * rho, 1.0e-9)


def _initial_positions_along_x(config: RodConfig, z_height: float = 1.0) -> np.ndarray:
    """Match common Isaac Lab rod demos: segments along +X, fixed end at origin."""
    n = config.geometry.num_segments
    L = float(config.geometry.segment_length)
    pos = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        pos[i, 0] = (i + 0.5) * L
        pos[i, 2] = z_height
    return pos


class NewtonXPBDRodSolver:
    """XPBD Cosserat rod simulation via Newton ``SolverXPBDRod``.

    Provides a small surface similar to :class:`RodSolver`: ``step()``, ``positions`` as a
    Torch tensor ``(num_envs, N, 3)``, and ``time``. Does not replicate Warp/GXF collision or
    multi-stream ports; use :class:`RodSolver` for those features.

    When ``num_envs > 1``, all rods share a single Newton ``SolverXPBDRod`` instance which
    automatically activates its ``_BatchedRodWorkspace`` path for efficient parallel GPU
    execution.

    Args:
        config: Full :class:`RodConfig` (geometry, material, solver sections).
        num_envs: Number of parallel rod environments (≥ 1).  When > 1, Newton batched
            kernels are used automatically.
        solver_backend: Newton backend string, e.g. ``\"block_thomas\"``, ``\"split_thomas\"``,
            ``\"block_jacobi\"``, ``\"banded_cholesky\"``.
        floor_z: Floor plane Z for Newton kernel, or ``None`` to disable floor collision.
        add_ground_plane: If ``True``, call ``builder.add_ground_plane()`` before finalize.
        initial_z: Initial height of the rod centerline (world Z).
    """

    def __init__(
        self,
        config: RodConfig | None = None,
        num_envs: int = 1,
        *,
        solver_backend: str = "block_thomas",
        floor_z: float | None = None,
        add_ground_plane: bool = False,
        initial_z: float = 1.0,
    ):
        self.config = config or RodConfig()
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.num_envs = num_envs
        self.time = 0.0

        newton, SolverXPBDRod, xpbd_rod = _load_newton_xpbd()

        dev = self.config.device
        if dev in ("cpu", "CPU"):
            finalize_device = "cpu"
        else:
            finalize_device = dev if ":" in dev else "cuda:0"

        builder = newton.ModelBuilder()
        SolverXPBDRod.register_custom_attributes(builder)

        positions = _initial_positions_along_x(self.config, z_height=initial_z)
        r = self.config.geometry.radius
        radius = float(r[0]) if isinstance(r, list) else float(r)
        pm = _particle_mass_from_rod_config(self.config)
        torsion_mod = float(
            self.config.material.shear_modulus
            if self.config.material.shear_modulus is not None
            else self.config.material.young_modulus / (2.0 * (1.0 + self.config.material.poisson_ratio))
        )

        # Add one rod per environment into the same ModelBuilder.
        # Newton's SolverXPBDRod activates _BatchedRodWorkspace automatically when N > 1.
        for _env_i in range(num_envs):
            xpbd_rod.add_elastic_rod(
                builder,
                positions=positions,
                radius=radius,
                particle_mass=pm,
                bend_stiffness=float(max(self.config.material.bend_stiffness, 1.0e-4)),
                twist_stiffness=float(max(self.config.material.twist_stiffness, 1.0e-4)),
                young_modulus=float(self.config.material.young_modulus),
                torsion_modulus=torsion_mod,
                lock_root=True,
                lock_root_rotation=True,
            )
        if add_ground_plane:
            builder.add_ground_plane()

        self._newton = newton
        self._SolverXPBDRod = SolverXPBDRod
        self.model = builder.finalize(device=finalize_device)
        g = self.config.solver.gravity
        self.model.set_gravity(g)

        damp = float(max(self.config.material.damping, 0.0))
        self.solver = SolverXPBDRod(
            model=self.model,
            linear_damping=damp,
            angular_damping=damp,
            solver_backend=solver_backend,
            floor_z=floor_z,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        n = self.config.geometry.num_segments
        self._positions_torch = torch.zeros((num_envs, n, 3), device=self.config.device, dtype=torch.float32)
        self._sync_positions_from_state(self.state_0)

    def _sync_positions_from_state(self, state) -> None:
        """Copy particle positions from ``state.particle_q`` into ``_positions_torch``.

        Uses ``wp.to_torch`` for a zero-copy GPU view — no CPU round-trip.  Each rod's
        slice is identified via ``solver._rod_particle_starts``.
        """
        q_torch = wp.to_torch(state.particle_q)  # (total_particles, 3), GPU, zero-copy
        for env_i in range(self.num_envs):
            start = self.solver._rod_particle_starts[env_i]
            n_pts = self.solver._rods[env_i].num_points
            self._positions_torch[env_i].copy_(q_torch[start : start + n_pts])

    @property
    def positions(self) -> torch.Tensor:
        """Particle positions, shape ``(num_envs, num_segments, 3)`` (world, metres).

        Returns ``(1, N, 3)`` for single-env (backward-compatible) and
        ``(num_envs, N, 3)`` for multi-env.
        """
        return self._positions_torch

    def step(self, dt: float | None = None) -> None:
        """Advance by ``config.solver.dt`` with ``config.solver.num_substeps`` substeps."""
        dt_total = float(dt if dt is not None else self.config.solver.dt)
        n_sub = max(1, int(self.config.solver.num_substeps))
        sub_dt = dt_total / n_sub

        s0, s1 = self.state_0, self.state_1
        for _ in range(n_sub):
            self.solver.step(s0, s1, self.control, self.contacts, sub_dt)
            s0, s1 = s1, s0

        self.state_0, self.state_1 = s0, s1
        self._sync_positions_from_state(s0)
        self.time += dt_total

    # ------------------------------------------------------------------
    # Root control
    # ------------------------------------------------------------------

    def set_root_orientation(self, env_idx: int, q: "torch.Tensor | tuple") -> None:
        """Set the root particle orientation for environment ``env_idx`` directly on GPU.

        Args:
            env_idx: Rod / environment index (0-based).
            q: Quaternion in **(x, y, z, w)** order, as a length-4 ``torch.Tensor``,
               a ``(4,)`` numpy array, or a ``(x, y, z, w)`` tuple/list.
        """
        if isinstance(q, torch.Tensor):
            q = q.float().cpu().tolist()
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        self.solver.set_root_orientation(rod_idx=env_idx, q=wp.quat(x, y, z, w))

    def apply_proximal_control(
        self,
        push_velocity: "torch.Tensor",
        rotate_velocity: "torch.Tensor",
        dt: float,
        env_idx: int = 0,
    ) -> None:
        """Apply proximal (root-end) velocity control to a rod via zero-copy GPU views.

        Translates the root particle by ``push_velocity * dt`` and rotates the root
        frame by integrating ``rotate_velocity * dt`` as an angle-axis step.  All
        modifications are applied directly to the Newton workspace arrays via
        ``wp.to_torch`` (no CPU sync).

        For multi-env calls with scalar-like velocities, invoke once per env using the
        ``env_idx`` argument, or broadcast by looping over all environments.

        Args:
            push_velocity: ``(3,)`` linear velocity [m/s] for the root particle.
            rotate_velocity: ``(3,)`` angular velocity [rad/s] for the root frame.
            dt: Time-step [s].
            env_idx: Environment / rod index (0-based).
        """
        push = push_velocity.float()
        rot = rotate_velocity.float()

        bws = self.solver._batched_ws
        if bws is None:
            ws = self.solver._rods[env_idx]
            pos_t = wp.to_torch(ws.positions_wp)     # (N, 3), GPU zero-copy
            ori_t = wp.to_torch(ws.orientations_wp)  # (N, 4), GPU zero-copy, xyzw
        else:
            pos_all = wp.to_torch(bws.positions)         # (total, 3)
            ori_all = wp.to_torch(bws.orientations)      # (total, 4)
            offset = bws.rod_offsets_cpu[env_idx]
            n_pts = self.solver._rods[env_idx].num_points
            pos_t = pos_all[offset : offset + n_pts]
            ori_t = ori_all[offset : offset + n_pts]

        dev = pos_t.device
        push = push.to(dev)
        rot = rot.to(dev)

        # Translate root particle
        pos_t[0] = pos_t[0] + push * dt

        # Rotate root frame: integrate angular velocity via angle-axis quaternion step
        angle = rot.norm() * dt
        if float(angle) > 1.0e-8:
            axis = rot / rot.norm()
            s = torch.sin(angle * 0.5)
            dq_xyz = axis * s
            dq_w = torch.cos(angle * 0.5)
            dq = torch.cat([dq_xyz, dq_w.unsqueeze(0)])  # (4,) xyzw
            q0 = ori_t[0]  # xyzw
            x1, y1, z1, w1 = dq[0], dq[1], dq[2], dq[3]
            x2, y2, z2, w2 = q0[0], q0[1], q0[2], q0[3]
            new_q = torch.stack([
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ])
            ori_t[0] = new_q / (new_q.norm() + 1.0e-9)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def max_delta_lambda(self) -> float:
        """Peak Lagrange-multiplier update magnitude from the last solver step.

        Useful for convergence diagnostics.  Returns the batched value when
        ``num_envs > 1`` (i.e. the maximum across all rods).
        """
        bws = self.solver._batched_ws
        if bws is None:
            ws = self.solver._rods[0]
            return float(ws._delta_lambda_max_wp.numpy()[0])
        return float(bws._delta_lambda_max.numpy()[0])

    @property
    def max_correction(self) -> float:
        """Peak positional correction magnitude from the last solver step.

        Returns the batched value when ``num_envs > 1`` (maximum across all rods).
        """
        bws = self.solver._batched_ws
        if bws is None:
            ws = self.solver._rods[0]
            return float(ws._correction_max_wp.numpy()[0])
        return float(bws._correction_max.numpy()[0])


__all__ = ["NewtonXPBDRodSolver", "orientations_xyzw_along_polyline"]
