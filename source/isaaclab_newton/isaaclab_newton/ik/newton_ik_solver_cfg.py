# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils.configclass import configclass


@configclass
class NewtonIKSolverCfg:
    """Configuration for the Newton inverse-kinematics solver.

    Holds solver hyperparameters only. Objectives (and their residual weights)
    are configured separately as a list of
    :class:`~isaaclab_newton.ik.newton_ik_objectives_cfg.NewtonIKObjectiveCfg`
    passed to the solver. Command semantics for manager-based actions
    (``command_type``, ``use_relative_mode``) live on the action cfg.

    :attr:`class_type` selects the solver implementation, so an alternative
    solver can be dropped in via config without changing callers.
    """

    class_type: type | str = "isaaclab_newton.ik.newton_ik_solver:NewtonIKSolver"
    """Solver implementation, as a type or a ``"module:Class"`` string.

    Instantiated as ``class_type(cfg, model=..., num_envs=..., device=...,
    objectives=..., link_resolver=...)``.
    """

    optimizer: str = "lm"
    """Newton IK optimizer backend. Supported values are ``"lm"`` and ``"lbfgs"``."""

    jacobian_mode: str = "analytic"
    """Newton IK Jacobian backend. Supported values are ``"analytic"``, ``"autodiff"``, and ``"mixed"``."""

    sampler: str = "none"
    """Initial seed sampler. Supported values are ``"none"``, ``"gauss"``, ``"roberts"``, and ``"uniform"``."""

    n_seeds: int = 1
    """Number of candidate seeds per IK problem. Must be ``1`` when ``sampler="none"``."""

    noise_std: float = 0.1
    """Gaussian sampling standard deviation used when ``sampler="gauss"``."""

    rng_seed: int = 12345
    """Random seed used by stochastic samplers."""

    iterations: int = 24
    """Number of Newton IK solver iterations per action application.

    The default keeps manager-based action applications affordable while still
    giving the Newton solver several refinement steps per control update.
    Increase this for harder targets or tighter residual requirements.
    """

    step_size: float = 1.0
    """LM step scale passed to Newton ``IKSolver.step``. Ignored by L-BFGS."""

    lambda_initial: float = 0.1
    """Initial damping value for the Newton Levenberg-Marquardt optimizer.

    This moderate default favors stable updates near singular configurations
    over aggressive first-step motion.
    """
