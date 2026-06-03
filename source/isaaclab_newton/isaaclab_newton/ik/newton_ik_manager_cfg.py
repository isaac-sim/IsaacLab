# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils.configclass import configclass


@configclass
class NewtonIKManagerCfg:
    """Configuration for the Newton inverse-kinematics manager."""

    command_type: str = "pose"
    """Action command type for manager-based actions: ``"position"`` or ``"pose"``."""

    use_relative_mode: bool = True
    """Whether manager-based action commands are relative to the current target pose."""

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

    position_weight: float = 1.0
    """Default residual weight for pose position objectives."""

    rotation_weight: float = 1.0
    """Default residual weight for pose rotation objectives."""

    joint_limit_weight: float | None = 0.1
    """Residual weight for the joint-limit objective. Set to ``None`` to disable it."""

    use_persistent_seed: bool = False
    """Whether solved joint coordinates should be reused as the next IK seed."""
