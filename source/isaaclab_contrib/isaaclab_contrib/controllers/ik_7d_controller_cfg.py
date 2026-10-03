# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ik_7d redundant-arm inverse-kinematics controller."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils.configclass import configclass


@configclass
class Ik7dControllerCfg:
    """Configuration for :class:`~isaaclab_contrib.controllers.Ik7dController`."""

    robot_name: str = MISSING
    """ik_7d model name, e.g. ``"G2_t2_crs"``. Must match the URDF;
    ``arm_plane_angle`` is code-generated per variant."""

    urdf_path: str = MISSING
    """Path to the URDF the ik_7d model is built from.

    This should be the same URDF the articulation's USD was converted from.
    ik_7d builds its own kinematic model from it, independently of Isaac Sim.
    """

    arms: list[str] = ["left", "right"]
    """Which arms this controller drives, in action-vector order.

    Each entry must be a key of :attr:`Ik7dController.ARM_GROUPS`. Listing one
    arm halves the action dimension and leaves the other arm holding position.
    """

    ik_mode: int = 1
    """``ik_7d.IKMode``: 0 = IK6D (pose only), 1 = IK7D (+ swivel), 2 = IK8D."""

    apa_mode: Literal["free_delta", "absolute", "hold"] = "free_delta"
    """How the last element of each arm's action is read as a swivel command.

    * ``"free_delta"`` -- a signed offset from home, routed through the arm's
      probed free direction. A positive action swivels outward on either arm.
    * ``"absolute"`` -- the action is the swivel angle in radians, unmodified.
    * ``"hold"`` -- the action is ignored; the measured swivel angle is
      re-commanded, pinning the elbow where it is.
    """

    articulation_name: str = "robot"
    """Scene entity name of the articulation, used to look up the base link."""

    base_link_name: str = "base_link"
    """Body whose frame ik_7d's poses are expressed in. Must agree with
    ``fk7d(...).base_frame`` or every commanded pose is offset."""

    seed_from_measured: bool = True
    """Whether to warm-start each solve from the measured joint state.

    ik_7d is a single-step warm-started SQP, so the seed is the previous point on
    the trajectory rather than an initial guess. ``True`` closes the loop through
    the robot, correcting disturbances on the next tick; ``False`` feeds the
    previous solution back, which is smoother but lets command and measurement
    drift apart silently.
    """

    hold_on_failure: bool = True
    """Whether to re-issue the last converged solution when the QP fails.

    ik_7d reports failure out of band (``getDebugMsg().sqp_iterations < 0``) and
    hands back the seed unchanged, so a failure is indistinguishable from a
    perfectly-tracked stationary target unless checked explicitly.
    """

    show_ik_warnings: bool = True
    """Whether to print a rate-limited warning when the QP fails."""

    warning_period_s: float = 1.0
    """Minimum seconds between QP-failure warnings. ik_7d's own diagnostics are
    per-solve, which at 200 Hz across two arms is 400 lines a second."""
