# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .differential_ik import DifferentialIKController


@configclass
class DifferentialIKControllerCfg:
    """Configuration for differential inverse kinematics controller."""

    class_type: type[DifferentialIKController] | str = "{DIR}.differential_ik:DifferentialIKController"
    """The associated controller class."""

    command_type: Literal["position", "pose"] = MISSING
    """Type of task-space command to control the articulation's body.

    If "position", then the controller only controls the position of the articulation's body.
    Otherwise, the controller controls the pose of the articulation's body.
    """

    use_relative_mode: bool = False
    """Whether to use relative mode for the controller. Defaults to False.

    If True, then the controller treats the input command as a delta change in the position/pose.
    Otherwise, the controller treats the input command as the absolute position/pose.
    """

    ik_method: Literal["pinv", "svd", "trans", "dls", "adaptive_dls"] = MISSING
    """Method for computing inverse of Jacobian."""

    ik_params: dict[str, float] | None = None
    """Parameters for the inverse-kinematics method. Defaults to None, in which case the default
    parameters for the method are used.

    - Moore-Penrose pseudo-inverse ("pinv"):
        - "k_val": Scaling of computed delta-joint positions (default: 1.0).
    - Adaptive Singular Value Decomposition ("svd"):
        - "k_val": Scaling of computed delta-joint positions (default: 1.0).
        - "min_singular_value": Single values less than this are suppressed to zero (default: 1e-5).
    - Jacobian transpose ("trans"):
        - "k_val": Scaling of computed delta-joint positions (default: 1.0).
    - Damped Moore-Penrose pseudo-inverse ("dls"):
        - "lambda_val": Damping coefficient (default: 0.01).
    - Manipulability-aware damped least squares ("adaptive_dls"):
        - "lambda_min": Baseline damping coefficient used away from singularities (default: 0.05).
        - "lambda_max": Maximum damping coefficient, reached as the smallest task-Jacobian
          singular value approaches zero (default: 0.2).
        - "sigma_thresh": Smallest-singular-value threshold below which the damping ramps
          quadratically from ``lambda_min`` toward ``lambda_max`` (Maciejewski-Klein style)
          (default: 0.02).
    """

    orientation_weight: float | tuple[float, float, float] | None = None
    """Soft weight on the orientation task rows for ``"pose"`` command types. Defaults to ``None``
    (the orientation rows keep weight 1, i.e. unchanged behavior).

    A scalar weights all three orientation rows equally; a per-axis ``(wx, wy, wz)`` weights the
    base-frame orientation axes independently. Scaling an orientation row (and its error) by a
    weight de-emphasises -- or, at weight 0, drops -- that rotation DOF in the solve without
    changing the task dimensionality. This is useful for arms that cannot serve a full 6-DOF pose
    (e.g. a 5-DOF arm) so the unreachable orientation DOF degrades gracefully instead of leaking
    error into the position rows. Ignored for ``"position"`` command types.
    """

    joint_limit_avoidance_gain: float = 0.0
    """Gain for the null-space joint-limit-avoidance bias. ``0`` disables it (default).

    When positive, a center-seeking joint velocity (active only within
    :attr:`joint_limit_avoidance_margin` of a limit) is projected into the null space of the
    position task rows, so it keeps joints off their limits without perturbing the commanded
    end-effector position. Active only once joint limits are provided via
    :meth:`~isaaclab.controllers.differential_ik.DifferentialIKController.set_joint_pos_limits`
    (the IK action term injects them automatically when ``joint_limit_avoidance_gain > 0``).
    """

    joint_limit_avoidance_margin: float = 0.3
    """Joint-range margin within which the joint-limit-avoidance bias activates (1 at the limit,
    ramping to 0 at ``joint_limit_avoidance_margin`` away from it). Units match the joints
    (e.g. [rad] for revolute joints)."""

    def __post_init__(self):
        # check valid input
        if self.command_type not in ["position", "pose"]:
            raise ValueError(f"Unsupported inverse-kinematics command: {self.command_type}.")
        if self.ik_method not in ["pinv", "svd", "trans", "dls", "adaptive_dls"]:
            raise ValueError(f"Unsupported inverse-kinematics method: {self.ik_method}.")
        # default parameters for different inverse kinematics approaches.
        default_ik_params = {
            "pinv": {"k_val": 1.0},
            "svd": {"k_val": 1.0, "min_singular_value": 1e-5},
            "trans": {"k_val": 1.0},
            "dls": {"lambda_val": 0.01},
            "adaptive_dls": {"lambda_min": 0.05, "lambda_max": 0.2, "sigma_thresh": 0.02},
        }
        # update parameters for IK-method if not provided
        ik_params = default_ik_params[self.ik_method].copy()
        if self.ik_params is not None:
            ik_params.update(self.ik_params)
        self.ik_params = ik_params
        # validate adaptive_dls parameters
        if self.ik_method == "adaptive_dls":
            if self.ik_params["sigma_thresh"] <= 0.0:
                raise ValueError(f"adaptive_dls sigma_thresh must be > 0, got {self.ik_params['sigma_thresh']}.")
            if self.ik_params["lambda_min"] > self.ik_params["lambda_max"]:
                raise ValueError(
                    f"adaptive_dls lambda_min ({self.ik_params['lambda_min']}) must be <= "
                    f"lambda_max ({self.ik_params['lambda_max']})."
                )
        # validate optional orientation weighting / joint-limit-avoidance settings
        if self.orientation_weight is not None and not isinstance(self.orientation_weight, (int, float)):
            if len(self.orientation_weight) != 3:
                raise ValueError(
                    "orientation_weight must be a scalar or a length-3 (wx, wy, wz) tuple, got "
                    f"{self.orientation_weight}."
                )
        if self.joint_limit_avoidance_gain < 0.0:
            raise ValueError(f"joint_limit_avoidance_gain must be >= 0, got {self.joint_limit_avoidance_gain}.")
        if self.joint_limit_avoidance_margin <= 0.0:
            raise ValueError(f"joint_limit_avoidance_margin must be > 0, got {self.joint_limit_avoidance_margin}.")
