# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physically grasp the Utah teapot and pour MPM water during a 70 cm rise.

Run with ``uv run isaaclab example rizon-sharpa-teapot``. This experimental
example uses a fixed index grasp and motion.

The proprietary robot bundle is not distributed with Isaac Lab. Obtain the
generated USD and its ``textures/`` directory from the Fabrics-Sim asset
maintainer under your applicable access and license terms:
https://gitlab-master.nvidia.com/dex/fabrics-sim/-/tree/d0dbd1ddaefc4996db546949a7dfb37e39afcbeb/src/fabrics_sim/models/robots/urdf/rizon4s_sharpa/rizon4s_sharpa_no_spheres

Keep ``rizon4s_sharpa_no_spheres_generated.usd`` beside ``textures/`` and set
``ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT`` to that directory. The example checks
the USD and its textures against the pinned bundle digest before launching.
"""

from __future__ import annotations

import math
import runpy
import sys
from pathlib import Path

TABLETOP_CONTAINER_POS = (-0.355, 0.0, 0.90041)
CONTAINER_BASE_POS = (-0.340, 0.0, 1.020)
BOWL_SCALE = 1.5
BOWL_BASE_POS = (-0.144, 0.0, 0.9004073255)
ROBOT_BASE_POS = (-0.930, 0.0, 1.15041)
PICKUP_TIME = 5.6
INITIAL_LIFT_TIME = 5.0
POUR_PREFIX_TIME = PICKUP_TIME + INITIAL_LIFT_TIME
HOLD_TIME = 0.55
TILT_TIME = 4.0
LIFT_TIME = 24.0
LIFT_HEIGHT = 0.70
RECOVERY_TIME = 5.0
SEQUENCE_DURATION = POUR_PREFIX_TIME + HOLD_TIME + TILT_TIME + LIFT_TIME + RECOVERY_TIME + 0.45
_POUR_ANGLE = math.radians(20.0)
_UPPER_ANGLE = math.radians(49.0)
_SPOUT_LOWERING_ANGLE = math.radians(35.0)
_SPOUT_LOCAL_POS = (0.111, 0.0, 0.0805)
_SPOUT_POUR_POS = (-0.169, 0.0, 0.9937642035)


def _blend(raw: float) -> tuple[float, float]:
    """Return quintic progress and its derivative with respect to raw progress."""
    progress = max(0.0, min(1.0, raw))
    return (
        progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress)),
        30.0 * progress**2 * (1.0 - progress) ** 2 if 0.0 < raw < 1.0 else 0.0,
    )


def pose_at_time(sim_time: float) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Return the teapot root position [m], XYZW quaternion and linear/angular twist [m/s, rad/s]."""
    if sim_time < POUR_PREFIX_TIME:
        progress, derivative = _blend((sim_time - PICKUP_TIME) / INITIAL_LIFT_TIME)
        displacement = tuple(b - a for a, b in zip(TABLETOP_CONTAINER_POS, CONTAINER_BASE_POS, strict=True))
        return (
            tuple(a + progress * delta for a, delta in zip(TABLETOP_CONTAINER_POS, displacement, strict=True)),
            (0.0, 0.0, 0.0, 1.0),
            (*tuple(derivative * delta / INITIAL_LIFT_TIME for delta in displacement), 0.0, 0.0, 0.0),
        )

    time = sim_time - POUR_PREFIX_TIME - HOLD_TIME
    tilt, tilt_derivative = _blend(time / TILT_TIME)
    lift, lift_derivative = _blend((time - TILT_TIME) / LIFT_TIME)
    recovery_raw = (time - TILT_TIME - LIFT_TIME) / RECOVERY_TIME
    recovery, recovery_derivative = _blend(recovery_raw)
    # Increase tilt gradually so the fuller open pot drains through the spout before its rim lowers.
    angle = _POUR_ANGLE * tilt + (_UPPER_ANGLE - _POUR_ANGLE) * lift - _UPPER_ANGLE * recovery
    angular_speed = (
        _POUR_ANGLE * tilt_derivative / TILT_TIME
        + (_UPPER_ANGLE - _POUR_ANGLE) * lift_derivative / LIFT_TIME
        - _UPPER_ANGLE * recovery_derivative / RECOVERY_TIME
    )
    lift_speed = LIFT_HEIGHT * lift_derivative / LIFT_TIME

    # Lower the outlet over the bowl before tilting, preserving the calibrated path.
    alignment, alignment_derivative = _blend(angle / _SPOUT_LOWERING_ANGLE)
    alignment_speed = alignment_derivative * angular_speed / _SPOUT_LOWERING_ANGLE
    x, y, z = _SPOUT_LOCAL_POS
    sine, cosine = math.sin(angle), math.cos(angle)
    rotated_spout = (x * cosine + z * sine, y, -x * sine + z * cosine)
    displacement = tuple(
        target - outlet - origin
        for target, outlet, origin in zip(_SPOUT_POUR_POS, rotated_spout, CONTAINER_BASE_POS, strict=True)
    )
    position = tuple(a + alignment * delta for a, delta in zip(CONTAINER_BASE_POS, displacement, strict=True))
    target_velocity = (angular_speed * (x * sine - z * cosine), 0.0, angular_speed * (x * cosine + z * sine))
    velocity = tuple(
        alignment_speed * delta + alignment * speed for delta, speed in zip(displacement, target_velocity, strict=True)
    )

    # Compensate for the longer falling stream, then ease forward for the final drips.
    gap = 0.028
    fall_speed = math.sqrt(0.45**2 + 2.0 * 9.81 * gap)
    raised_fall_speed = math.sqrt(0.45**2 + 2.0 * 9.81 * (gap + LIFT_HEIGHT * lift))
    forward_speed = 0.30 - 0.04 * lift
    extra_fall_time = (raised_fall_speed - fall_speed) / 9.81
    retreat_speed = -0.04 * lift_speed / LIFT_HEIGHT * extra_fall_time + forward_speed * lift_speed / raised_fall_speed
    drip, drip_derivative = _blend(recovery_raw * RECOVERY_TIME / 0.8)
    position = (
        position[0] - forward_speed * extra_fall_time + 0.01 * drip,
        position[1],
        position[2] + LIFT_HEIGHT * lift,
    )
    velocity = (velocity[0] - retreat_speed + 0.01 * drip_derivative / 0.8, velocity[1], velocity[2] + lift_speed)
    return position, (0.0, math.sin(0.5 * angle), 0.0, math.cos(0.5 * angle)), (*velocity, 0.0, angular_speed, 0.0)


def approach_offset_at_time(sim_time: float) -> tuple[float, ...]:
    """Descend beside the pot, align with the handle and insert the index finger [m]."""
    if sim_time < 1.8:
        first, second = (-0.09, -0.16, 0.20), (0.0, -0.10, 0.065)
        raw = (sim_time - 0.3) / 1.5
    elif sim_time < 2.6:
        first, second = (0.0, -0.10, 0.065), (0.0, -0.08, 0.0)
        raw = (sim_time - 1.8) / 0.8
    else:
        first, second = (0.0, -0.08, 0.0), (0.0, 0.0, 0.0)
        raw = (sim_time - 2.6) / 0.8
    progress, _ = _blend(raw)
    return tuple(a + progress * (b - a) for a, b in zip(first, second, strict=True))


def main() -> None:
    """Run the shared teapot scene with the fixed physical index grasp."""
    original_args = sys.argv[:]
    sys.argv[1:1] = [
        "--robot",
        "rizon_sharpa",
        "--fill_level",
        "0.70",
        "--fluid_render_mode",
        "particles",
        "--max_steps",
        str(round(800 * SEQUENCE_DURATION)),
    ]
    try:
        runpy.run_path(str(Path(__file__).parent / "demos" / "teapot_fill.py"), run_name="__main__")
    finally:
        sys.argv[:] = original_args


if __name__ == "__main__":
    main()
