# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Identified motor parameters for the Newton-native BAM servo controller."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

BAM_XL330_M6_PARAMS_FILE: str = str(Path(__file__).parent / "data" / "bam_xl330_m6.json")
"""Path of the BAM parameters vendored with Isaac Lab (Dynamixel XL330, ``m6`` model)."""

# Friction-model flags of the BAM model family, mirroring ``bam/model.py`` (``models``).
# Only the models Isaac Lab supports are listed; see :meth:`BamMotorParams.from_json`.
_BAM_MODEL_FLAGS: dict[str, dict[str, bool]] = {
    "m1": {"stribeck": False, "load_dependent": False, "directional": False, "quadratic": False},
    "m2": {"stribeck": True, "load_dependent": False, "directional": False, "quadratic": False},
    "m5": {"stribeck": True, "load_dependent": True, "directional": True, "quadratic": False},
    "m6": {"stribeck": True, "load_dependent": True, "directional": True, "quadratic": True},
}


@dataclass(frozen=True)
class BamMotorParams:
    """Identified motor, firmware and friction parameters of one BAM actuator model.

    The values are per actuator *type* (not per joint or per environment): they come from
    fitting the BAM model to bench measurements of a specific servo. Isaac Lab ships the
    identified parameters of the Dynamixel XL330 in
    ``isaaclab/actuators/data/bam_xl330_m6.json``; see the ``ATTRIBUTION.md`` next to it.

    The friction-model flags select which terms of the budget are active and follow the
    BAM ``m1``--``m6`` model family (``m6``, the model Isaac Lab uses, enables all of them).

    Attributes:
        kt: Motor torque constant [N.m/A], equivalently the back-EMF constant [V.s/rad].
        R: Motor winding resistance [Ohm].
        armature: Rotor inertia reflected through the gearbox [kg.m^2].
        error_gain: Converts ``kp`` times the position error [rad] into a duty cycle [-].
        max_pwm: Largest duty-cycle magnitude the firmware can command [-].
        max_current: Firmware current limit [A], or None to disable the current limiter.
        kp: Nominal firmware proportional gain [-].
        vin: Nominal supply voltage [V].
        friction_base: Load-independent Coulomb friction [N.m].
        friction_viscous: Viscous friction coefficient [N.m.s/rad].
        friction_stribeck: Extra Coulomb friction at rest, from the Stribeck effect [N.m].
        dtheta_stribeck: Velocity scale over which the Stribeck effect decays [rad/s].
        alpha: Exponent shaping the Stribeck decay [-].
        load_friction_motor: Gearbox friction per unit of motor-side torque [N.m/N.m].
        load_friction_external: Gearbox friction per unit of external torque [N.m/N.m].
        load_friction_motor_stribeck: Stribeck part of ``load_friction_motor`` [N.m/N.m].
        load_friction_external_stribeck: Stribeck part of ``load_friction_external`` [N.m/N.m].
        load_friction_motor_quad: Quadratic back-driving friction coefficient [N.m/(N.m)^2].
        load_friction_external_quad: Quadratic driving friction coefficient [N.m/(N.m)^2].
        stribeck: Whether the Stribeck (near-zero-velocity) friction terms are active.
        load_dependent: Whether the gearbox friction grows with the transmitted torque.
        directional: Whether the load-dependent friction distinguishes the motor side from
            the external side. Requires ``load_dependent``.
        quadratic: Whether the quadratic load-coupling term is active. Requires
            ``directional`` and ``stribeck``.
    """

    kt: float
    R: float
    armature: float
    error_gain: float
    max_pwm: float
    max_current: float | None
    kp: float
    vin: float
    friction_base: float
    friction_viscous: float
    friction_stribeck: float = 0.0
    dtheta_stribeck: float = 1.0
    alpha: float = 1.0
    load_friction_motor: float = 0.0
    load_friction_external: float = 0.0
    load_friction_motor_stribeck: float = 0.0
    load_friction_external_stribeck: float = 0.0
    load_friction_motor_quad: float = 0.0
    load_friction_external_quad: float = 0.0
    stribeck: bool = False
    load_dependent: bool = False
    directional: bool = False
    quadratic: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> BamMotorParams:
        """Load parameters from a BAM parameter file.

        The file layout is the one BAM writes in ``bam/params/<motor>/<model>.json``,
        extended with the firmware constants that BAM keeps in code
        (``error_gain``, ``max_pwm``, ``max_current``, ``kp``, ``vin``). Keys that do not
        name a field of this class (such as ``q_offset`` or ``actuator``) are ignored, so
        an upstream parameter file can be vendored verbatim.

        Args:
            path: Path of the JSON file to read.

        Returns:
            The parsed parameters, with the friction-model flags set from the ``model`` key.

        Raises:
            KeyError: If the file does not declare a ``model``, or declares a BAM friction
                model that this port does not implement (the non-directional load-dependent
                models ``m3`` and ``m4``).
            TypeError: If a parameter required by the declared model is missing.
        """
        content = json.loads(Path(path).read_text())
        model_name = content.get("model")
        if model_name not in _BAM_MODEL_FLAGS:
            raise KeyError(
                f"BAM parameter file '{path}' declares model {model_name!r}, which is not supported."
                f" Supported models: {sorted(_BAM_MODEL_FLAGS)}."
            )
        field_names = {field.name for field in fields(cls)}
        values = {key: value for key, value in content.items() if key in field_names}
        return cls(**values, **_BAM_MODEL_FLAGS[model_name])
