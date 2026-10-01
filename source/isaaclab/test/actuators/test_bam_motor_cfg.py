# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BAM fit import and configuration validation."""

import json

import pytest

from isaaclab.actuators import BamMotorCfg
from isaaclab.utils import to_dict, validate

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("firmware_constants", [False, True])
def test_load_rhoban_fit(tmp_path, firmware_constants):
    """Import fitted values without adopting deployment settings or guessing firmware constants."""
    values = dict(
        actuator="xl330",
        model="m6",
        kt=0.36,
        R=2.8,
        friction_base=0.005,
        friction_viscous=0.006,
        load_friction_motor_quad=0.01,
        armature=0.0018,
        q_offset=0.02,
        kp=400.0,
        vin=7.5,
    )
    if firmware_constants:
        values.update(error_gain=0.003, max_pwm=0.9, max_current=1.75)
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(values))
    motor = BamMotorCfg.from_json(path)
    assert motor.model == "m6"
    assert motor.resistance == 2.8
    assert motor.kt == 0.36
    assert motor.friction_base == 0.005
    assert motor.friction_viscous == 0.006
    assert motor.load_friction_motor_quad == 0.01
    if firmware_constants:
        assert motor.error_gain == 0.003
        assert motor.max_pwm == 0.9
        assert motor.max_current == 1.75
    else:
        assert motor.max_pwm == 1.0
        assert motor.max_current == 0.0
        with pytest.raises(TypeError, match="error_gain"):
            validate(motor)
        motor.error_gain = 0.003
    validate(motor)
    assert not {"actuator", "kp", "vin", "armature", "q_offset", "R"} & to_dict(motor).keys()


@pytest.mark.parametrize("model", ["m3", "m4", "invalid"])
def test_unsupported_friction_model_is_rejected(model):
    """Unsupported variants cannot silently select a different friction law."""
    motor = BamMotorCfg(
        model=model, kt=0.36, resistance=2.8, error_gain=0.003, friction_base=0.005, friction_viscous=0.006
    )
    with pytest.raises(ValueError, match="model"):
        validate(motor)
