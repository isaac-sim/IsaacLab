# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BAM motor configuration validation."""

import pytest

from isaaclab.actuators import BamMotorCfg
from isaaclab.utils import validate

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("model", ["m3", "m4", "invalid"])
def test_unsupported_friction_model_is_rejected(model):
    """Unsupported variants cannot silently select a different friction law."""
    motor = BamMotorCfg(
        model=model, kt=0.36, resistance=2.8, error_gain=0.003, friction_base=0.005, friction_viscous=0.006
    )
    with pytest.raises(ValueError, match="model"):
        validate(motor)
