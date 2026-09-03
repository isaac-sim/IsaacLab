# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest


def test_base_setup_deformable_body_raises_not_implemented():
    from isaaclab.physics import PhysicsManager

    with pytest.raises(NotImplementedError, match="does not support deformable bodies"):
        PhysicsManager.setup_deformable_body(None, "volume", None, None, None)
