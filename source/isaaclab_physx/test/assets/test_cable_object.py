# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from isaaclab_physx.assets import CableObject

from isaaclab.assets import CableObjectCfg


def test_cable_object_reports_unsupported_backend():
    """Test that constructing a cable object reports unsupported PhysX behavior."""
    cfg = CableObjectCfg(prim_path="/World/Cable")

    with pytest.raises(NotImplementedError, match="CableObject is not supported by the PhysX backend"):
        CableObject(cfg)
