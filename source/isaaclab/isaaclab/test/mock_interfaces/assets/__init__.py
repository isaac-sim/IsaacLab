# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock asset interfaces for testing without Isaac Sim."""

from .mock_articulation import MockArticulation, MockArticulationData
from .mock_rigid_object import MockRigidObject, MockRigidObjectData
from .mock_rigid_object_collection import MockRigidObjectCollection, MockRigidObjectCollectionData
from .factories import (
    create_mock_articulation,
    create_mock_humanoid,
    create_mock_quadruped,
    create_mock_rigid_object,
    create_mock_rigid_object_collection,
)
