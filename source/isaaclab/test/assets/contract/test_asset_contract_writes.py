# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared asset write contract: writers, read-backs, partial selections, cache invalidation, and ordered writes."""

import pytest

from ._articulation_contract_cases import (  # noqa: F401
    TestArticulationPartialWriteRows,
    TestArticulationWritersBody,
    TestArticulationWritersJoint,
    TestArticulationWritersRoot,
    TestArticulationWritersTendon,
    test_deprecated_joint_friction_writers,
)
from ._articulation_ordering_contract_cases import (  # noqa: F401
    TestArticulationOrderingBodyWriters,
    TestArticulationOrderingComWrites,
    TestArticulationOrderingJointWriters,
    TestArticulationOrderingOperations,
    TestArticulationOrderingRootWriteParity,
    TestArticulationOrderingWriteParity,
)
from ._rigid_object_collection_contract_cases import (  # noqa: F401
    TestCollectionCacheInvalidation,
    TestCollectionPartialWriteCells,
    TestCollectionWritersBody,
    TestCollectionWritersPose,
)
from ._rigid_object_contract_cases import (  # noqa: F401
    TestRigidObjectCacheInvalidation,
    TestRigidObjectPartialWriteRows,
    TestRigidObjectWritersBody,
    TestRigidObjectWritersRoot,
)

pytestmark = pytest.mark.integration
