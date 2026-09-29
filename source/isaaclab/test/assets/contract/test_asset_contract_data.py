# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared asset data contract: property shapes and dtypes, aliases, lazy buffers, and ordered reads."""

import pytest

from ._articulation_contract_cases import (  # noqa: F401
    TestArticulationDataAliases,
    TestArticulationDataProperties,
    TestArticulationTendons,
)
from ._articulation_ordering_contract_cases import (  # noqa: F401
    TestArticulationOrderingAllocation,
    TestArticulationOrderingBodyState,
    TestArticulationOrderingJointState,
)
from ._rigid_object_collection_contract_cases import (  # noqa: F401
    TestCollectionDataAliases,
    TestCollectionDataProperties,
    TestCollectionViewReshape,
)
from ._rigid_object_contract_cases import (  # noqa: F401
    TestRigidObjectDataAliases,
    TestRigidObjectDataProperties,
)

pytestmark = pytest.mark.integration
