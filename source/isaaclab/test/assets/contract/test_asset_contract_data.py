# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: F401

"""Shared asset data contract tests."""

import pytest

from ._articulation_contract_cases import (  # noqa: F401
    TestArticulationDataAliases,
    TestArticulationDataBodyState,
    TestArticulationDataDefaults,
    TestArticulationDataDerivedProperties,
    TestArticulationDataJointState,
    TestArticulationDataRootState,
    TestArticulationDataTendonState,
    articulation_iface,
)
from ._articulation_contract_utils import get_articulation
from ._articulation_ordering_contract_cases import (  # noqa: F401
    TestArticulationDataBodyState as TestArticulationOrderingDataBodyState,
)
from ._articulation_ordering_contract_cases import (
    TestArticulationDataJointState as TestArticulationOrderingDataJointState,
)
from ._articulation_ordering_contract_cases import (
    TestArticulationOrderingAllocation,
)
from ._rigid_object_collection_contract_cases import (  # noqa: F401
    TestCollectionCacheInvalidation,
    TestCollectionDataAliases,
    TestCollectionDataBodyState,
    TestCollectionDataDefaults,
    TestCollectionDataDerived,
    TestCollectionDataMass,
    TestCollectionDataSliced,
    TestCollectionViewReshape,
    collection_iface,
)
from ._rigid_object_collection_contract_utils import get_rigid_object_collection
from ._rigid_object_contract_cases import (  # noqa: F401
    TestRigidObjectCacheInvalidation,
    TestRigidObjectDataAliases,
    TestRigidObjectDataBodyState,
    TestRigidObjectDataDefaults,
    TestRigidObjectDataDerivedProperties,
    TestRigidObjectDataRootState,
    rigid_object_iface,
)
from ._rigid_object_contract_utils import get_rigid_object
from .capabilities import backend_parameters


@pytest.mark.parametrize("backend", backend_parameters("cuda"))
def test_articulation_data_has_one_cuda_smoke_per_backend(backend: str) -> None:
    """Exercise the distinct CUDA allocation path once for each articulation backend."""
    articulation, _ = get_articulation(backend, num_instances=2, num_joints=4, num_bodies=3, device="cuda:0")

    assert str(articulation.data.joint_pos.device) == "cuda:0"


@pytest.mark.parametrize("backend", backend_parameters("cuda"))
def test_rigid_object_data_has_one_cuda_smoke_per_backend(backend: str) -> None:
    """Exercise the distinct CUDA allocation path once for each rigid-object backend."""
    rigid_object, _ = get_rigid_object(backend, num_instances=2, device="cuda:0")

    assert str(rigid_object.data.root_link_pose_w.device) == "cuda:0"


@pytest.mark.parametrize("backend", backend_parameters("cuda"))
def test_collection_data_has_one_cuda_smoke_per_backend(backend: str) -> None:
    """Exercise the distinct CUDA allocation path once for each collection backend."""
    collection, _ = get_rigid_object_collection(backend, num_instances=2, num_bodies=3, device="cuda:0")

    assert str(collection.data.body_link_pose_w.device) == "cuda:0"
