# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: F401

"""Shared asset API contract tests."""

from importlib.util import find_spec

import pytest

pytestmark = pytest.mark.unit

from . import public_surface
from ._articulation_contract_cases import TestArticulationDataRootState as _ArticulationDataContract
from ._articulation_contract_cases import TestArticulationDataTendonState as _ArticulationTendonDataContract
from ._articulation_contract_cases import (  # noqa: F401
    TestArticulationFinderReturnModes,
    TestArticulationFinders,
    TestArticulationIndexResolution,
    TestArticulationProperties,
    TestArticulationTendonFinders,
    TestArticulationTendonProperties,
    TestResolveMatchingNamesCache,
    articulation_iface,
)
from ._articulation_contract_cases import TestArticulationWritersRoot as _ArticulationWriteContract
from ._articulation_ordering_contract_cases import TestArticulationOrderingAllocation as _ArticulationOrderingContract
from ._rigid_object_collection_contract_cases import (
    TestCollectionCacheInvalidation as _CollectionCacheInvalidationContract,
)
from ._rigid_object_collection_contract_cases import (  # noqa: F401
    TestCollectionFinderReturnModes,
    TestCollectionFinders,
    TestCollectionIndexResolution,
    TestCollectionProperties,
    collection_iface,
)
from ._rigid_object_contract_cases import TestRigidObjectCacheInvalidation as _RigidObjectCacheInvalidationContract
from ._rigid_object_contract_cases import (  # noqa: F401
    TestRigidObjectFinderReturnModes,
    TestRigidObjectFinders,
    TestRigidObjectIndexResolution,
    TestRigidObjectProperties,
    rigid_object_iface,
)
from .capabilities import (
    BACKEND_STATUSES,
    BackendDeclaration,
    BackendStatus,
    backend_parameters,
    evaluate_backend,
)
from .public_surface import (
    BASE_SURFACE_CLASSES,
    PUBLIC_SURFACE_CLASSIFICATIONS,
    ContractKind,
    unclassified_public_members,
)


def _available_contract_manager_globals() -> dict[str, object]:
    """Return installed backend bindings touched by the mock factories."""
    import inspect

    bindings: dict[str, object] = {}
    if "physx" in BACKEND_STATUSES_BY_NAME and BACKEND_STATUSES_BY_NAME["physx"].available:
        from isaaclab_physx.physics import PhysxManager

        bindings["physx.get_physics_sim_view"] = inspect.getattr_static(PhysxManager, "get_physics_sim_view")
    if "newton" in BACKEND_STATUSES_BY_NAME and BACKEND_STATUSES_BY_NAME["newton"].available:
        import isaaclab_newton.assets.articulation.articulation_data as articulation_data
        import isaaclab_newton.assets.rigid_object.rigid_object_data as rigid_object_data
        import isaaclab_newton.assets.rigid_object_collection.rigid_object_collection as rigid_object_collection
        from isaaclab_newton.assets.rigid_object_collection import (
            rigid_object_collection_data,
        )

        bindings.update(
            {
                "newton.articulation_data": articulation_data.SimulationManager,
                "newton.rigid_object_data": rigid_object_data.SimulationManager,
                "newton.collection": rigid_object_collection.SimulationManager,
                "newton.collection_data": rigid_object_collection_data.SimulationManager,
            }
        )
    return bindings


def _assert_identical_bindings(actual: dict[str, object], expected: dict[str, object]) -> None:
    """Assert exact binding identity without invoking mock equality operators."""
    assert actual.keys() == expected.keys()
    assert all(actual[name] is original for name, original in expected.items())


BACKEND_STATUSES_BY_NAME = {status.declaration.name: status for status in BACKEND_STATUSES}


@pytest.mark.parametrize("reverse", [False, True])
def test_contract_factory_manager_patch_supports_cross_suite_asset_access(reverse: bool) -> None:
    """Keep each manager available while rigid, collection, and articulation methods run."""
    from ._articulation_contract_utils import get_articulation
    from ._manager_patch_scope import contract_manager_patch_scope
    from ._rigid_object_collection_contract_utils import get_rigid_object_collection
    from ._rigid_object_contract_utils import get_rigid_object

    available_backends = [name for name in ("physx", "newton") if BACKEND_STATUSES_BY_NAME[name].available]
    if reverse:
        available_backends.reverse()
    original_bindings = _available_contract_manager_globals()

    with contract_manager_patch_scope():
        for backend in available_backends:
            rigid_object, _ = get_rigid_object(backend, device="cpu")
            collection, _ = get_rigid_object_collection(backend, device="cpu")
            articulation, _ = get_articulation(backend, device="cpu")

            assert rigid_object.data.root_link_pose_w.shape == (2,)
            assert collection.data.body_link_pose_w.shape == (2, 3)
            assert articulation.data.root_link_pose_w.shape == (2,)

    _assert_identical_bindings(_available_contract_manager_globals(), original_bindings)


def test_contract_factory_modules_keep_production_manager_bindings_after_collection() -> None:
    """Keep optional backend imports free of import-time manager substitutions."""
    import inspect

    if BACKEND_STATUSES_BY_NAME["physx"].available:
        from isaaclab_physx.physics import PhysxManager

        assert isinstance(inspect.getattr_static(PhysxManager, "get_physics_sim_view"), classmethod)
    if BACKEND_STATUSES_BY_NAME["newton"].available:
        from isaaclab_newton.physics import NewtonManager

        bindings = _available_contract_manager_globals()
        assert all(manager is NewtonManager for name, manager in bindings.items() if name.startswith("newton."))


def test_backend_declaration_reports_missing_required_module() -> None:
    """Classify a declared backend as unavailable when an explicit dependency is missing."""
    declaration = BackendDeclaration(
        name="example",
        required_modules=("present", "missing"),
        capabilities=frozenset({"api"}),
    )

    status = evaluate_backend(declaration, lambda name: name == "present", cuda_available=True)

    assert not status.available
    assert status.reason == "missing required module: missing"


def test_backend_declaration_reports_required_cuda_runtime() -> None:
    """Classify a declared backend as unavailable when its mock contract requires CUDA."""
    declaration = BackendDeclaration(
        name="example",
        required_modules=("present",),
        capabilities=frozenset({"api"}),
        requires_cuda_runtime=True,
    )

    status = evaluate_backend(declaration, lambda name: True, cuda_available=False)

    assert not status.available
    assert status.reason == "CUDA runtime unavailable"


def test_backend_parameters_keep_unavailable_backends_as_explicit_skips() -> None:
    """Represent an unavailable declared backend explicitly instead of returning an empty matrix."""
    declaration = BackendDeclaration(
        name="example",
        required_modules=("missing",),
        capabilities=frozenset({"api"}),
    )
    status = BackendStatus(declaration, available=False, reason="missing required module: missing")

    parameters = backend_parameters("api", statuses=(status,))

    assert len(parameters) == 1
    assert parameters[0].values == ("example",)
    assert parameters[0].marks[0].kwargs["reason"] == "missing required module: missing"


def test_backend_parameters_keep_unsupported_behavior_as_explicit_skip() -> None:
    """Represent explicitly unsupported behavior with the declared backend-specific reason."""
    declaration = BackendDeclaration(
        name="example",
        required_modules=("present",),
        capabilities=frozenset({"api"}),
        unsupported={"writes": "write path is intentionally unsupported"},
    )
    status = BackendStatus(declaration, available=True, reason=None)

    parameters = backend_parameters("writes", statuses=(status,))

    assert len(parameters) == 1
    assert parameters[0].values == ("example",)
    assert parameters[0].marks[0].kwargs["reason"] == "write path is intentionally unsupported"


def test_installed_backend_packages_cannot_remain_silently_unavailable() -> None:
    """Require every installed declared backend package to have a runnable contract status."""
    unavailable = {
        status.declaration.name: status.reason
        for status in BACKEND_STATUSES
        if find_spec(status.declaration.required_modules[-1]) is not None
        and not status.declaration.requires_cuda_runtime
        and not status.available
    }

    assert unavailable == {}


@pytest.mark.parametrize(
    ("test_method", "expected_capability"),
    [
        (_ArticulationDataContract.test_root_link_pose_w, "data"),
        (_ArticulationWriteContract.test_write_root_pose_to_sim_index, "writes"),
        (_RigidObjectCacheInvalidationContract.test_pose_write_invalidates_pose_dependent_caches, "writes"),
        (_CollectionCacheInvalidationContract.test_pose_write_invalidates_pose_dependent_caches, "writes"),
        (_ArticulationOrderingContract.test_backends_allocate_shadows_only_for_nonidentity_ordering, "ordering"),
        (_ArticulationTendonDataContract.test_fixed_tendon_stiffness, "fixed_tendons"),
        (_ArticulationTendonDataContract.test_spatial_tendon_stiffness, "spatial_tendons"),
    ],
)
def test_contract_cases_declare_their_distinct_backend_capability(test_method, expected_capability: str) -> None:
    """Classify data, write, ordering, and tendon cases with their actual backend capability."""
    assert getattr(test_method, "contract_backend_capability", None) == expected_capability


def test_newton_fixed_tendon_contract_is_not_skipped_as_spatially_unsupported() -> None:
    """Keep Newton fixed-tendon coverage runnable while its spatial-tendon behavior remains unsupported."""
    backend_mark = next(
        mark
        for mark in _ArticulationTendonDataContract.test_fixed_tendon_stiffness.pytestmark
        if mark.name == "parametrize" and mark.args[0] == "backend"
    )
    newton_parameter = next(parameter for parameter in backend_mark.args[1] if parameter.values == ("newton",))

    assert not any(mark.name == "skip" for mark in newton_parameter.marks)


@pytest.mark.parametrize(
    "capability",
    [
        "com_orientation_write",
        "fixed_tendon_extended_data",
        "fixed_tendon_extended_write_index",
        "fixed_tendon_write_mask",
        "fixed_tendon_write_to_sim_index",
        "fixed_tendon_write_to_sim_mask",
    ],
)
def test_newton_partial_articulation_support_has_reasoned_capability_skips(capability: str) -> None:
    """Keep Newton's partial CoM and fixed-tendon support explicit in the contract matrix."""
    parameter = backend_parameters(capability, names=("newton",))[0]

    assert parameter.values == ("newton",)
    assert parameter.marks[0].name == "skip"
    assert parameter.marks[0].kwargs["reason"]


def test_public_surface_reports_an_omitted_declared_member() -> None:
    """Report a newly declared public member until a contract explicitly classifies it."""

    class SyntheticAsset:
        @property
        def added_member(self) -> int:
            return 1

    unclassified = unclassified_public_members((SyntheticAsset,), {})

    assert unclassified == {"SyntheticAsset.added_member"}


def test_public_surface_accepts_a_classified_declared_member() -> None:
    """Accept a declared public member after assigning its contract family."""

    class SyntheticAsset:
        @property
        def added_member(self) -> int:
            return 1

    classifications = {"SyntheticAsset.added_member": ContractKind.API}

    assert unclassified_public_members((SyntheticAsset,), classifications) == set()


def test_public_surface_audit_reports_a_missing_mapping() -> None:
    """Report a declared member that has no explicit contract mapping."""

    class SyntheticAsset:
        added_member = 1

    audit = public_surface.audit_public_surface((SyntheticAsset,), {})

    assert audit.missing == frozenset({"SyntheticAsset.added_member"})


def test_public_surface_audit_reports_a_stale_mapping() -> None:
    """Report a mapped member that is no longer declared by its base class."""

    class SyntheticAsset:
        added_member = 1

    mapping = {
        "SyntheticAsset.added_member": public_surface.PublicMemberContract(
            kind=ContractKind.API,
        ),
        "SyntheticAsset.removed_member": public_surface.PublicMemberContract(
            kind=ContractKind.API,
        ),
    }

    audit = public_surface.audit_public_surface((SyntheticAsset,), mapping)

    assert audit.stale == frozenset({"SyntheticAsset.removed_member"})


def test_public_surface_audit_reports_an_unreasoned_exclusion() -> None:
    """Reject unsupported or out-of-scope mappings that omit their rationale."""

    class SyntheticAsset:
        added_member = 1

    mapping = {
        "SyntheticAsset.added_member": public_surface.PublicMemberContract(
            kind=ContractKind.OUT_OF_SCOPE,
            reason=None,
        )
    }

    audit = public_surface.audit_public_surface((SyntheticAsset,), mapping)

    assert audit.unreasoned == frozenset({"SyntheticAsset.added_member"})


def test_base_public_surface_has_an_explicit_contract_classification() -> None:
    """Require the classification inventory and current public surface to match exactly."""
    audit = public_surface.audit_public_surface(BASE_SURFACE_CLASSES, public_surface.PUBLIC_SURFACE_CONTRACTS)

    assert audit.is_valid, audit.format_errors()
