# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: F401

"""Shared asset API contract tests."""

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
from ._rigid_object_collection_contract_cases import (  # noqa: F401
    TestCollectionFinderReturnModes,
    TestCollectionFinders,
    TestCollectionIndexResolution,
    TestCollectionProperties,
    collection_iface,
)
from ._rigid_object_contract_cases import (  # noqa: F401
    TestRigidObjectFinderReturnModes,
    TestRigidObjectFinders,
    TestRigidObjectIndexResolution,
    TestRigidObjectProperties,
    rigid_object_iface,
)
from .capabilities import BackendDeclaration, BackendStatus, backend_parameters, evaluate_backend
from .public_surface import (
    BASE_SURFACE_CLASSES,
    PUBLIC_SURFACE_CLASSIFICATIONS,
    ContractKind,
    unclassified_public_members,
)


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


def test_base_public_surface_has_an_explicit_contract_classification() -> None:
    """Require every public member declared by the shared asset bases to have an explicit classification."""
    unclassified = unclassified_public_members(BASE_SURFACE_CLASSES, PUBLIC_SURFACE_CLASSIFICATIONS)

    assert unclassified == set(), f"Unclassified public asset members: {sorted(unclassified)}"
