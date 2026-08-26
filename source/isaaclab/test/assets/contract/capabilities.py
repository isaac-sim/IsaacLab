# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit backend capability declarations for shared asset contracts."""

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.util import find_spec

import pytest
import warp as wp
from _pytest.mark.structures import ParameterSet


@dataclass(frozen=True)
class BackendDeclaration:
    """Declare dependencies and supported shared-contract families for a backend."""

    name: str
    required_modules: tuple[str, ...]
    capabilities: frozenset[str]
    unsupported: dict[str, str] = field(default_factory=dict)
    requires_cuda_runtime: bool = False


@dataclass(frozen=True)
class BackendStatus:
    """Describe whether one declared backend can run shared contracts."""

    declaration: BackendDeclaration
    available: bool
    reason: str | None


_SHARED_CAPABILITIES = frozenset({"api", "data", "writes", "ordering", "fixed_tendons", "cuda"})
_EXTENDED_ARTICULATION_CAPABILITIES = frozenset(
    {
        "com_orientation_write",
        "fixed_tendon_extended_data",
        "fixed_tendon_extended_write_index",
        "fixed_tendon_write_mask",
        "fixed_tendon_write_to_sim_index",
        "fixed_tendon_write_to_sim_mask",
    }
)

BACKEND_DECLARATIONS = (
    BackendDeclaration(
        name="physx",
        required_modules=("carb", "isaaclab_physx"),
        capabilities=_SHARED_CAPABILITIES
        | _EXTENDED_ARTICULATION_CAPABILITIES
        | {"index_resolution", "spatial_tendons"},
    ),
    BackendDeclaration(
        name="newton",
        required_modules=("isaaclab_newton",),
        capabilities=_SHARED_CAPABILITIES | {"index_resolution"},
        unsupported={
            "com_orientation_write": "Newton stores center-of-mass position but not orientation",
            "fixed_tendon_extended_data": "Newton does not expose extended fixed-tendon data",
            "fixed_tendon_extended_write_index": "Newton does not implement extended fixed-tendon index writers",
            "fixed_tendon_write_mask": "Newton does not implement fixed-tendon mask writers",
            "fixed_tendon_write_to_sim_index": (
                "Newton fixed-tendon write-to-sim does not resolve a default environment selector"
            ),
            "fixed_tendon_write_to_sim_mask": "Newton does not implement fixed-tendon mask write-to-sim",
            "spatial_tendons": "Newton does not support spatial tendons",
        },
    ),
    BackendDeclaration(
        name="ovphysx",
        required_modules=("ovphysx", "isaaclab_ov"),
        capabilities=_SHARED_CAPABILITIES | _EXTENDED_ARTICULATION_CAPABILITIES | {"spatial_tendons"},
        unsupported={"index_resolution": "OVPhysX does not expose the shared index-resolution helpers"},
        # The mock contract allocates pinned host staging buffers even for CPU tensors.
        requires_cuda_runtime=True,
    ),
)


def evaluate_backend(
    declaration: BackendDeclaration,
    module_available: Callable[[str], bool],
    cuda_available: bool,
) -> BackendStatus:
    """Evaluate one backend declaration against the current process."""
    for module_name in declaration.required_modules:
        if not module_available(module_name):
            return BackendStatus(declaration, available=False, reason=f"missing required module: {module_name}")
    if declaration.requires_cuda_runtime and not cuda_available:
        return BackendStatus(declaration, available=False, reason="CUDA runtime unavailable")
    return BackendStatus(declaration, available=True, reason=None)


def backend_parameters(
    capability: str,
    *,
    statuses: tuple[BackendStatus, ...] | None = None,
    names: tuple[str, ...] | None = None,
) -> list[ParameterSet]:
    """Return explicit pytest parameters for one backend capability."""
    use_runtime_statuses = statuses is None
    if statuses is None:
        statuses = BACKEND_STATUSES
    if names is not None:
        statuses = tuple(status for status in statuses if status.declaration.name in names)

    parameters = []
    for status in statuses:
        declaration = status.declaration
        reason = status.reason
        if reason is None and capability == "cuda" and use_runtime_statuses and not _CUDA_AVAILABLE:
            reason = "CUDA runtime unavailable"
        if reason is None and capability not in declaration.capabilities:
            reason = declaration.unsupported.get(capability)
            if reason is None:
                raise ValueError(f"{declaration.name} has no declaration for capability {capability!r}")
        marks = () if reason is None else pytest.mark.skip(reason=reason)
        parameters.append(pytest.param(declaration.name, marks=marks, id=declaration.name))
    return parameters


def contract_backend(capability: str, *, names: tuple[str, ...] | None = None) -> Callable:
    """Parametrize a contract test with its explicit backend capability."""
    parametrize = pytest.mark.parametrize("backend", backend_parameters(capability, names=names), indirect=False)

    def decorator(test_function: Callable) -> Callable:
        decorated = parametrize(test_function)
        decorated.contract_backend_capability = capability
        return decorated

    return decorator


def available_backends(capability: str) -> list[str]:
    """Return available backends that explicitly support a capability."""
    return [
        status.declaration.name
        for status in BACKEND_STATUSES
        if status.available and capability in status.declaration.capabilities
    ]


def require_backend_capability(backend: str, capability: str) -> None:
    """Skip the current case with the declared reason when a backend lacks a capability."""
    declaration = next(declaration for declaration in BACKEND_DECLARATIONS if declaration.name == backend)
    if capability in declaration.capabilities:
        return
    reason = declaration.unsupported.get(capability)
    if reason is None:
        raise ValueError(f"{backend} has no declaration for capability {capability!r}")
    pytest.skip(reason)


def backend_unavailable_reasons() -> dict[str, str]:
    """Return explicit reasons for backends unavailable to this contract process."""
    return {status.declaration.name: status.reason for status in BACKEND_STATUSES if status.reason is not None}


def _module_available(module_name: str) -> bool:
    """Return whether an explicitly declared optional module can be resolved."""
    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


_CUDA_AVAILABLE = wp.is_cuda_available()
BACKEND_STATUSES = tuple(
    evaluate_backend(declaration, _module_available, _CUDA_AVAILABLE) for declaration in BACKEND_DECLARATIONS
)
