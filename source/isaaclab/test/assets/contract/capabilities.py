# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit backend capability declarations for the shared asset contracts.

Every contract test is parametrized over all declared backends. A backend that is unavailable in this process, or
that declares a capability unsupported, still collects as a skipped case with the declared reason, so a missing
backend or an unsupported branch never silently disappears from the matrix.
"""

from dataclasses import dataclass, field
from importlib.util import find_spec

import pytest
import warp as wp
from _pytest.mark.structures import ParameterSet

# Install the kitless import stubs before resolving the declared backend dependencies.
from . import _contract_boot  # noqa: F401


@dataclass(frozen=True)
class BackendDeclaration:
    """Dependencies and supported contract capabilities of one backend."""

    name: str
    required_modules: tuple[str, ...]
    capabilities: frozenset[str]
    unsupported: dict[str, str] = field(default_factory=dict)
    requires_cuda_runtime: bool = False


@dataclass(frozen=True)
class BackendStatus:
    """Whether one declared backend can run the contracts in this process."""

    declaration: BackendDeclaration
    available: bool
    reason: str | None


_SHARED_CAPABILITIES = frozenset({"api", "data", "writes", "ordering", "fixed_tendons"})
_PHYSX_FAMILY_CAPABILITIES = frozenset(
    {"fixed_tendon_extended", "spatial_tendons", "fixed_tendon_target_scheduling", "mocked_dynamics"}
)

BACKEND_DECLARATIONS = (
    BackendDeclaration(
        name="physx",
        required_modules=("carb", "isaaclab_physx"),
        capabilities=_SHARED_CAPABILITIES | _PHYSX_FAMILY_CAPABILITIES | {"index_resolution"},
    ),
    BackendDeclaration(
        name="newton",
        required_modules=("isaaclab_newton",),
        capabilities=_SHARED_CAPABILITIES | {"index_resolution"},
        unsupported={
            "fixed_tendon_extended": "Newton does not implement fixed-tendon limit stiffness, rest length, or offset",
            "spatial_tendons": "Newton does not support spatial tendons",
            "fixed_tendon_target_scheduling": (
                "Newton commands tendon targets through a MuJoCo tendon actuator the mocked contract does not build"
            ),
            "mocked_dynamics": "Newton computes task-space dynamics from a solver model that the mocked contract lacks",
        },
    ),
    BackendDeclaration(
        name="ovphysx",
        required_modules=("ovphysx", "isaaclab_ov"),
        capabilities=_SHARED_CAPABILITIES | _PHYSX_FAMILY_CAPABILITIES,
        unsupported={"index_resolution": "OVPhysX does not expose the joint and body index-resolution helpers"},
        # The mocked bindings allocate pinned host staging buffers even for CPU tensors.
        requires_cuda_runtime=True,
    ),
)


def _module_available(module_name: str) -> bool:
    """Return whether an explicitly declared dependency can be resolved."""
    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _evaluate_backend(declaration: BackendDeclaration, cuda_available: bool) -> BackendStatus:
    """Evaluate one backend declaration against the current process."""
    for module_name in declaration.required_modules:
        if not _module_available(module_name):
            return BackendStatus(declaration, available=False, reason=f"missing required module: {module_name}")
    if declaration.requires_cuda_runtime and not cuda_available:
        return BackendStatus(declaration, available=False, reason="CUDA runtime unavailable")
    return BackendStatus(declaration, available=True, reason=None)


BACKEND_STATUSES = tuple(_evaluate_backend(declaration, wp.is_cuda_available()) for declaration in BACKEND_DECLARATIONS)
_STATUS_BY_NAME = {status.declaration.name: status for status in BACKEND_STATUSES}


def _unsupported_reason(declaration: BackendDeclaration, capability: str) -> str | None:
    """Return why a backend does not run a capability, or None when it does."""
    if capability in declaration.capabilities:
        return None
    reason = declaration.unsupported.get(capability)
    if reason is None:
        raise ValueError(f"{declaration.name} has no declaration for capability {capability!r}")
    return reason


def backend_parameters(capability: str, *, names: tuple[str, ...] | None = None) -> list[ParameterSet]:
    """Return one pytest parameter per declared backend, skipped with its reason when it cannot run.

    Args:
        capability: Contract capability the test exercises.
        names: Optional subset of backend names for a backend-specific branch. Defaults to every backend.

    Returns:
        Backend parameters in declaration order.
    """
    parameters = []
    for status in BACKEND_STATUSES:
        declaration = status.declaration
        if names is not None and declaration.name not in names:
            continue
        reason = status.reason or _unsupported_reason(declaration, capability)
        marks = () if reason is None else pytest.mark.skip(reason=reason)
        parameters.append(pytest.param(declaration.name, marks=marks, id=declaration.name))
    return parameters


def contract_backend(capability: str, *, names: tuple[str, ...] | None = None) -> pytest.MarkDecorator:
    """Parametrize ``backend`` over the declared backends for one capability."""
    return pytest.mark.parametrize("backend", backend_parameters(capability, names=names))


def requires_backend(name: str) -> pytest.MarkDecorator:
    """Skip a backend-specific test with the declared reason when that backend is unavailable."""
    reason = _STATUS_BY_NAME[name].reason
    return pytest.mark.skipif(reason is not None, reason=f"{name}: {reason}")


def require_backend_capability(backend: str, capability: str) -> None:
    """Skip the current case with the declared reason when a backend lacks a capability."""
    reason = _unsupported_reason(_STATUS_BY_NAME[backend].declaration, capability)
    if reason is not None:
        pytest.skip(reason)


def available_backends() -> list[str]:
    """Return the declared backends that can run in this process."""
    return [status.declaration.name for status in BACKEND_STATUSES if status.available]
