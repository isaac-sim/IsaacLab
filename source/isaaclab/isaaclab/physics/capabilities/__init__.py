# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capability protocols for Scene Data Provider extensibility.

The capability registry replaces the implicit single-channel framing of
the original Scene Data Provider design. Each provider service — typed
GPU buffer access, USD Fabric freshness, Newton state access, customer
extensions — is expressed as a runtime-checkable :class:`typing.Protocol`.

Consumers query the active provider by capability *type*::

    cap = provider.get_capability(GpuTransformBuffer)
    if cap is not None:
        transforms = cap.get_body_transforms(TransformFormat.MAT44)

See :doc:`/source/refs/adr/0002-capability-based-provider-extensibility`
for the full design rationale.
"""

from ._protocols import GpuTransformBuffer, UsdFabric
from ._validator import (
    CapabilityRequirementError,
    validate_consumer_capabilities,
)

__all__ = [
    "CapabilityRequirementError",
    "GpuTransformBuffer",
    "UsdFabric",
    "validate_consumer_capabilities",
]
