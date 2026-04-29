# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wire-up validator for consumer capability requirements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..base_scene_data_provider import BaseSceneDataProvider


class CapabilityRequirementError(RuntimeError):
    """Raised when one or more registered consumers have unmet capability
    requirements against the active Scene Data Provider.

    The error message lists every consumer with at least one missing
    capability, so a single error report covers all wire-up failures.
    """


def validate_consumer_capabilities(
    provider: BaseSceneDataProvider,
    consumers: list[Any],
) -> None:
    """Validate that every consumer's capability requirements are satisfied.

    Each consumer may declare two ``ClassVar`` attributes:

    - ``required_capabilities``: a tuple of capability types, all of which
      must be present.
    - ``required_one_of``: a tuple of tuples; from each inner tuple at least
      one capability type must be present.

    Args:
        provider: The active Scene Data Provider.
        consumers: All registered consumer instances.

    Raises:
        CapabilityRequirementError: One or more consumers have unmet
            requirements. The message lists every offender.
    """
    available = provider.list_capabilities()
    failures: list[str] = []

    for consumer in consumers:
        consumer_label = _label(consumer)
        missing_required: list[type] = []
        missing_one_of: list[tuple[type, ...]] = []

        for cap_type in getattr(consumer, "required_capabilities", ()):
            if cap_type not in available:
                missing_required.append(cap_type)

        for group in getattr(consumer, "required_one_of", ()):
            if not any(c in available for c in group):
                missing_one_of.append(tuple(group))

        if missing_required or missing_one_of:
            parts: list[str] = []
            if missing_required:
                parts.append(
                    "missing required: ["
                    + ", ".join(_cap_name(c) for c in missing_required)
                    + "]"
                )
            if missing_one_of:
                parts.append(
                    "missing one-of: ["
                    + ", ".join(
                        "(" + " | ".join(_cap_name(c) for c in g) + ")"
                        for g in missing_one_of
                    )
                    + "]"
                )
            failures.append(f"  - {consumer_label}: " + "; ".join(parts))

    if failures:
        offered = ", ".join(sorted(_cap_name(c) for c in available)) or "<none>"
        raise CapabilityRequirementError(
            "Consumer capability requirements not met by the active Scene"
            " Data Provider. Offered capabilities: ["
            + offered
            + "]\n"
            + "\n".join(failures)
        )


def _cap_name(cap_type: type) -> str:
    return f"{cap_type.__module__}.{cap_type.__name__}"


def _label(consumer: Any) -> str:
    cls = type(consumer)
    return f"{cls.__module__}.{cls.__name__}"
