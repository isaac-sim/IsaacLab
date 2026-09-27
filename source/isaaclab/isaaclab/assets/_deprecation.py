# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecation warnings shared by the asset base classes."""

from __future__ import annotations

import warnings


def warn_renamed_function(old_name: str, new_name: str) -> None:
    """Warn that a deprecated function forwards to its renamed replacement.

    The warning is attributed to the caller of the deprecated function.
    """
    warnings.warn(
        f"The function '{old_name}' will be deprecated in a future release. Please use '{new_name}' instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def warn_renamed_member(
    old_name: str, new_name: str, *, kind: str, release: str = "a future release", detail: str = ""
) -> None:
    """Warn that a deprecated property or method forwards to its renamed replacement.

    Args:
        old_name: Deprecated member name.
        new_name: Replacement member name.
        kind: Member kind used in the message, e.g. ``"property"`` or ``"method"``.
        release: Release in which the member is removed.
        detail: Optional sentence appended to the message.
    """
    warnings.warn(
        f"The `{old_name}` {kind} will be deprecated in {release}. Please use `{new_name}` instead.{detail}",
        DeprecationWarning,
        stacklevel=3,
    )
