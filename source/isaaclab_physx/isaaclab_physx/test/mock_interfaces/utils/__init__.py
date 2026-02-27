# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for mock PhysX interfaces."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .mock_shared_metatype import MockSharedMetatype
    from .patching import mock_articulation_view, mock_rigid_body_view, mock_rigid_contact_view, patch_articulation_view, patch_rigid_body_view, patch_rigid_contact_view

from isaaclab.utils.module import lazy_export

lazy_export(
    ("mock_shared_metatype", "MockSharedMetatype"),
    ("patching", [
        "mock_articulation_view",
        "mock_rigid_body_view",
        "mock_rigid_contact_view",
        "patch_articulation_view",
        "patch_rigid_body_view",
        "patch_rigid_contact_view",
    ]),
)
