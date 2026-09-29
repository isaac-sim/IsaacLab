# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixtures shared by the focused asset contract gate."""

from collections.abc import Iterator

import pytest

from ._manager_patch_scope import contract_manager_patch_scope


@pytest.fixture(autouse=True)
def _restore_contract_manager_patches() -> Iterator[None]:
    """Limit factory manager substitutions to one test invocation."""
    with contract_manager_patch_scope():
        yield
