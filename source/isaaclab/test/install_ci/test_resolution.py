# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Catch dependencies that cannot be resolved without the committed lock."""

from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.mark.resolve
def test_dependency_graph_resolves(checkout: Path, run: Callable[..., str]) -> None:
    run("uv", "lock", "--upgrade", "--dry-run", cwd=checkout)
