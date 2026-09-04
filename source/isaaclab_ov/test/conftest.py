# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest configuration for the isaaclab_ov test suite."""

import pytest


@pytest.fixture(autouse=True)
def _clear_async_rendering_env(monkeypatch):
    """Clear an ambient ``ISAAC_LAB_ASYNC_RENDERING`` for every test.

    Several fixtures resolve the render strategy from this variable. An ambient override would
    flip their fakes onto the asynchronous path and fail synchronous-only contracts. Tests that
    exercise the override set the variable themselves.
    """
    monkeypatch.delenv("ISAAC_LAB_ASYNC_RENDERING", raising=False)
