# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for contributed environments that run without Isaac Sim."""

import os

# TODO: Remove once usd-core>=26.5 is the minimum. Earlier releases can corrupt
# the heap while parsing Newton payloads concurrently, so disable USD concurrency
# before importing modules that may initialize OpenUSD.
os.environ["PXR_WORK_THREAD_LIMIT"] = "1"

import pytest

import isaaclab_tasks  # noqa: F401

# Local imports should be imported last
from contrib_env_test_utils import contrib_environment_params, num_envs  # isort: skip
from env_test_utils import _run_environments  # isort: skip


@pytest.mark.parametrize("task_name", contrib_environment_params("kitless"))
def test_contrib_environments_kitless(task_name):
    _run_environments(task_name, device="cuda", num_envs=num_envs(task_name))
