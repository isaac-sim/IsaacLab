# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton environment smoke tests.

The MJWarp preset is a kit-less backend, so this suite runs without ``AppLauncher``. Tasks carrying
Kit camera sensors still need the Kit runtime and are filtered out; ``test_environments.py`` already
smoke-tests each of them on its default backend.
"""

import os

# Limit OpenUSD's work-thread pool to one thread to avoid a race condition in usd-core<26.5, which
# corrupts native state while collecting collider descriptors and aborts the process during the
# Newton USD import. Set at module scope rather than in a fixture, unlike the same workaround in
# rendering_test_utils.py: kit-less, OpenUSD builds its task arena while tests are collected, and
# WorkSetConcurrencyLimit does not shrink an arena that already exists.
# TODO: Remove once usd-core>=26.5 is the minimum - that release fixes the race condition.
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

import pytest

import isaaclab_tasks  # noqa: F401

# Local imports should be imported last
from env_test_utils import _run_environments, setup_environment  # isort: skip


@pytest.mark.parametrize("num_envs, device", [(2, "cuda"), (1, "cuda")])
@pytest.mark.parametrize(
    "task_name",
    setup_environment(
        include_play=False,
        multi_agent=False,
        newton_mjwarp_envs=True,
        tier="core",
        needs_kit=False,
        physics_preset_name="newton_mjwarp",
    ),
)
@pytest.mark.newton_ci
def test_environments_newton(task_name, num_envs, device):
    # run environments with MJWarp physics preset
    _run_environments(task_name, device, num_envs, physics_preset_name="newton_mjwarp", create_stage_in_memory=False)
