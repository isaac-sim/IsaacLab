# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heterogeneous tasks for vision-language-action policies: a Franka with a Robotiq gripper manipulating objects, on
Newton.

The tasks share the environment configuration, objects and success terms of this package. ``tasks`` holds the tasks
and the environment whose environments each run one of them, switched on reset by
:class:`~isaaclab.managers.HeterogeneousTaskManager`; ``config/<robot>`` registers them for a robot.
"""

import os

_ASSETS_REPO = "LiFanxing/HeteroTask"
"""Hugging Face dataset holding the tasks' assets."""

_ASSETS_REVISION = "f7eacb6db16e3cc4479c5f620686337edc4e209e"
"""Commit of :data:`_ASSETS_REPO` the tasks use, so they do not change when the dataset does."""


def _assets_dir() -> str:
    """Return the environment variable's location, the Hugging Face cache's copy of the revision, or its URL."""
    if os.environ.get("VLA_BENCHMARKS_ASSETS_DIR"):
        return os.environ["VLA_BENCHMARKS_ASSETS_DIR"]
    url = f"https://huggingface.co/datasets/{_ASSETS_REPO}/resolve/{_ASSETS_REVISION}"
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:
        return url
    try:
        return snapshot_download(_ASSETS_REPO, repo_type="dataset", revision=_ASSETS_REVISION, local_files_only=True)
    except LocalEntryNotFoundError:
        return url


# TODO: Point to the Isaac Lab Nucleus once the assets are uploaded there.
VLA_BENCHMARKS_ASSETS_DIR = _assets_dir()
"""Root of the tasks' assets: the objects from the RoboLab and Omniverse libraries as complete USD files, and the
robot table and dome light texture. Taken from the environment variable of the same name if set, e.g. a local copy;
otherwise from the Hugging Face cache when it holds the dataset's fixed commit, which
``hf download LiFanxing/HeteroTask --repo-type dataset --revision <commit>`` puts there (``HF_HUB_CACHE`` selects the
cache directory); otherwise from that commit on Hugging Face. A local copy avoids checking every cached remote file
against the server on each run. The robot is that of :obj:`~isaaclab_assets.robots.franka.FRANKA_ROBOTIQ_GRIPPER_CFG`.
"""
