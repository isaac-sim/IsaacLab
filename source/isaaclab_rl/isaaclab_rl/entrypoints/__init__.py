# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified train and play entrypoints for Isaac Lab reinforcement learning workflows.

The :func:`train` and :func:`play` functions accept typed requests, while
:func:`run_train_cli` and :func:`run_play_cli` dispatch raw command-line arguments.
Both select the backend implementation via the ``--rl_library`` argument or the
:attr:`TrainingRequest.backend` field.

The :func:`zero_agent` and :func:`random_agent` functions (and their
:func:`run_zero_agent_cli` and :func:`run_random_agent_cli` counterparts) run the
checkpoint-free variations of playback, which need no reinforcement learning backend.

Example:

.. code-block:: python

    from isaaclab_rl import TrainingRequest, train

    train(TrainingRequest(backend="rsl_rl", task="Isaac-Cartpole", max_iterations=100))
"""

from .api import BackendName, PlaybackRequest, SimpleAgentRequest, TrainingRequest, play, random_agent, train, zero_agent
from .dispatch import run_play_cli, run_random_agent_cli, run_train_cli, run_zero_agent_cli

__all__ = [
    "BackendName",
    "PlaybackRequest",
    "SimpleAgentRequest",
    "TrainingRequest",
    "play",
    "random_agent",
    "run_play_cli",
    "run_random_agent_cli",
    "run_train_cli",
    "run_zero_agent_cli",
    "train",
    "zero_agent",
]
