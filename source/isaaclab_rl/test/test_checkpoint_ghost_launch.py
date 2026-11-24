# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint selection of the RSL-RL play script, which Checkpoint Ghosts also use."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("rsl_rl")

from isaaclab_rl.entrypoints.backends import play_rsl_rl


def test_checkpoint_selectors_pass_the_library_and_agent_to_the_resolver(monkeypatch):
    resolver = Mock(return_value="/runs/model_5.pt")
    monkeypatch.setattr(play_rsl_rl, "resolve_checkpoint_selector", resolver)
    args = SimpleNamespace(checkpoint="latest", task="Isaac-Reach-Franka", agent="rsl_rl_cfg_entry_point")

    path = play_rsl_rl._resolve_checkpoint(args, Mock(), Mock(), "/logs/rsl_rl/reach_franka")

    assert path == "/runs/model_5.pt"
    assert resolver.call_args.kwargs["library"] == "rsl_rl"
    assert resolver.call_args.kwargs["metadata"] == {"agent": "rsl_rl_cfg_entry_point"}


def test_the_ghost_manifest_name_matches_the_one_training_writes():
    from isaaclab.visualizers import checkpoint_ghost_play

    from isaaclab_rl.entrypoints.common import RUN_MANIFEST_FILENAME

    assert checkpoint_ghost_play._RUN_MANIFEST_FILENAME == RUN_MANIFEST_FILENAME  # core cannot import it
