# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the pretrained-checkpoint training utility."""

from argparse import Namespace

from scripts.tools.train_and_publish_checkpoints import (
    CheckpointJob,
    _play_command,
    _select_physics_variants,
    _training_command,
)


def test_job_commands_use_uv_run_isaaclab() -> None:
    """Training and playback must use the uv-managed Isaac Lab CLI."""
    job = CheckpointJob(
        workflow="rsl_rl",
        task_name="Isaac-Test",
        physics_backend="physx",
        render_backend="none",
        physics_selector="isaacsim_physx",
    )
    args = Namespace(max_iterations=None, num_envs=None)

    train_command = _training_command(job, args, smoke=False)
    play_command = _play_command(job, args, "/tmp/checkpoint.pt")

    assert train_command[:4] == ["uv", "run", "isaaclab", "train"]
    assert play_command[:4] == ["uv", "run", "isaaclab", "play"]
    assert train_command[-1] == "physics=isaacsim_physx"
    assert play_command[-1] == "physics=isaacsim_physx"


def test_select_physics_variants_uses_concrete_isaac_sim_physx() -> None:
    """The normalized PhysX job must not resolve through the automatic selector."""
    variants = ["physx", "isaacsim_physx", "ovphysx", "newton_mjwarp"]

    selections = _select_physics_variants("Isaac-Test", variants, "physx", ["physx", "newton"])

    assert selections == [("physx", "isaacsim_physx"), ("newton", "newton_mjwarp")]


def test_select_physics_variants_does_not_fall_back_to_automatic_physx() -> None:
    """A task without a concrete Isaac Sim selector must not run as OvPhysX."""
    selections = _select_physics_variants("Isaac-Test", ["physx", "ovphysx"], "physx", ["physx"])

    assert selections == []
