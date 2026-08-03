# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the pretrained-checkpoint training utility."""

from scripts.tools.train_and_publish_checkpoints import _select_physics_variants


def test_select_physics_variants_uses_concrete_isaac_sim_physx() -> None:
    """The normalized PhysX job must not resolve through the automatic selector."""
    variants = ["physx", "isaacsim_physx", "ovphysx", "newton_mjwarp"]

    selections = _select_physics_variants("Isaac-Test", variants, "physx", ["physx", "newton"])

    assert selections == [("physx", "isaacsim_physx"), ("newton", "newton_mjwarp")]


def test_select_physics_variants_does_not_fall_back_to_automatic_physx() -> None:
    """A task without a concrete Isaac Sim selector must not run as OvPhysX."""
    selections = _select_physics_variants("Isaac-Test", ["physx", "ovphysx"], "physx", ["physx"])

    assert selections == []
