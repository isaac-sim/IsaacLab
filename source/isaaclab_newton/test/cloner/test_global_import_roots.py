# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for clone-external Newton USD import scoping."""

from isaaclab_newton.cloner.replicate import _global_import_roots

from pxr import Usd


def test_global_import_roots_excludes_replicated_environment_tree():
    """Global import roots contain clone-external branches without entering ``/World/envs``."""
    stage = Usd.Stage.CreateInMemory()
    for path in (
        "/physicsScene",
        "/World/envs/env_0/Robot",
        "/World/envs/env_1/Robot",
        "/World/ground",
        "/GlobalAsset",
    ):
        stage.DefinePrim(path, "Xform")

    roots = _global_import_roots(stage, ["/World/envs/env_0"])

    assert set(roots) == {"/physicsScene", "/World/ground", "/GlobalAsset"}
    assert all(not path.startswith("/World/envs") for path in roots)


def test_global_import_roots_preserves_heterogeneous_prototype_exclusion():
    """Asset-level heterogeneous sources still exclude their shared environment branch."""
    stage = Usd.Stage.CreateInMemory()
    for path in (
        "/physicsScene",
        "/World/envs/env_0/Object",
        "/World/envs/env_1/Object",
        "/World/ground",
    ):
        stage.DefinePrim(path, "Xform")

    roots = _global_import_roots(stage, ["/World/envs/env_0/Object", "/World/envs/env_1/Object"])

    assert set(roots) == {"/physicsScene", "/World/ground"}
