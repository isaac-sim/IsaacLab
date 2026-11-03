# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve prebuilt USDs and optional read-only source references for physical asset tests."""

import hashlib
import os
from functools import cache
from pathlib import Path

import pytest

from isaaclab.utils.assets import retrieve_file_path

_SOURCE_REVISION = "d424a0c899f6b33cbd3daeb279913134349c0b63"
# Reference XML identities bind optional external fixtures to the original asset pin.
# They never select a runtime USD, fetch upstream files or invoke an importer.
_SOURCE_SHA256 = {
    "robot_walk.xml": "07af5e482200dfd2a7bf80ddde371153c88768a3a5aebc2f58ea42136fc7dadb",
    "robot_allcollisions.xml": "7a6fdf437f5a80c7348ad801f43f906b997a834389cb70cca5e1e8517ba38044",
    "robot_allcollisions_rollers.xml": "c201a72ecbc4dd8e110c840cb79b1647aab59e6c76db7369a88c34a33307447e",
    "ball.xml": "54a455bf454a9b6167655381df91593bbca86695d8d71e29fb6af69454c7c865",
}


@cache
def prebuilt_usd_path(path: str) -> str:
    """Resolve a hosted USD through the SDK cache; availability errors fail visibly."""
    return retrieve_file_path(path)


def spawn_test_assets(configs):
    """Build one physical fixture through the native homogeneous replication API."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject
    from isaaclab.cloner import CloneCfg, clone_plan_from_env_0, replicate
    from isaaclab.sim import SimulationContext

    sim_utils.create_prim("/World/envs/env_0", "Xform")
    clone_plan_from_env_0(CloneCfg(), configs.values(), num_envs=1, env_spacing=0.0)
    assets = {
        name: Articulation(cfg) if isinstance(cfg, ArticulationCfg) else RigidObject(cfg)
        for name, cfg in configs.items()
    }
    replicate(SimulationContext.instance().get_clone_plan())
    return assets


def source_mjcf_path(filename: str) -> str:
    """Read an explicitly supplied pinned MJCF fixture, without fetching or converting it."""
    single = os.environ.get("MICRODUCK_MJCF_PATH") if filename == "robot_walk.xml" else None
    directory = os.environ.get("MICRODUCK_MJCF_DIR")
    if not single and not directory:
        pytest.skip(
            f"Optional source fidelity check needs {filename} from {_SOURCE_REVISION};"
            " set MICRODUCK_MJCF_DIR (or MICRODUCK_MJCF_PATH for walking). No source is fetched."
        )
    path = Path(single) if single else Path(directory) / filename
    if not path.is_file():
        pytest.skip(f"Optional pinned MicroDuck source fixture not present: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == _SOURCE_SHA256[filename], f"Source fixture is not {filename} from {_SOURCE_REVISION}: {path}"
    return str(path)
