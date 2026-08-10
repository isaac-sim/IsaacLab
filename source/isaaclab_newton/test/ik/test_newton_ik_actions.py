# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_newton.envs.mdp.actions.newton_ik_actions import (
    _finalize_prototype_model,
    _resolve_prototype_articulation_path,
)


class _Geometry:
    pass


class _Builder:
    def __init__(self) -> None:
        self.shape_source = [_Geometry(), None]

    def finalize(self, *, device: str):
        del device
        return self.shape_source


def test_resolve_prototype_articulation_path_includes_nested_root():
    path = _resolve_prototype_articulation_path("/World/envs/env_0", "/Robot", "/torso")

    assert path == "/World/envs/env_0/Robot/torso"


def test_finalize_prototype_model_isolates_shared_geometry():
    builder = _Builder()
    original_shape_source = builder.shape_source
    original_geometry = builder.shape_source[0]

    finalized_shape_source = _finalize_prototype_model(builder, "cpu")

    assert builder.shape_source is original_shape_source
    assert builder.shape_source[0] is original_geometry
    assert finalized_shape_source is not original_shape_source
    assert finalized_shape_source[0] is not original_geometry
    assert finalized_shape_source[1] is None
