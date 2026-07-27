# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for vectorized interactive-scene state conversion."""

from types import SimpleNamespace

import torch

from isaaclab.scene import InteractiveScene


def test_relative_deformable_state_preserves_environment_and_node_axes():
    """Deformable nodes round-trip through per-environment origins."""

    class SceneHarness(InteractiveScene):
        @property
        def device(self) -> str:
            return "cpu"

        @property
        def env_origins(self) -> torch.Tensor:
            return self._test_env_origins

        def write_data_to_sim(self) -> None:
            pass

    class DeformableHarness:
        def __init__(self, positions: torch.Tensor) -> None:
            self.data = SimpleNamespace(
                nodal_pos_w=SimpleNamespace(torch=positions),
                nodal_vel_w=SimpleNamespace(torch=torch.zeros_like(positions)),
            )
            self.written_positions = None

        def write_nodal_pos_to_sim(self, positions, env_ids) -> None:
            self.written_positions = positions.clone()

        def write_nodal_velocity_to_sim(self, velocities, env_ids) -> None:
            pass

        def write_data_to_sim(self) -> None:
            pass

    origins = torch.tensor([[0.0, 0.0, 0.0], [2.5, -1.0, 0.5]])
    local_positions = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 10.0
    absolute_positions = local_positions + origins[:, None, :]
    deformable = DeformableHarness(absolute_positions)
    scene = SceneHarness.__new__(SceneHarness)
    scene._test_env_origins = origins
    scene._ALL_INDICES = torch.arange(2)
    scene._articulations = {}
    scene._deformable_objects = {"cloth": deformable}
    scene._rigid_objects = {}
    scene._surface_grippers = {}

    relative_state = scene.get_state(is_relative=True)

    torch.testing.assert_close(
        relative_state["deformable_object"]["cloth"]["nodal_position"],
        local_positions,
    )
    scene.reset_to(relative_state, is_relative=True)
    torch.testing.assert_close(deformable.written_positions, absolute_positions)
