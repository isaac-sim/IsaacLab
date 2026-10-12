# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for running an environment step from CUDA graphs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.utils.step_capture import EnvStepCapture
from isaaclab.utils.step_graph import StepOp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")


class _TorchGraph:
    """Records a function into a Torch CUDA graph without executing it."""

    def __init__(self, fn):
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            fn()

    def launch(self):
        self.graph.replay()


class _Env:
    """Environment double whose step reads ``state`` and then replaces it with a new tensor."""

    def __init__(self):
        self.device = "cuda:0"
        self.common_step_counter = 0
        self._sim_step_counter = 0
        self._physics_handles_decimation = False
        self.physics_dt = 0.01
        self.cfg = SimpleNamespace(decimation=1)
        self.video_recorders = []
        physics_ops: list[StepOp] = []
        physics = SimpleNamespace(step_ops=lambda: physics_ops, capture_graph=_TorchGraph)
        self.sim = SimpleNamespace(is_rendering=False, physics_manager=physics, finish_physics_step=lambda: None)
        self.has_rtx_sensors = False
        scene_cfg = SimpleNamespace(lazy_sensor_update=True)
        self.scene = SimpleNamespace(cfg=scene_cfg, sensors={}, notify_update_replayed=lambda dt: None)
        self.state = torch.zeros(4, device=self.device)
        self.observed = torch.zeros(4, device=self.device)

    def observe(self):
        # reads the state the previous step bound
        self.observed.copy_(self.state * 2.0)

    def advance(self, action: torch.Tensor):
        # replaces the buffer instead of writing it in place, as many terms do
        self.state = self.state + action


def _rollout(captured: bool, num_steps: int = 6) -> list[torch.Tensor]:
    env = _Env()
    capture = EnvStepCapture(
        env,
        lambda physics_ops: [
            StepOp(env.observe, True, "observe"),
            StepOp(lambda: env.advance(capture.action), True, "advance"),
        ],
        lambda: None,
        enabled=captured,
    )
    observed = []
    for step in range(num_steps):
        action = torch.full((4,), float(step + 1), device=env.device)
        graph = capture.prepare(action)
        if graph is None:
            env.observe()
            env.advance(action)
        else:
            capture.launch(graph, action)
        observed.append(env.observed.clone())
    assert (capture.graph is not None and capture.graph.captured) == captured
    return observed


def test_rebound_buffers_reach_earlier_readers_after_each_replay():
    """A buffer replaced every step reaches the operations that read it before the replacement, as when eager."""
    for eager, captured in zip(_rollout(captured=False), _rollout(captured=True)):
        torch.testing.assert_close(captured, eager)
