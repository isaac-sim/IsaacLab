# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A fake world model and a world-model simulation context, for tests without weights, a GPU or Kit."""

import contextlib
import json

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_neural.physics import WorldModelCfg
from isaaclab_neural.world_model import ModelCfg, WorldModel, WorldModelSpec

N, J, T, H, W = 2, 3, 4, 6, 8
"""Environments, joints, frames per chunk and frame size."""

DT = 1.0 / 30


class FakeModel(WorldModel):
    """Frame ``t`` of chunk ``c`` is filled with ``10 * c + t``; the start frames with 200 + the environment."""

    chunks = 0
    steps = []
    resets = []

    def spec(self):
        return WorldModelSpec(chunk_length=T, fps=30, image_size=(H, W))

    def reset(self, env_ids, images, prompts, seeds):
        FakeModel.resets.append(prompts)
        return np.stack([np.full((H, W, 3), 200 + int(i), np.uint8) for i in env_ids])

    def step(self, joint_targets, start_joint_state):
        FakeModel.chunks += 1
        FakeModel.steps.append((joint_targets.copy(), start_joint_state.copy()))
        out = np.zeros((self.num_envs, T, H, W, 3), np.uint8)
        for t in range(T):
            out[:, t] = 10 * FakeModel.chunks + t
        return out

    def close(self):
        pass


@configclass
class FakeModelCfg(ModelCfg):
    class_type: type = FakeModel


def write_manifest(directory, states=None, split="train"):
    """Write one reset record per joint state and return the manifest path."""
    (directory / "start.png").write_bytes(b"x")
    states = [np.full(J, 0.1 * (i + 1)) for i in range(2)] if states is None else states
    rows = [
        {"image": "start.png", "state": np.asarray(s).tolist(), "prompt": f"prompt {i}", "split": split}
        for i, s in enumerate(states)
    ]
    path = directory / "resets.json"
    path.write_text(json.dumps(rows))
    return str(path)


@contextlib.contextmanager
def world_model_sim(directory, num_envs=N, envs_per_reset_record=N, states=None, split="train"):
    """A simulation context on the world-model backend. Yields the simulation."""
    FakeModel.chunks, FakeModel.steps, FakeModel.resets = 0, [], []
    physics = WorldModelCfg(
        model=FakeModelCfg(),
        reset_manifest=write_manifest(directory, states, split),
        split=split,
        envs_per_reset_record=envs_per_reset_record,
    )
    with sim_utils.build_simulation_context(sim_cfg=SimulationCfg(dt=DT, device="cpu", physics=physics)) as sim:
        yield sim
