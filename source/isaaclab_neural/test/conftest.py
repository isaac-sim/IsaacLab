# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A fake world model and a world-model simulation context, for tests without weights, a GPU or Kit."""

import contextlib
import json

import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_neural.physics import WorldModelCfg
from isaaclab_neural.world_model import ModelCfg, WorldModel, WorldModelSpec

N, J, T, H, W = 2, 3, 4, 6, 8
"""Environments, joints, frames per chunk and frame size."""

DT = 1.0 / 30


class FakeModel(WorldModel):
    """Frame ``t`` of chunk ``c`` of view ``v`` is filled with ``10 * c + t + 100 * v``."""

    views = ("camera",)
    chunks = 0
    steps = []
    resets = []
    calls = []

    def __init__(self, cfg, num_envs, device):
        self.cfg, self.num_envs, self.device = cfg, num_envs, device
        self.closed = False

    def spec(self):
        return WorldModelSpec(chunk_length=T, fps=30, image_size=(H, W), views=self.views)

    def reset(self, env_ids, images, prompts, seeds):
        assert all(image.device == torch.device(self.device) for image in images.values())
        FakeModel.calls.append("reset")
        FakeModel.resets.append(
            (
                list(env_ids),
                {view: image.cpu().numpy().copy() for view, image in images.items()},
                list(prompts),
                list(seeds),
            )
        )

    def step(self, joint_targets, start_joint_state):
        assert joint_targets.device == start_joint_state.device == torch.device(self.device)
        FakeModel.calls.append("step")
        FakeModel.chunks += 1
        FakeModel.steps.append((joint_targets.cpu().numpy().copy(), start_joint_state.cpu().numpy().copy()))
        out = torch.zeros((self.num_envs, T, H, W, 3), dtype=torch.uint8, device=self.device)
        for t in range(T):
            out[:, t] = 10 * FakeModel.chunks + t
        return {view: out + 100 * v for v, view in enumerate(self.views)}

    def close(self):
        self.closed = True


@configclass
class FakeModelCfg(ModelCfg):
    class_type: type = FakeModel


def write_manifest(directory, states=None, split="train", views=None):
    """Write one reset record per joint state and return the manifest path."""
    states = [np.full(J, 0.1 * (i + 1)) for i in range(2)] if states is None else states
    for i in range(len(states)):
        Image.fromarray(np.full((H, W, 3), 200 + i, np.uint8)).save(directory / f"start_{i}.png")
    rows = []
    for i, state in enumerate(states):
        image = f"start_{i}.png"
        if views is not None:
            image = {}
            for v, view in enumerate(views):
                path = f"{view}_{i}.png"
                Image.fromarray(np.full((H, W, 3), 200 + i + 20 * v, np.uint8)).save(directory / path)
                image[view] = path
        rows.append({"image": image, "state": np.asarray(state).tolist(), "prompt": f"prompt {i}", "split": split})
    path = directory / "resets.json"
    path.write_text(json.dumps(rows))
    return str(path)


@contextlib.contextmanager
def world_model_sim(
    directory, num_envs=N, envs_per_reset_record=N, states=None, split="train", device="cpu", views=None
):
    """A simulation context on the world-model backend. Yields the simulation."""
    FakeModel.chunks, FakeModel.steps, FakeModel.resets = 0, [], []
    FakeModel.calls = []
    FakeModel.views = ("camera",) if views is None else tuple(views)
    physics = WorldModelCfg(
        model=FakeModelCfg(),
        reset_manifest=write_manifest(directory, states, split, views),
        split=split,
        envs_per_reset_record=envs_per_reset_record,
    )
    with sim_utils.build_simulation_context(sim_cfg=SimulationCfg(dt=DT, device=device, physics=physics)) as sim:
        yield sim
