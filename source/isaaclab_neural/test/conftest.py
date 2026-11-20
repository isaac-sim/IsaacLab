# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A fake world model and a world-model simulation context, for tests without weights, a GPU or Kit."""

import contextlib
import json

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_neural.physics import NeuralCfg
from isaaclab_neural.world_model import ModelCfg, WorldModel, WorldModelSpec

N, J, T, H, W = 2, 3, 4, 6, 8
"""Environments, joints, frames per chunk and frame size."""

DT = 1.0 / 30


class FakeModel(WorldModel):
    """RGB frame ``t`` of chunk ``c`` of view ``v`` is filled with ``10 * c + t + 100 * v``, and its depth with a
    tenth of that."""

    views = ("camera",)
    outputs = ("rgb",)
    chunks = 0
    steps = []
    resets = []
    closed = 0

    def __init__(self, cfg, num_envs, device):
        self.num_envs, self.device = num_envs, device

    def spec(self):
        return WorldModelSpec(
            chunk_length=T, fps=30, image_size=(H, W), views=FakeModel.views, outputs=FakeModel.outputs
        )

    def reset(self, env_ids, images, prompts, seeds):
        first = images[FakeModel.views[0]]["rgb"]
        FakeModel.resets.append((env_ids, first[:, 0, 0, 0].tolist(), prompts, seeds))

    def step(self, joint_targets, start_joint_state):
        FakeModel.chunks += 1
        FakeModel.steps.append((joint_targets.cpu().numpy().copy(), start_joint_state.cpu().numpy().copy()))
        out = torch.zeros((self.num_envs, T, H, W, 3), dtype=torch.uint8, device=self.device)
        for t in range(T):
            out[:, t] = 10 * FakeModel.chunks + t
        frames = {}
        for v, view in enumerate(FakeModel.views):
            rgb = out + 100 * v
            frames[view] = {"rgb": rgb, "depth": rgb[..., :1].float() / 10}
            frames[view] = {output: frames[view][output] for output in FakeModel.outputs}
        return frames

    def close(self):
        FakeModel.closed += 1


@configclass
class FakeModelCfg(ModelCfg):
    class_type: type = FakeModel


def write_manifest(directory, states=None, split="train", views=None, outputs=None):
    """Write one reset record per joint state and return the manifest path. Record ``i``'s start image is filled
    with ``200 + i``, plus ``20 * v`` for view ``v`` when ``views`` gives one image per view. With ``outputs``, each
    view also has a start depth of a tenth of that, as ``.npy``."""
    from PIL import Image

    states = [np.full(J, 0.1 * (i + 1)) for i in range(2)] if states is None else states
    rows = []
    for i, s in enumerate(states):
        Image.fromarray(np.full((H, W, 3), 200 + i, dtype=np.uint8)).save(directory / f"start{i}.png")
        image = f"start{i}.png"
        if views is not None:
            for v, view in enumerate(views):
                Image.fromarray(np.full((H, W, 3), 200 + i + 20 * v, dtype=np.uint8)).save(directory / f"{view}{i}.png")
            image = {view: f"{view}{i}.png" for view in views}
        if outputs is not None:
            image = {}
            for v, view in enumerate(views or ("camera",)):
                value = 200 + i + 20 * v
                Image.fromarray(np.full((H, W, 3), value, dtype=np.uint8)).save(directory / f"{view}{i}.png")
                np.save(directory / f"{view}{i}.npy", np.full((H, W, 1), value / 10, dtype=np.float32))
                paths = {"rgb": f"{view}{i}.png", "depth": f"{view}{i}.npy"}
                image[view] = {output: paths[output] for output in outputs}
        rows.append({"image": image, "state": np.asarray(s).tolist(), "prompt": f"prompt {i}", "split": split})
    path = directory / "resets.json"
    path.write_text(json.dumps(rows))
    return str(path)


@contextlib.contextmanager
def world_model_sim(
    directory, num_envs=N, envs_per_reset_record=N, states=None, split="train", device="cpu", views=None, outputs=None
):
    """A simulation context on the world-model backend, with a model of one view or of ``views``, predicting RGB or
    ``outputs``. Yields the simulation."""
    FakeModel.chunks, FakeModel.steps, FakeModel.resets, FakeModel.closed = 0, [], [], 0
    FakeModel.views = ("camera",) if views is None else tuple(views)
    FakeModel.outputs = ("rgb",) if outputs is None else tuple(outputs)
    physics = NeuralCfg(
        model_cfg=FakeModelCfg(),
        reset_manifest=write_manifest(directory, states, split, views, outputs),
        split=split,
        envs_per_reset_record=envs_per_reset_record,
    )
    with sim_utils.build_simulation_context(sim_cfg=SimulationCfg(dt=DT, device=device, physics=physics)) as sim:
        yield sim
