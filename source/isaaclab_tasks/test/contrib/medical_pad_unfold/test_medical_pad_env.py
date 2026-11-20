# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The medical-pad task end to end on the world-model backend, without Kit or Cosmos weights.

The real task configuration runs with three stand-ins: a world model that returns frames of known brightness, a
robot USD with the 58 recorded joints (URDF import needs the importers) and a classifier that scores a frame by its
brightness.
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_neural.mdp import at_chunk_boundary
from isaaclab_neural.world_model import ModelCfg, WorldModel, WorldModelSpec

from isaaclab_tasks.contrib.medical_pad_unfold.h2_robot import H2_JOINT_NAMES
from isaaclab_tasks.contrib.medical_pad_unfold.medical_pad_unfold_env_cfg import (
    CHUNK,
    CLASSIFIER,
    FPS,
    MedicalPadUnfoldEnvCfg,
)
from isaaclab_tasks.utils.hydra import resolve_presets

H, W, N = 32, 48, 2
JOINTS = len(H2_JOINT_NAMES)


def author_robot(path):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Cube.Define(stage, "/Robot/base").GetPrim())
    for name in H2_JOINT_NAMES:
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Robot/{name}")
        joint.CreateAxisAttr("Z")
    stage.Save()


class BrightnessModel(WorldModel):
    """Chunk ``c`` is brightness ``60 * c`` plus a ramp of 10 across the chunk."""

    chunks = 0

    def __init__(self, cfg, num_envs, device):
        self.num_envs, self.device = num_envs, device

    def spec(self):
        return WorldModelSpec(chunk_length=CHUNK, fps=FPS, image_size=(H, W))

    def reset(self, env_ids, images, prompts, seeds):
        pass

    def step(self, joint_targets, start_joint_state):
        BrightnessModel.chunks += 1
        ramp = torch.linspace(0, 1, CHUNK, dtype=torch.float64, device=self.device)[None, :, None, None, None]
        frames = (60 * BrightnessModel.chunks + 10 * ramp).to(torch.uint8)
        return frames.expand(self.num_envs, CHUNK, H, W, 3).contiguous()

    def close(self):
        pass


@configclass
class BrightnessModelCfg(ModelCfg):
    class_type: type = BrightnessModel


class BrightnessClassifier:
    """The class of a frame is its brightness // 50 and its score is brightness / 100."""

    reward_values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    def __init__(self, checkpoint, config_file, device):
        pass

    def score_frames(self, frames):
        brightness = frames.float().mean(dim=(2, 3, 4))
        labels = (brightness // 50).long().clamp(0, 4)
        return brightness / 100, torch.nn.functional.one_hot(labels, 5).float()


def frame_brightness(chunk, t):
    """Brightness of frame ``t`` of the ``chunk``-th chunk the fake model generates (1-based)."""
    return float(np.uint8(60 * chunk + 10 * np.linspace(0, 1, CHUNK)[t]))


@pytest.fixture
def make_env(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "isaaclab_tasks.contrib.medical_pad_unfold.classifier.MedicalPadClassifier", BrightnessClassifier
    )
    robot = tmp_path / "h2.usda"
    author_robot(robot)
    # The start image has brightness 20.
    Image.fromarray(np.full((H, W, 3), 20, np.uint8)).save(tmp_path / "start.png")
    manifest = tmp_path / "resets.json"
    manifest.write_text(
        json.dumps([{"image": "start.png", "state": [0.0] * JOINTS, "prompt": "unfold", "split": "train"}])
    )
    envs = []

    def make(success_class=4, failure_class=0, streak_frames=3, extra_termination=None):
        BrightnessModel.chunks = 0
        cfg = resolve_presets(MedicalPadUnfoldEnvCfg())
        cfg.scene.num_envs = N
        cfg.scene.robot.spawn = sim_utils.UsdFileCfg(usd_path=str(robot))
        cfg.scene.camera.height, cfg.scene.camera.width = H, W
        cfg.sim.device = "cpu"
        cfg.sim.physics.model = BrightnessModelCfg()
        cfg.sim.physics.reset_manifest = str(manifest)
        classifier = {
            **CLASSIFIER,
            "success_class": success_class,
            "failure_class": failure_class,
            "streak_frames": streak_frames,
        }
        for term in (cfg.rewards.progress, cfg.terminations.success, cfg.terminations.failure):
            term.params["classifier"] = classifier
        if extra_termination is not None:
            cfg.terminations.extra = extra_termination
        envs.append(ManagerBasedRLEnv(cfg))
        return envs[-1]

    yield make
    for env in envs:
        env.close()


def run(env, steps):
    """Reward and terminated flag of environment 0 for each step."""
    env.reset()
    action = torch.zeros(N, JOINTS)
    out = []
    for _ in range(steps):
        _, reward, terminated, _, _ = env.step(action)
        out.append((reward[0].item(), bool(terminated[0])))
    return out


def test_the_task_runs_with_stock_managers(make_env):
    env = make_env(streak_frames=10**6)
    obs, _ = env.reset()
    assert obs["image"].shape == (N, 3, H, W) and obs["policy"].shape == (N, JOINTS)
    action = torch.full((N, JOINTS), 0.25)
    rewards = [env.step(action)[1].clone() for _ in range(CHUNK)]
    # The model ran once, on the last step of the chunk, and the reward is paid then.
    assert BrightnessModel.chunks == 1
    assert all(torch.count_nonzero(reward) == 0 for reward in rewards[:-1]) and rewards[-1].all()
    assert torch.allclose(env.step(action)[0]["policy"], action)


def test_frames_after_the_success_frame_are_not_rewarded(make_env):
    # Chunk 4 is the first of class 4 (brightness >= 200); three such frames end the episode at its frame 2.
    out = run(make_env(), 4 * CHUNK)
    rewards, done = [r for r, _ in out], [d for _, d in out]
    assert [i + 1 for i, d in enumerate(done) if d] == [4 * CHUNK]
    assert [i + 1 for i, r in enumerate(rewards) if r] == [CHUNK, 2 * CHUNK, 3 * CHUNK, 4 * CHUNK]
    terminal = frame_brightness(4, 2) / 100
    assert rewards[-1] == pytest.approx(terminal - frame_brightness(3, CHUNK - 1) / 100, abs=1e-5)
    assert sum(rewards) == pytest.approx(terminal, abs=1e-4)
    assert frame_brightness(4, CHUNK - 1) / 100 > terminal  # counting the whole chunk would have paid more


def test_an_episode_that_fails_ends_when_its_chunk_completes(make_env):
    env = make_env(success_class=0, failure_class=4)
    out = run(env, 4 * CHUNK)
    assert [i + 1 for i, (_, d) in enumerate(out) if d] == [4 * CHUNK]
    assert env.termination_manager.get_term("failure")[0] and not env.termination_manager.get_term("success")[0]


def test_each_environment_resets_at_its_own_next_boundary(make_env):
    def first_environment(env):
        done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        done[0] = True
        return done

    env = make_env(streak_frames=10**6, extra_termination=DoneTerm(func=at_chunk_boundary(first_environment)))
    env.reset()
    action = torch.zeros(N, JOINTS)
    for step in range(1, CHUNK + 1):
        terminated = env.step(action)[2]
        # The condition holds from the first step, but environment 0 only ends when the chunk completes.
        assert bool(terminated[0]) == (step == CHUNK) and not terminated[1]
    assert env.episode_length_buf.tolist() == [0, CHUNK]  # environment 1 goes on


def test_the_judgement_starts_again_with_each_episode(make_env):
    env = make_env()
    run(env, 4 * CHUNK)  # the episode succeeded in chunk 4 and the environment was reset
    tracker = env._medical_pad_tracker
    assert tracker.success_streak[0] > 0 and tracker.previous[0] > 0
    env.step(torch.zeros(N, JOINTS))
    assert tracker.previous.tolist() == [0.0, 0.0]
    assert tracker.success_streak.tolist() == [0, 0] and tracker.failure_streak.tolist() == [0, 0]
