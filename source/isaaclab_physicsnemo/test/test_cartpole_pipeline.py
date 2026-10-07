# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record, train, and deploy a learned Cartpole through the Isaac Lab environment API."""

import pytest

pytest.importorskip("physicsnemo_robotics")

import torch
from isaaclab_physicsnemo import (
    SmoothRandomActions,
    TrainingSettings,
    TrajectoryDatasetRecorder,
    apply_learned_physics,
    preflight_learned_physics,
    train_dynamics_artifact,
)
from isaaclab_physicsnemo.tasks import CARTPOLE_REMOVE_ROOT_TRANSLATION, CartpoleTeacherEnvCfg

from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv

_NUM_ENVS = 16
_TINY_MODEL = {"temporal_depth": 1, "num_heads": 2, "hidden_size": 32, "head_hidden": 16}


def _make_cfg(device: str) -> CartpoleTeacherEnvCfg:
    cfg = CartpoleTeacherEnvCfg()
    cfg.scene.num_envs = _NUM_ENVS
    cfg.sim.device = device
    cfg.seed = 0
    return cfg


def _step(env: CartpoleEnv, steps: int, on_step=None) -> int:
    """Step with smooth random actions and return the number of finished episodes."""
    actions = SmoothRandomActions(env.num_envs, env.cfg.action_space, device=env.device, seed=0)
    dones, finished = None, 0
    with torch.inference_mode():
        for _ in range(steps):
            observations, _, terminated, truncated, _ = env.step(actions(dones))
            assert torch.isfinite(observations["policy"]).all()
            dones = terminated | truncated
            finished += int(dones.sum())
            if on_step is not None:
                on_step(dones)
    return finished


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_learned_cartpole_round_trip(tmp_path, device):
    """A dataset recorded with MuJoCo Warp yields an artifact that deploys through ``NewtonCfg``.

    Deployment reuses the recording configuration with a different environment count, resets the
    fixed-base cart through the normal environment API, and must pass every preflight check. A physics
    step that differs from the artifact frame is rejected when the environment is built.
    """
    env = CartpoleEnv(cfg=_make_cfg(device))
    try:
        env.reset(seed=0)
        with TrajectoryDatasetRecorder(
            env,
            tmp_path / "dataset",
            scenario_id="test-cartpole",
            remove_root_translation=CARTPOLE_REMOVE_ROOT_TRANSLATION,
        ) as recorder:
            finished = _step(env, 240, recorder.record)
        assert finished > 0, "the recording should contain asynchronous environment resets"
        assert recorder.episodes_written >= finished

        summary = train_dynamics_artifact(
            tmp_path / "dataset",
            env.sim.physics_manager.get_model(),
            tmp_path / "training",
            remove_root_translation=CARTPOLE_REMOVE_ROOT_TRANSLATION,
            settings=TrainingSettings(max_updates=20, batch_size=64, log_every=10, model_kwargs=_TINY_MODEL),
            device=device,
        )
    finally:
        env.close()
    SimulationContext.clear_instance()
    assert summary.parameter_updates == 20

    cfg = _make_cfg(device)
    cfg.scene.num_envs = _NUM_ENVS // 2
    cfg.sim.dt = 0.5
    artifact = apply_learned_physics(cfg, summary.artifact_path)
    assert cfg.sim.dt == pytest.approx(1.0 / 120.0)

    env = CartpoleEnv(cfg=cfg)
    try:
        env.reset(seed=0)
        report = preflight_learned_physics(env.sim.physics_manager, artifact)
        assert report.passed, str(report)
        _step(env, 30)
    finally:
        env.close()
    SimulationContext.clear_instance()

    cfg.sim.dt = 1.0 / 60.0
    with pytest.raises(ValueError, match="artifact frame duration"):
        CartpoleEnv(cfg=cfg)
    SimulationContext.clear_instance()
