# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the play (inference) benchmark bundle, builder, and stepping helper."""

import json
import os

import pytest
import torch

from isaaclab.test.benchmark import builders
from isaaclab.test.benchmark.schema import (
    SCHEMA_VERSION,
    GpuDeviceInfo,
    Hardware,
    MeanStd,
    PlayBundle,
    Resources,
    RunConfig,
    RunIdentity,
    Runtime,
    StartupTime,
    Versions,
)
from isaaclab.test.benchmark.serialize import write_bundle_file


def _versions() -> Versions:
    return Versions(
        isaaclab="4.6.8",
        isaacsim="5.0.0",
        kit="107.1.0",
        newton="0.1.2",
        warp="1.7.3",
        mjwarp="0.0.4",
        torch="2.5.1",
        rsl_rl="2.3.0",
        rl_games=None,
        skrl=None,
        sb3=None,
        git_commit="3d42b11d513",
        git_branch="develop",
        git_dirty=False,
    )


def _hardware() -> Hardware:
    return Hardware(
        hostname="benchmark-host",
        gpu_devices=[GpuDeviceInfo(name="NVIDIA H100 80GB", mem_gb=80.0, compute_cap="9.0")],
        cpu_name="AMD EPYC 7763",
        cpu_count=64,
        ram_gb=512.0,
    )


def _run_identity() -> RunIdentity:
    return RunIdentity(
        run_id="rsl-rl_newton_mjwarp_Isaac-Ant-Direct-v0_20260422-131500_seed42",
        framework="rsl_rl",
        config=RunConfig(physics_backend="newton_mjwarp", rendering_backend="none"),
        task="Isaac-Ant-Direct-v0",
        seed=42,
        start_time_utc="2026-04-22T13:15:00Z",
        end_time_utc="2026-04-22T13:15:30Z",
        duration_s=30.0,
        status="completed",
        num_envs=64,
    )


def _runtime() -> Runtime:
    return Runtime(
        startup_time_s=StartupTime(app_launch=18.4, env_creation=22.9, first_step=4.1),
        iterations_completed=300,
        total_wall_time_s=12.0,
        steps_per_iteration=1,
        iteration_time_s=MeanStd(mean=0.04, std=0.004),
        collection_fps=MeanStd(mean=1_142_000.0, std=9_500.0),
        total_fps=MeanStd(mean=1_142_000.0, std=9_500.0),
        iterations_per_s=MeanStd(mean=25.0, std=1.0),
    )


def _resources() -> Resources:
    return Resources(
        gpu_util_pct=MeanStd(mean=87.2, std=6.1),
        gpu_mem_gb=MeanStd(mean=18.4, std=0.3, peak=19.2),
        cpu_util_pct=MeanStd(mean=31.5, std=4.8),
        ram_gb=MeanStd(mean=22.1, std=0.4, peak=24.8),
    )


def _minimal_play_bundle() -> PlayBundle:
    return PlayBundle(
        run=_run_identity(),
        versions=_versions(),
        hardware=_hardware(),
        runtime=_runtime(),
        resources=_resources(),
        success_rate=0.83,
        reward=MeanStd(mean=1823.4, std=44.0, peak=1980.0),
        ep_length=MeanStd(mean=987.0, std=12.0, peak=1000.0),
        checkpoint_path="logs/rsl_rl/ant/2026-04-22_13-15-00/model_499.pt",
        video_path=None,
    )


def test_play_bundle_round_trip(tmp_path):
    """PlayBundle round-trips through JSON with framework, success_rate, reward, and checkpoint."""
    bundle = _minimal_play_bundle()
    path = os.path.join(tmp_path, "play.json")
    write_bundle_file(bundle, path)

    with open(path) as f:
        data = json.load(f)

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["run"]["framework"] == "rsl_rl"
    assert data["success_rate"] == pytest.approx(0.83)
    assert data["reward"]["peak"] == pytest.approx(1980.0)
    assert data["ep_length"]["mean"] == pytest.approx(987.0)
    assert data["checkpoint_path"].endswith("model_499.pt")
    assert data["video_path"] is None
    assert "learning" not in data


def test_build_play_bundle_passes_fields_through():
    """build_play_bundle forwards every field unchanged onto the PlayBundle."""
    reward = MeanStd(mean=12.0, std=1.0, peak=15.0)
    ep_length = MeanStd(mean=100.0, std=5.0, peak=120.0)
    b = builders.build_play_bundle(
        run=_run_identity(),
        versions=_versions(),
        hardware=_hardware(),
        runtime=_runtime(),
        resources=_resources(),
        success_rate=0.7,
        reward=reward,
        ep_length=ep_length,
        checkpoint_path="/tmp/m.pt",
    )
    assert isinstance(b, PlayBundle)
    assert b.run.framework == "rsl_rl"
    assert b.success_rate == pytest.approx(0.7)
    assert b.reward is reward
    assert b.ep_length is ep_length
    assert b.checkpoint_path == "/tmp/m.pt"
    assert b.video_path is None


class _PlayEnv:
    """Minimal fake env: env 0 finishes a 2-step episode on the 2nd step, env 1 never finishes."""

    num_envs = 2
    device = "cpu"

    def __init__(self):
        self.unwrapped = self
        self._calls = 0

    def reset(self):
        return torch.zeros(2, 3), {}

    def step(self, actions):
        self._calls += 1
        dones = torch.tensor([True, False]) if self._calls == 2 else torch.tensor([False, False])
        extras = {"log": {"Episode_Reward/success": torch.tensor(1.0)}}
        return torch.zeros(2, 3), torch.ones(2), dones, extras


def test_run_play_loop_aggregates_episodes():
    """run_play_loop times each frame and aggregates returns/lengths over completed episodes."""
    from isaaclab.test.benchmark.stepping import run_play_loop

    env = _PlayEnv()
    step_times, reward, ep_length, success_rate = run_play_loop(env, policy=lambda obs: torch.zeros(2, 1), num_frames=3)
    assert len(step_times) == 3
    assert all(t >= 0.0 for t in step_times)
    # env 0 completed a 2-step episode with reward 1 per step -> return 2.0, length 2.
    assert reward.mean == pytest.approx(2.0)
    assert ep_length.mean == pytest.approx(2.0)
    assert success_rate == pytest.approx(1.0)
