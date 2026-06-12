# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the ported config-drift guard + step-time debug KPIs."""

import build_bench_result as bbr
from _helpers import write_info

# ----- config-drift guard ---------------------------------------------------


def _snapshot(**over):
    base = {"task_id": "Isaac-Cartpole", "num_envs": 4096, "seed": 42, "num_frames": 300, "physics_token": "physx"}
    base.update(over)
    return base


def test_drift_none_when_matching():
    info = {"task": "Isaac-Cartpole", "num_envs": 4096, "seed": 42, "num_frames": 300, "physics": "physx"}
    assert bbr._config_drift(info, _snapshot()) is None


def test_drift_detects_num_envs():
    info = {"task": "Isaac-Cartpole", "num_envs": 256, "physics": "physx"}
    assert "num_envs(ran=256,want=4096)" in bbr._config_drift(info, _snapshot())


def test_drift_detects_task():
    info = {"task": "Isaac-Other", "num_envs": 4096}
    assert "task(ran=Isaac-Other,want=Isaac-Cartpole)" in bbr._config_drift(info, _snapshot())


def test_drift_detects_physics_backend():
    info = {"task": "Isaac-Cartpole", "physics": "physx"}
    snap = _snapshot(physics_token="newton_mjwarp")
    assert "physics(ran=physx,want=newton_mjwarp)" in bbr._config_drift(info, snap)


def test_drift_skips_default_physics():
    # "default" means the launch used a preset bundle, not physics=; don't false-fail.
    info = {"task": "Isaac-Cartpole", "physics": "default"}
    assert bbr._config_drift(info, _snapshot(physics_token="newton_mjwarp")) is None


def test_drift_allows_more_frames():
    info = {"task": "Isaac-Cartpole", "num_frames": 400}
    assert bbr._config_drift(info, _snapshot()) is None


def test_drift_empty_info_is_none():
    assert bbr._config_drift({}, _snapshot()) is None


# ----- benchmark_info extraction --------------------------------------------


def test_extract_benchmark_info(tmp_path):
    info_path = write_info(
        tmp_path,
        [1000.0] * 10,
        benchmark_info={"task": "Isaac-Cartpole", "num_envs": 4096, "physics": "newton_mjwarp"},
    )
    got = bbr._extract_benchmark_info(info_path)
    assert got["task"] == "Isaac-Cartpole"
    assert got["num_envs"] == 4096
    assert got["physics"] == "newton_mjwarp"


# ----- step-time debug KPIs -------------------------------------------------


def test_debug_kpis_p99_and_outliers(tmp_path):
    # 1 warm-up spike (dropped) + steady 10ms with one 30ms outlier.
    steps = [200.0] + [10.0] * 50 + [30.0]
    info_path = write_info(tmp_path, [1.0] * len(steps), step_times=steps)
    kpis = bbr._extract_debug_kpis(info_path, frozenset({0}))
    assert kpis["steady_frames"] == 51
    assert kpis["p99_over_median"] >= 1.0
    assert kpis["outlier_count"] == 1  # the 30ms step is > 2x the 10ms median
    assert "warmup_flag" not in kpis  # warm-up spike was excluded


def test_debug_kpis_warmup_flag(tmp_path):
    # First *kept* frame is a 5x spike -> warm-up window too small.
    steps = [50.0] + [10.0] * 30
    info_path = write_info(tmp_path, [1.0] * len(steps), step_times=steps)
    kpis = bbr._extract_debug_kpis(info_path, frozenset())
    assert "warmup_flag" in kpis


def test_debug_kpis_empty_without_step_times(tmp_path):
    info_path = write_info(tmp_path, [1000.0] * 10)
    assert bbr._extract_debug_kpis(info_path, frozenset()) == {}
