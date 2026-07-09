# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the recovery-report acceptance gate."""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import recovery_report  # noqa: E402
from data_contract import CANONICAL_JOINT_ORDER  # noqa: E402

KP = torch.full((7,), 1000.0)
KD = torch.full((7,), 40.0)


@pytest.fixture()
def dataset(tmp_path):
    T = 100
    data = {
        "time": torch.arange(T, dtype=torch.float32) / 200.0,
        "des_dof_pos": torch.zeros(T, 7),
        "dof_pos": torch.zeros(T, 7),
        "dof_vel": torch.zeros(T, 7),
        "dof_tau_est": torch.zeros(T, 7),
        "joint_names": list(CANONICAL_JOINT_ORDER),
        "active_joint_names": list(CANONICAL_JOINT_ORDER),
        "sample_rate": 200.0,
        "kp_used": KP,
        "kd_used": KD,
        "state_fresh": torch.ones(T, dtype=torch.uint8),
        "state_stamps": torch.arange(T, dtype=torch.float64) / 200.0,
        "mode": "synthetic",
    }
    path = tmp_path / "chirp_data.pt"
    torch.save(data, path)
    return str(path)


def make_run(tmp_path, params: torch.Tensor, joint_order=None, seed=3, meta=True, with_mean=True):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    blob = {
        "sim_params": params,
        "score": 1e-6,
        "iteration": 0,
        "env_index": 0,
        "joint_order": list(joint_order or CANONICAL_JOINT_ORDER),
        "seed": seed,
    }
    if meta:
        blob["run_metadata"] = {"usd_digest": "abc", "use_cuda_graph": True}
    torch.save(blob, run / "best_candidate.pt")
    if with_mean:
        torch.save(
            {
                "sim_params": params,
                "joint_order": blob["joint_order"],
                "seed": seed,
                "iteration": 0,
                "run_metadata": blob.get("run_metadata", {"usd_digest": "abc", "use_cuda_graph": True}),
            },
            run / "mean_000.pt",
        )
    return str(run)


def run_main(data, run, max_mape=5.0):
    sys.argv = ["recovery_report.py", "--data", data, "--run", run, "--max_mape", str(max_mape)]
    return recovery_report.main()


def test_exact_recovery_passes(tmp_path, dataset):
    run = make_run(tmp_path, torch.cat([KP, KD]))
    assert run_main(dataset, run) == 0
    with open(os.path.join(run, "recovery_result.json")) as f:
        result = json.load(f)
    assert result["pass"] and result["seed"] == 3


def test_bad_recovery_fails(tmp_path, dataset):
    run = make_run(tmp_path, torch.cat([KP * 2.4, KD * 2.7]))  # 140%/170% error
    assert run_main(dataset, run) == 1


def test_reversed_joint_order_rejected(tmp_path, dataset):
    run = make_run(tmp_path, torch.cat([KP, KD]), joint_order=list(reversed(CANONICAL_JOINT_ORDER)))
    with pytest.raises(ValueError, match="joint_order"):
        run_main(dataset, run)


def test_missing_seed_rejected(tmp_path, dataset):
    run = make_run(tmp_path, torch.cat([KP, KD]), seed=None)
    with pytest.raises(ValueError, match="seed"):
        run_main(dataset, run)


def test_missing_metadata_rejected(tmp_path, dataset):
    run = make_run(tmp_path, torch.cat([KP, KD]), meta=False)
    with pytest.raises(ValueError, match="run_metadata"):
        run_main(dataset, run)


def test_missing_mean_fails(tmp_path, dataset):
    run = make_run(tmp_path, torch.cat([KP, KD]), with_mean=False)
    assert run_main(dataset, run) == 1


def test_foreign_mean_seed_rejected(tmp_path, dataset):
    run = make_run(tmp_path, torch.cat([KP, KD]))
    mean = torch.load(os.path.join(run, "mean_000.pt"), weights_only=False)
    mean["seed"] = 999
    torch.save(mean, os.path.join(run, "mean_000.pt"))
    with pytest.raises(ValueError, match="seed"):
        run_main(dataset, run)


def test_foreign_mean_digest_rejected(tmp_path, dataset):
    run = make_run(tmp_path, torch.cat([KP, KD]))
    mean = torch.load(os.path.join(run, "mean_000.pt"), weights_only=False)
    mean["run_metadata"] = {"usd_digest": "OTHER", "use_cuda_graph": True}
    torch.save(mean, os.path.join(run, "mean_000.pt"))
    with pytest.raises(ValueError, match="usd_digest"):
        run_main(dataset, run)


def test_mean_without_metadata_rejected(tmp_path, dataset):
    run = make_run(tmp_path, torch.cat([KP, KD]))
    mean = torch.load(os.path.join(run, "mean_000.pt"), weights_only=False)
    del mean["run_metadata"]
    torch.save(mean, os.path.join(run, "mean_000.pt"))
    with pytest.raises(ValueError, match="run_metadata"):
        run_main(dataset, run)
