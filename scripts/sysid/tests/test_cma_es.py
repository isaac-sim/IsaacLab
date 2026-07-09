# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU regression tests for the CMA-ES optimizer artifact contract.

Guards the round-2/3 fixes: best_candidate.pt and best_trajectory.pt always
describe the same evaluated candidate, burn-in masking excludes samples from
the loss but not from the trajectory buffer, and provenance (seed, joint_order,
run_metadata) is persisted in every artifact.
"""

import os
import sys

import pytest
import torch

pytest.importorskip("cmaes")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cma_es import CMAESOptimizer  # noqa: E402

JOINTS = ["fr3_joint1", "fr3_joint2"]
T = 6
POP = 4


def make_opt(tmp_path) -> CMAESOptimizer:
    n = len(JOINTS)
    bounds = torch.tensor([[0.0, 10.0]] * (2 * n))
    data = {
        "time": torch.arange(T, dtype=torch.float32) / 200.0,
        "dof_pos": torch.zeros(T, n),
        "des_dof_pos": torch.zeros(T, n),
    }
    return CMAESOptimizer(
        bounds=bounds,
        population_size=POP,
        log_dir=str(tmp_path),
        joint_order=list(JOINTS),
        max_iteration=5,
        data=data,
        device="cpu",
        seed=3,
        run_metadata={"marker": "regression"},
    )


def run_generation(opt: CMAESOptimizer, per_env_error: list[float]) -> None:
    """One full fake replay: constant per-env error on both joints."""
    err = torch.tensor(per_env_error).unsqueeze(1).expand(POP, len(JOINTS))
    for _ in range(T):
        opt.tell(err.clone(), torch.zeros(POP, len(JOINTS)))
    opt.evolve()


class TestArtifactPairing:
    def test_best_candidate_and_trajectory_pair(self, tmp_path):
        opt = make_opt(tmp_path)
        gen0_params = opt.sim_params.clone()
        run_generation(opt, [3.0, 2.0, 1.0, 4.0])  # env 2 best
        log_dir = opt.writer.log_dir

        best = torch.load(os.path.join(log_dir, "best_candidate.pt"), weights_only=False)
        assert best["iteration"] == 0 and best["env_index"] == 2
        assert best["joint_order"] == JOINTS and best["seed"] == 3
        assert best["run_metadata"]["marker"] == "regression"
        assert torch.equal(best["sim_params"], gen0_params[2].cpu())
        # score = mean over steps of sum over joints of err² = 2 * 1²
        assert best["score"] == pytest.approx(2.0)

        traj = torch.load(os.path.join(log_dir, "best_trajectory.pt"), weights_only=False)
        assert traj.shape == (T, len(JOINTS))
        assert torch.allclose(traj, torch.ones_like(traj))  # env 2's constant error rollout

    def test_worse_generation_preserves_pair(self, tmp_path):
        opt = make_opt(tmp_path)
        run_generation(opt, [3.0, 2.0, 1.0, 4.0])
        log_dir = opt.writer.log_dir
        best_before = torch.load(os.path.join(log_dir, "best_candidate.pt"), weights_only=False)
        traj_before = torch.load(os.path.join(log_dir, "best_trajectory.pt"), weights_only=False)

        run_generation(opt, [10.0, 10.0, 10.0, 10.0])  # all worse
        best_after = torch.load(os.path.join(log_dir, "best_candidate.pt"), weights_only=False)
        traj_after = torch.load(os.path.join(log_dir, "best_trajectory.pt"), weights_only=False)
        assert best_after["iteration"] == best_before["iteration"] == 0
        assert torch.equal(best_after["sim_params"], best_before["sim_params"])
        assert torch.equal(traj_after, traj_before)  # gen-1 trajectory must NOT leak in

    def test_burn_in_masking(self, tmp_path):
        opt = make_opt(tmp_path)
        one = torch.ones(POP, len(JOINTS))
        zero = torch.zeros(POP, len(JOINTS))
        opt.tell(one, zero, count=False)  # masked: buffered, not scored
        opt.tell(one, zero, count=True)
        assert opt.scores_counter == 1 and opt._buffer_idx == 2
        assert torch.allclose(opt.sim_dof_pos_buffer[:, 0, :], one)
        opt.evolve()
        # scores averaged over COUNTED steps only: sum over 2 joints of 1² = 2
        assert torch.allclose(opt.scores_buffer[0], torch.full((POP,), 2.0))

    def test_config_and_mean_provenance(self, tmp_path):
        opt = make_opt(tmp_path)
        log_dir = opt.writer.log_dir
        cfg = torch.load(os.path.join(log_dir, "config.pt"), weights_only=False)
        assert cfg["seed"] == 3 and cfg["run_metadata"]["marker"] == "regression"

        opt.save_checkpoint(torch.arange(4, dtype=torch.float32), 0)
        mean = torch.load(os.path.join(log_dir, "mean_000.pt"), weights_only=False)
        assert mean["joint_order"] == JOINTS and mean["seed"] == 3
        assert torch.equal(mean["sim_params"], torch.arange(4, dtype=torch.float32))


class TestFinishedPath:
    def test_finished_writes_final_checkpoint_and_preserves_pair(self, tmp_path):
        n = len(JOINTS)
        bounds = torch.tensor([[0.0, 10.0]] * (2 * n))
        data = {
            "time": torch.arange(T, dtype=torch.float32) / 200.0,
            "dof_pos": torch.zeros(T, n),
            "des_dof_pos": torch.zeros(T, n),
        }
        opt = CMAESOptimizer(
            bounds=bounds,
            population_size=POP,
            log_dir=str(tmp_path),
            joint_order=list(JOINTS),
            max_iteration=2,
            data=data,
            device="cpu",
            seed=3,
            save_interval=0,  # only finished() writes checkpoints
        )
        # Causal ordering: gen0 holds the best-ever, gen1 is strictly worse,
        # THEN finished() runs its final-checkpoint path — the original
        # final-generation overwrite bug would clobber the gen0 pair here.
        run_generation(opt, [3.0, 2.0, 1.0, 4.0])  # gen0: env2 best (traj of ones)
        assert not opt.finished()
        run_generation(opt, [10.0, 10.0, 10.0, 10.0])  # gen1: all worse
        assert opt.finished()
        log_dir = opt.writer.log_dir
        mean = torch.load(os.path.join(log_dir, "mean_001.pt"), weights_only=False)
        assert mean["joint_order"] == JOINTS and mean["seed"] == 3
        best = torch.load(os.path.join(log_dir, "best_candidate.pt"), weights_only=False)
        traj = torch.load(os.path.join(log_dir, "best_trajectory.pt"), weights_only=False)
        assert best["iteration"] == 0 and best["env_index"] == 2
        assert torch.allclose(traj, torch.ones_like(traj))  # gen0 pair survives finished()
