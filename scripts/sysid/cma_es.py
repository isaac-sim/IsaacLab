# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Ported from the agile sysid stack (agile/sysid/cma_es.py), itself adapted from
# pace-sim2real (ETH Zurich RSL / NVIDIA Isaac). The fitted parameters here are
# the implicit-actuator PD gains {stiffness, damping} written solver-side via
# write_joint_stiffness_to_sim_index / write_joint_damping_to_sim_index, instead
# of the original armature/friction/delay set.

from __future__ import annotations

import os
from datetime import datetime

import cmaes
import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter


class CMAESOptimizer:
    """CMA-ES optimizer fitting per-joint implicit-actuator gains.

    Parameter layout (physical space, N = len(joint_order)):
        stiffness_idx = slice(0, N)
        damping_idx   = slice(N, 2N)

    One CMA-ES generation == one full trajectory replay; each of the
    ``population_size`` parallel envs runs one candidate. ``population_size``
    must equal ``num_envs``.
    """

    def __init__(
        self,
        bounds: torch.Tensor,
        population_size: int,
        log_dir: str | os.PathLike,
        joint_order: list[str],
        max_iteration: int,
        data: dict,
        device: str,
        epsilon: float | None = None,
        sigma: float = 0.5,
        save_interval: int = 10,
        save_optimization_process: bool = False,
        initial_mean: torch.Tensor | None = None,
        warmstart_sigma_scale: float = 1.0,
        plateau_patience: int = 0,
        plateau_min_delta: float = 1e-4,
        seed: int = 0,
        run_metadata: dict | None = None,
    ) -> None:
        self.joint_order = joint_order
        self.max_iteration = max_iteration
        self.epsilon = epsilon
        self.save_interval = save_interval
        self.device = device
        self.save_optimization_process = save_optimization_process
        self.plateau_patience = plateau_patience
        self.plateau_min_delta = plateau_min_delta
        self._sigma_history: list[float] = []
        self._timer_start = datetime.now()
        self._initial_mean = initial_mean

        self.seed = seed
        self.run_metadata = dict(run_metadata or {})

        folder_time = datetime.now().strftime("%y_%m_%d_%H-%M-%S")
        log_dir = os.path.join(log_dir, folder_time)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = TensorboardSummaryWriter(log_dir=log_dir)
        torch.save(
            {
                "bounds": bounds,
                "joint_order": joint_order,
                "seed": seed,
                "run_metadata": self.run_metadata,
                "dof_pos": data["dof_pos"],
                "des_dof_pos": data["des_dof_pos"],
                "time": data["time"],
            },
            os.path.join(log_dir, "config.pt"),
        )

        self.bounds = bounds

        bounds_normalized = torch.ones_like(bounds)
        bounds_normalized[:, 0] *= -1

        if initial_mean is not None:
            mean_normalized = (
                2.0 * (initial_mean.to(bounds.device) - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) - 1.0
            ).clamp(-1.0, 1.0)
        else:
            mean_normalized = torch.zeros_like(bounds[:, 0])

        self.optimizer = cmaes.CMA(
            mean=mean_normalized.cpu().numpy(),
            sigma=sigma * warmstart_sigma_scale,
            bounds=bounds_normalized.cpu().numpy(),
            seed=seed,
            population_size=population_size,
        )

        self.scores_counter = 0
        self._buffer_idx = 0
        self.iteration_counter = 0

        # Best-ever EVALUATED candidate. The CMA mean is a distribution
        # statistic that never ran in the sim — report both, and rollout the
        # mean before trusting it (see README acceptance gates).
        self.best_score = float("inf")
        self.best_sim_params: torch.Tensor | None = None
        self.best_iteration = -1

        self.scores = torch.zeros(population_size, device=device)
        self.scores_buffer = torch.zeros((max_iteration, population_size), device=device)
        self.sim_dof_pos_buffer = torch.zeros(
            (population_size, data["dof_pos"].shape[0], len(joint_order)), device=device
        )

        self.params = torch.zeros((population_size, bounds.shape[0]), device=device)
        self.sim_params = torch.zeros_like(self.params)
        if save_optimization_process:
            self.sim_params_buffer = torch.zeros((max_iteration, population_size, bounds.shape[0]), device=device)

        num_joints = len(joint_order)
        self.stiffness_idx = slice(0, num_joints)
        self.damping_idx = slice(num_joints, 2 * num_joints)

        self._reset_population()
        print("CMA-ES optimizer initialized.")
        print("Current iteration:", self.iteration_counter)

    # ------------------------------------------------------------------
    # Core optimizer interface
    # ------------------------------------------------------------------

    def ask(self) -> list:
        return self.optimizer.ask()

    def tell(self, sim_dof_pos: torch.Tensor, real_dof_pos: torch.Tensor, count: bool = True) -> None:
        """Accumulate the replay loss for the current trajectory step.

        Loss per env = mean over COUNTED steps of the SUM over fitted joints of
        squared position error (rad²) — summed across joints, not averaged over
        them. ``count=False`` records the trajectory sample but excludes it from
        the loss (burn-in masking for the shaper warm-up window).
        """
        if count:
            self.scores += torch.sum(torch.square(sim_dof_pos - real_dof_pos), dim=1)
            self.scores_counter += 1
        self.sim_dof_pos_buffer[:, self._buffer_idx, :] = sim_dof_pos
        self._buffer_idx += 1

    def evolve(self) -> None:
        """Advance CMA-ES one generation using accumulated scores."""
        self.scores /= self.scores_counter
        self.scores_buffer[self.iteration_counter, :] = self.scores
        gen_min, gen_min_idx = torch.min(self.scores, dim=0)
        if gen_min.item() < self.best_score:
            self.best_score = gen_min.item()
            self.best_sim_params = self.sim_params[gen_min_idx].detach().clone()
            self.best_iteration = self.iteration_counter
            # best_candidate.pt and best_trajectory.pt always describe the SAME
            # evaluated candidate (params, score, generation, rollout).
            best_traj = self.sim_dof_pos_buffer[gen_min_idx].detach().clone().cpu()
            torch.save(
                {
                    "sim_params": self.best_sim_params.cpu(),
                    "score": self.best_score,
                    "iteration": self.best_iteration,
                    "env_index": int(gen_min_idx.item()),
                    "joint_order": self.joint_order,
                    "seed": self.seed,
                    "run_metadata": self.run_metadata,
                },
                os.path.join(self.writer.log_dir, "best_candidate.pt"),
            )
            torch.save(best_traj, os.path.join(self.writer.log_dir, "best_trajectory.pt"))
        self._sigma_history.append(self.params.std(dim=0).max().item())
        if self.save_optimization_process:
            self.sim_params_buffer[self.iteration_counter, :, :] = self.sim_params
        solutions = [
            (self.params[i].cpu().numpy(), self.scores[i].item()) for i in range(self.optimizer.population_size)
        ]
        self.optimizer.tell(solutions)
        if self.save_interval > 0 and self.iteration_counter % self.save_interval == 0:
            self.save_checkpoint(
                self._params_to_sim_params(torch.tensor(self.optimizer._mean, device=self.device)),
                self.iteration_counter,
            )
        self._print_iteration()
        self._reset_population()
        self.scores = torch.zeros_like(self.scores)
        self.scores_counter = 0
        self._buffer_idx = 0
        self.iteration_counter += 1
        print("CMA-ES iteration:", self.iteration_counter)

    def finished(self) -> bool:
        finished = self.max_iteration <= self.iteration_counter
        stop_reason = "max_iterations" if finished else None

        if self.iteration_counter > 0:
            row = self.scores_buffer[self.iteration_counter - 1, :]
            diff_score = (row.max() - row.min()) / row.min()
            if self.epsilon is not None and diff_score < self.epsilon:
                finished = True
                stop_reason = "epsilon"

        if self.plateau_patience > 0 and len(self._sigma_history) >= self.plateau_patience:
            window = self._sigma_history[-self.plateau_patience :]
            sigma_max = max(window)
            self.writer.add_scalar("0_Episode/plateau_sigma", sigma_max, self.iteration_counter)
            if sigma_max < self.plateau_min_delta:
                finished = True
                stop_reason = stop_reason or "plateau"

        if finished:
            print(f"CMA-ES optimization finished ({stop_reason}).")
            self.save_checkpoint(
                self._params_to_sim_params(torch.tensor(self.optimizer._mean, device=self.device)),
                self.iteration_counter - 1,
                finished=True,
            )
        return finished

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_population(self) -> None:
        for i in range(self.optimizer.population_size):
            self.params[i, :] = torch.tensor(self.optimizer.ask(), device=self.device)
        self.sim_params = self._params_to_sim_params(self.params)

    def _params_to_sim_params(self, params: torch.Tensor) -> torch.Tensor:
        sim_params = (params + 1.0) / 2.0
        sim_params = self.bounds[:, 0] + sim_params * (self.bounds[:, 1] - self.bounds[:, 0])
        return sim_params

    def get_best_sim_params(self) -> torch.Tensor:
        """Physical-space CMA distribution mean — NOT an evaluated candidate."""
        best_params = torch.tensor(self.optimizer._mean, device=self.device)
        return self._params_to_sim_params(best_params)

    def get_best_ever(self) -> tuple[torch.Tensor | None, float, int]:
        """Best evaluated candidate across all generations: (params, score, iteration)."""
        return self.best_sim_params, self.best_score, self.best_iteration

    def _print_iteration(self) -> None:
        min_score, min_index = torch.min(self.scores, dim=0)
        max_score = torch.max(self.scores)
        print("Max score:", max_score.item())
        print("Min score:", min_score.item(), "at index:", min_index.item())
        print("Stiffness:", self.sim_params[min_index, self.stiffness_idx].tolist())
        print("Damping:", self.sim_params[min_index, self.damping_idx].tolist())
        print(f"Elapsed: {(datetime.now() - self._timer_start).total_seconds():.1f}s")
        self._timer_start = datetime.now()
        self._log()

    def _log(self) -> None:
        min_score, min_score_index = torch.min(self.scores, dim=0)
        max_score, _ = torch.max(self.scores, dim=0)
        for i, jname in enumerate(self.joint_order):
            self.writer.add_histogram(
                f"1_Stiffness/dist_{jname}", self.sim_params[:, self.stiffness_idx][:, i], self.iteration_counter
            )
            self.writer.add_histogram(
                f"2_Damping/dist_{jname}", self.sim_params[:, self.damping_idx][:, i], self.iteration_counter
            )
            self.writer.add_scalar(
                f"1_Stiffness/best_{jname}",
                self.sim_params[min_score_index, self.stiffness_idx][i].item(),
                self.iteration_counter,
            )
            self.writer.add_scalar(
                f"2_Damping/best_{jname}",
                self.sim_params[min_score_index, self.damping_idx][i].item(),
                self.iteration_counter,
            )
        self.writer.add_scalar("0_Episode/score", min_score.item(), self.iteration_counter)
        self.writer.add_scalar("0_Episode/max_score", max_score.item(), self.iteration_counter)
        self.writer.add_scalar("0_Episode/diff_score", (max_score - min_score) / min_score, self.iteration_counter)

    def save_checkpoint(self, mean: torch.Tensor, iteration: int, finished: bool = False) -> None:
        # best_trajectory.pt is written by the best-ever branch in evolve() so it
        # always pairs with best_candidate.pt; checkpoints only save the mean.
        # NOTE: the mean is a distribution statistic — fit.py rerolls it after
        # the fit so the reported mean score is evaluated.
        torch.save(
            {
                "sim_params": mean.cpu(),
                "joint_order": self.joint_order,
                "seed": self.seed,
                "iteration": iteration,
                "run_metadata": self.run_metadata,
            },
            os.path.join(self.writer.log_dir, f"mean_{iteration:03}.pt"),
        )
        if finished and self.save_optimization_process:
            torch.save(
                {"params_buffer": self.sim_params_buffer, "scores_buffer": self.scores_buffer},
                os.path.join(self.writer.log_dir, "progress.pt"),
            )

    def close(self) -> None:
        self.writer.close()
