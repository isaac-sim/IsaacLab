# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sparse pickup reference environment and its physical reset-state generator."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.reset import ResetStateBuffer
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import clone, instantiate
from isaaclab.utils.math import compute_pose_error

from isaaclab_tasks.utils import ResetStateSampler
from isaaclab_tasks.utils.sampling import UniformSamplingStrategyCfg

from . import mdp as pickup_mdp
from .cube_pickup_env_cfg import CubePickupEnvCfg


class CubePickupEnv(ManagerBasedRLEnv):
    """Train sparse pickup from snapshots of genuine simulated approach and grasp rollouts."""

    cfg: CubePickupEnvCfg

    def __init__(self, cfg: CubePickupEnvCfg, **kwargs):
        self.reset_buffer: ResetStateBuffer | None = None
        self.reset_sampler: ResetStateSampler | None = None
        super().__init__(cfg, **kwargs)
        self._hand_cfg = SceneEntityCfg("robot", body_names="panda_hand")
        self._hand_cfg.resolve(self.scene)
        self._bank_generator = torch.Generator(device=self.device)
        self._bank_generator.manual_seed(cfg.reset_buffer.seed)
        self._reset_generator = torch.Generator(device=self.device)
        self._reset_generator.manual_seed(torch.initial_seed() if cfg.seed is None else cfg.seed)
        self._all_ids = torch.arange(self.num_envs, device=self.device)
        self.generation_report: dict = {}
        self.command_manager.reset()
        if cfg.reset_mode != "table":
            self._collect_reset_states()

    def _tcp_pose(self) -> torch.Tensor:
        return pickup_mdp.hand_pose(self, self._hand_cfg)

    def _reset_table(self, env_ids: torch.Tensor, *, generating: bool = False) -> None:
        mdp.reset_scene_to_default(self, env_ids)
        cube = self.scene["object"]
        state = cube.data.default_root_state.torch[env_ids].clone()
        generator = self._bank_generator if generating else self._reset_generator
        offsets = torch.rand((env_ids.numel(), 2), device=self.device, generator=generator)
        state[:, 0] = 0.45 + 0.10 * offsets[:, 0]
        state[:, 1] = -0.08 + 0.16 * offsets[:, 1]
        state[:, :3] += self.scene.env_origins[env_ids]
        cube.write_root_pose_to_sim_index(root_pose=state[:, :7], env_ids=env_ids)
        cube.write_root_velocity_to_sim_index(root_velocity=state[:, 7:], env_ids=env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(0.0)

    def _physics_step(self, actions: torch.Tensor) -> None:
        """Advance only the action/physics path while producing reset candidates."""
        self.action_manager.process_action(actions)
        steps_per_call = self.cfg.decimation if self._physics_handles_decimation else 1
        for _ in range(0, self.cfg.decimation, steps_per_call):
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt * steps_per_call)

    def scripted_action(self, target: torch.Tensor, *, close: bool) -> torch.Tensor:
        """Create a bounded relative IK action for target poses [m, xyzw]."""
        current = self._tcp_pose()
        position_error, rotation_error = compute_pose_error(
            current[:, :3], current[:, 3:], target[:, :3], target[:, 3:], rot_error_type="axis_angle"
        )
        action = torch.zeros((self.num_envs, 7), device=self.device)
        action[:, :3] = (position_error / 0.025).clamp(-1.0, 1.0)
        action[:, 3:6] = (rotation_error / 0.1).clamp(-1.0, 1.0)
        action[:, 6] = -1.0 if close else 1.0
        return action

    def _drive(self, target: torch.Tensor, duration: float, *, close: bool = False) -> None:
        for _ in range(math.ceil(duration / self.step_dt)):
            self._physics_step(self.scripted_action(target, close=close))

    def _generate(self, env_ids: torch.Tensor, stage: str) -> torch.Tensor:
        # All candidates in one batch share the same generator phase. Unselected environments
        # may advance during prefill; the normal reset lifecycle runs before training begins.
        self._reset_table(env_ids, generating=True)
        self.action_manager.reset()
        rest = self._tcp_pose().clone()
        self._drive(rest, 0.2)
        if stage == "table":
            return self._candidate_valid(env_ids)

        cube_position = pickup_mdp.object_position(self).clone()
        target = rest.clone()
        target[:, 3:] = target.new_tensor((1.0, 0.0, 0.0, 0.0))
        target[:, :3] = cube_position
        target[:, 2] += 0.10
        if stage == "approach":
            fraction = torch.rand((self.num_envs, 1), device=self.device, generator=self._bank_generator)
            target[:, :3] = torch.lerp(rest[:, :3], target[:, :3], fraction)
        self._drive(target, 5.0)
        if stage == "approach":
            return self._candidate_valid(env_ids) & (
                torch.linalg.vector_norm(self._tcp_pose()[env_ids, :3] - target[env_ids, :3], dim=-1) < 0.02
            )

        target[:, :3] = cube_position
        if stage == "pregrasp":
            target[:, 2] += 0.10 * torch.rand((self.num_envs,), device=self.device, generator=self._bank_generator)
        self._drive(target, 2.0)
        aligned = torch.linalg.vector_norm(self._tcp_pose()[env_ids, :3] - target[env_ids, :3], dim=-1) < 0.015
        if stage == "pregrasp":
            return self._candidate_valid(env_ids) & aligned

        # Establish the grasp through physical finger closure, then prove it can lift.
        self._drive(target, 0.6, close=True)
        target[:, 2] += 0.025 + 0.073 * torch.rand((self.num_envs,), device=self.device, generator=self._bank_generator)
        self._drive(target, 1.0, close=True)
        height = pickup_mdp.object_position(self)[env_ids, 2]
        attached = torch.linalg.vector_norm(pickup_mdp.object_to_hand(self, self._hand_cfg)[env_ids], dim=-1) < 0.035
        valid = self._candidate_valid(env_ids) & aligned & attached & (height > 0.035)
        # Contact solver caches are not scene state. Restore the actual snapshot and continue
        # holding before accepting it, so a grasp must survive reconstruction of its contacts.
        state = self._capture(env_ids)
        self._restore_reset_state(state, env_ids)
        self._drive(target, 1.0, close=True)
        height = pickup_mdp.object_position(self)[env_ids, 2]
        return valid & self._candidate_valid(env_ids) & (height > 0.035)

    def _candidate_valid(self, env_ids: torch.Tensor) -> torch.Tensor:
        robot = self.scene["robot"]
        cube = self.scene["object"]
        position = pickup_mdp.object_position(self)[env_ids]
        goal_cfg = self.termination_manager.get_term_cfg("success")
        already_successful = goal_cfg.func(self, **goal_cfg.params)[env_ids]
        return (
            torch.isfinite(robot.data.joint_pos.torch[env_ids]).all(dim=-1)
            & torch.isfinite(cube.data.root_state_w.torch[env_ids]).all(dim=-1)
            & (position[:, 2] > 0.017)
            & ~already_successful
            & (position[:, 0] > 0.35)
            & (position[:, 0] < 0.65)
            & (position[:, 1].abs() < 0.20)
            & (torch.linalg.vector_norm(cube.data.root_lin_vel_w.torch[env_ids], dim=-1) < 0.15)
            & (robot.data.joint_vel.torch[env_ids].abs().amax(dim=-1) < 3.0)
        )

    def _capture(self, env_ids: torch.Tensor) -> dict:
        state = self.scene.get_state(is_relative=True)
        return {
            kind: {
                name: {field: value[env_ids].clone() for field, value in fields.items()}
                for name, fields in assets.items()
            }
            for kind, assets in state.items()
        }

    def _restore_reset_state(self, state: ResetStateBuffer.State, env_ids: torch.Tensor) -> None:
        self.scene.reset_to(state, env_ids=env_ids, is_relative=True)
        # Scene state contains physical velocities. The position-controlled IK action requires
        # zero desired joint velocities rather than continuing those captured motions.
        self.scene["robot"].actuators.target_command.set_velocity_index(
            value=torch.zeros_like(state["articulation"]["robot"]["joint_velocity"]), env_ids=env_ids
        )
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(0.0)

    def _collect_reset_states(self) -> None:
        self.reset_buffer = ResetStateBuffer(self.cfg.reset_buffer)
        generators = {
            stage: (lambda env_ids, stage=stage: self._generate(env_ids, stage))
            for stage in ("table", "approach", "pregrasp", "grasp")
        }
        self.reset_buffer.collect(self._all_ids, generators, self._capture)
        sampler_cfg = clone(self.cfg.reset_sampler)
        if self.cfg.reset_mode == "uniform":
            sampler_cfg.sampling.strategies = [UniformSamplingStrategyCfg()]
        self.reset_sampler = instantiate(sampler_cfg, self.num_envs, self.reset_buffer.num_states, self.device)
        self.generation_report = {
            "seed": self.cfg.reset_buffer.seed,
            "num_states": self.reset_buffer.num_states,
            "stages": {
                name: int((self.reset_buffer.group_ids == i).sum().item())
                for i, name in enumerate(self.reset_buffer.group_names)
            },
            "collection": self.reset_buffer.collection_report,
            "validation": "Finite state, low velocity, replayed physical grasp; no initial goal success.",
        }
        if self.cfg.log_dir:
            rank = os.getenv("RANK")
            filename = "reset_states.pt" if rank is None else f"reset_states_rank_{rank}.pt"
            self.save_reset_buffer(Path(self.cfg.log_dir) / filename)

    def save_reset_buffer(self, path: str | Path) -> None:
        """Save generated scene states and stage metadata for reproducing the training distribution."""
        if self.reset_buffer is None:
            raise ValueError("Table-only evaluation does not generate a reset buffer.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state": self.reset_buffer.state,
                "group_names": self.reset_buffer.group_names,
                "group_ids": self.reset_buffer.group_ids,
                "report": self.generation_report,
            },
            path,
        )
        path.with_suffix(".json").write_text(json.dumps(self.generation_report, indent=2) + "\n")

    def _reset_idx(self, env_ids: torch.Tensor | slice):
        ids = self.scene._ALL_INDICES[env_ids]
        success = self.termination_manager.get_term("success")[ids].clone()
        valid = (self.episode_length_buf[ids] > 0) & self.reset_buf[ids]
        metrics: dict[str, torch.Tensor] = {}
        rows = None
        if self.reset_sampler is not None:
            previous_rows = self.reset_sampler.row_ids[ids]
            previous_groups = self.reset_buffer.group_ids[previous_rows]
            for group_id, name in enumerate(self.reset_buffer.group_names):
                mask = valid & (previous_groups == group_id)
                if mask.any():
                    metrics[f"Reset/{name}_success"] = success[mask].float().mean()
            # Credit the row that produced the episode before clearing success and episode buffers.
            rows = self.reset_sampler.reset(ids, success=success, valid=valid)
            for group_id, name in enumerate(self.reset_buffer.group_names):
                mask = self.reset_buffer.group_ids == group_id
                metrics[f"Reset/{name}_probability"] = self.reset_sampler.probabilities[mask].sum()
        if valid.any():
            metrics["Metrics/success_rate"] = success[valid].float().mean()
        super()._reset_idx(env_ids)
        if rows is None:
            self._reset_table(ids)
        else:
            self._restore_reset_state(self.reset_buffer.get_state(rows), ids)
        self.extras["log"].update(metrics)
