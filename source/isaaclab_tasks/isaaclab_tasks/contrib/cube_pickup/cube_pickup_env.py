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
from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
from isaaclab.envs.reset import ResetStateBuffer
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import clone, instantiate
from isaaclab.utils.math import compute_pose_error, quat_apply_inverse

from ...utils import ResetStateSampler, collect_reset_states
from ...utils.sampling import UniformScoreCfg
from . import mdp as pickup_mdp
from .cube_pickup_env_cfg import CubePickupEnvCfg


class CubePickupEnv(ManagerBasedRLEnv):
    """Train sparse pickup from snapshots of genuine simulated approach and grasp rollouts."""

    cfg: CubePickupEnvCfg

    def __init__(self, cfg: CubePickupEnvCfg, **kwargs) -> None:
        self.reset_buffer: ResetStateBuffer | None = None
        self.reset_sampler: ResetStateSampler | None = None
        self.reset_group_names: tuple[str, ...] = ()
        self.reset_group_ids = torch.empty(0, dtype=torch.long, device=cfg.sim.device)
        self._step_outcome_counts: torch.Tensor | None = None
        self._requested_reset_state_ids: torch.Tensor | None = None
        self._arm_scale = torch.tensor(cfg.actions.arm.scale, dtype=torch.float32, device=cfg.sim.device).expand(6)
        self._arm_offset = torch.tensor(cfg.actions.arm.offset, dtype=torch.float32, device=cfg.sim.device).expand(6)
        offset = cfg.actions.arm.body_offset
        self._tcp_offset_pos = (
            torch.tensor(offset.pos, dtype=torch.float32, device=cfg.sim.device).expand(cfg.scene.num_envs, -1)
            if offset is not None
            else None
        )
        self._tcp_offset_rot = (
            torch.tensor(offset.rot, dtype=torch.float32, device=cfg.sim.device).expand(cfg.scene.num_envs, -1)
            if offset is not None
            else None
        )
        super().__init__(cfg, **kwargs)
        self._hand_cfg = SceneEntityCfg("robot", body_names="panda_hand")
        self._hand_cfg.resolve(self.scene)
        self.generation_report: dict = {}
        self.command_manager.reset()
        if cfg.reset_mode != "table":
            self._collect_reset_states()

    def reset(
        self,
        env_ids: torch.Tensor | slice | None = None,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[VecEnvObs, dict]:
        """Reset selected environments, optionally to explicit rows of the generated catalog.

        Args:
            env_ids: Environment indices or a slice. None resets all environments.
            seed: Random seed for the reset. None leaves the random state unchanged.
            options: Optional ``reset_state_ids`` tensor with one catalog row per selected
                environment, in the same order as ``env_ids``. Requires a generated catalog.

        Returns:
            Observations and reset information.
        """
        state_ids = options.get("reset_state_ids") if options is not None else None
        if state_ids is not None and self.reset_sampler is None:
            raise ValueError("Explicit reset_state_ids require a generated reset catalog.")
        self._requested_reset_state_ids = state_ids
        self._step_outcome_counts = None
        try:
            return super().reset(env_ids, seed=seed, options=options)
        finally:
            self._requested_reset_state_ids = None
            self._step_outcome_counts = None

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Publish fresh metrics for natural outcomes observed during this step."""
        self.extras["log"] = {}
        self._step_outcome_counts = None
        try:
            result = super().step(action)
            if self._step_outcome_counts is not None:
                counts = self._step_outcome_counts[0].cpu().tolist()
                rates = self._step_outcome_counts[1] / self._step_outcome_counts[0].clamp_min(1)
                if counts[0]:
                    self.extras["log"]["Metrics/success_rate"] = rates[0]
                for group_id, name in enumerate(self.reset_group_names):
                    if counts[1 + group_id]:
                        self.extras["log"][f"Reset/{name}_success"] = rates[1 + group_id]
            return result
        finally:
            self._step_outcome_counts = None

    def scripted_action(self, target: torch.Tensor, *, close: bool) -> torch.Tensor:
        """Create normalized relative IK actions for target grasp-center poses.

        Args:
            target: Poses with shape (num_envs, 7). Positions [m] are relative to environment
                origins in world axes; orientations are world-frame quaternions in xyzw order.
            close: Whether to close the gripper.

        Returns:
            Actions with shape (num_envs, 7): scaled root-frame translation and axis-angle
            rotation increments, followed by the binary gripper command (-1 closed, +1 open).
        """
        current = pickup_mdp.hand_pose(self, self._hand_cfg)
        position_error, rotation_error = compute_pose_error(
            current[:, :3], current[:, 3:], target[:, :3], target[:, 3:], rot_error_type="axis_angle"
        )
        root_orientation = self.scene["robot"].data.root_quat_w.torch
        delta = torch.cat(
            (
                quat_apply_inverse(root_orientation, position_error),
                quat_apply_inverse(root_orientation, rotation_error),
            ),
            dim=-1,
        )
        action = torch.zeros((self.num_envs, 7), device=self.device)
        action[:, :6] = torch.where(self._arm_scale != 0.0, (delta - self._arm_offset) / self._arm_scale, 0.0).clamp(
            -1.0, 1.0
        )
        action[:, 6] = -1.0 if close else 1.0
        return action

    def save_reset_buffer(self, path: str | Path) -> None:
        """Save generated scene states and stage metadata for reproducing the training distribution."""
        if self.reset_buffer is None:
            raise ValueError("Table-only evaluation does not generate a reset buffer.")
        path = Path(path)
        report_path = path.with_suffix(".json")
        if path.resolve() == report_path.resolve():
            raise ValueError("Reset bank and JSON report must use different paths.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state": self.reset_buffer.states,
                "group_names": self.reset_group_names,
                "group_ids": self.reset_group_ids,
                "report": self.generation_report,
            },
            path,
        )
        report_path.write_text(json.dumps(self.generation_report, indent=2) + "\n")

    def _reset_idx(self, env_ids: torch.Tensor | slice) -> None:
        ids = self.scene._ALL_INDICES[env_ids] if isinstance(env_ids, slice) else env_ids
        success = self.termination_manager.get_term("success")[ids]
        valid = (self.episode_length_buf[ids] > 0) & self.reset_buf[ids]
        metrics: dict[str, torch.Tensor] = {}
        rows = None
        if self._step_outcome_counts is None:
            self._step_outcome_counts = torch.zeros((2, 1 + len(self.reset_group_names)), device=self.device)
        self._step_outcome_counts[0, 0] += valid.sum()
        self._step_outcome_counts[1, 0] += (success & valid).sum()
        if self.reset_sampler is not None:
            previous_rows = self.reset_sampler.get_state_ids(ids)
            previous_groups = self.reset_group_ids[previous_rows]
            self._step_outcome_counts[0, 1:].scatter_add_(0, previous_groups, valid.float())
            self._step_outcome_counts[1, 1:].scatter_add_(0, previous_groups, (success & valid).float())
            rows = self.reset_sampler.reset(
                ids, success=success, valid=valid, state_ids=self._requested_reset_state_ids
            )
            probabilities = torch.zeros(len(self.reset_group_names), device=self.device).scatter_add_(
                0, self.reset_group_ids, self.reset_sampler.probabilities
            )
            for group_id, name in enumerate(self.reset_group_names):
                metrics[f"Reset/{name}_probability"] = probabilities[group_id]
        super()._reset_idx(env_ids)
        if rows is None:
            self._reset_table(ids)
        else:
            self._restore_reset_state(self.reset_buffer.get_state(rows), ids)
        self.extras["log"].update(metrics)

    def _reset_table(self, env_ids: torch.Tensor) -> None:
        mdp.reset_scene_to_default(self, env_ids)
        cube = self.scene["object"]
        state = cube.data.default_root_state.torch[env_ids].clone()
        offsets = torch.rand((env_ids.numel(), 2), device=self.device)
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

    def _drive(self, target: torch.Tensor, duration: float, *, close: bool = False) -> None:
        for _ in range(math.ceil(duration / self.step_dt)):
            self._physics_step(self.scripted_action(target, close=close))

    def _generate(self, env_ids: torch.Tensor, stage: str) -> torch.Tensor:
        self._reset_table(env_ids)
        self.action_manager.reset()
        rest = pickup_mdp.hand_pose(self, self._hand_cfg)
        self._drive(rest, 0.2)
        if stage == "table":
            return self._candidate_valid(env_ids)

        cube_position = pickup_mdp.object_position(self).clone()
        target = rest.clone()
        target[:, 3:] = target.new_tensor((1.0, 0.0, 0.0, 0.0))
        target[:, :3] = cube_position
        target[:, 2] += 0.10
        if stage == "approach":
            fraction = torch.rand((self.num_envs, 1), device=self.device)
            target[:, :3] = torch.lerp(rest[:, :3], target[:, :3], fraction)
        self._drive(target, 5.0)
        if stage == "approach":
            return self._candidate_valid(env_ids) & (
                torch.linalg.vector_norm(
                    pickup_mdp.hand_pose(self, self._hand_cfg)[env_ids, :3] - target[env_ids, :3], dim=-1
                )
                < 0.02
            )

        target[:, :3] = cube_position
        if stage == "pregrasp":
            target[:, 2] += 0.10 * torch.rand((self.num_envs,), device=self.device)
        self._drive(target, 2.0)
        aligned = (
            torch.linalg.vector_norm(
                pickup_mdp.hand_pose(self, self._hand_cfg)[env_ids, :3] - target[env_ids, :3], dim=-1
            )
            < 0.015
        )
        if stage == "pregrasp":
            return self._candidate_valid(env_ids) & aligned

        self._drive(target, 0.6, close=True)
        target[:, 2] += 0.025 + 0.073 * torch.rand((self.num_envs,), device=self.device)
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
        attached = torch.linalg.vector_norm(pickup_mdp.object_to_hand(self, self._hand_cfg)[env_ids], dim=-1) < 0.035
        return valid & self._candidate_valid(env_ids) & attached & (height > 0.035)

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

    def _capture(self, env_ids: torch.Tensor) -> ResetStateBuffer.State:
        state = self.scene.get_state(is_relative=True)
        return {
            kind: {name: {field: value[env_ids] for field, value in fields.items()} for name, fields in assets.items()}
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
        collection_cfg = clone(self.cfg.reset_buffer)
        collection_cfg.seed = (collection_cfg.seed + int(os.getenv("RANK", "0"))) % 2**32
        generators = {
            stage: (lambda env_ids, stage=stage: self._generate(env_ids, stage))
            for stage in ("table", "approach", "pregrasp", "grasp")
        }
        states, self.reset_group_ids, report = collect_reset_states(
            collection_cfg, self.scene._ALL_INDICES, generators, self._capture
        )
        self.reset_buffer = ResetStateBuffer(states)
        self.reset_group_names = tuple(generators)
        sampler_cfg = clone(self.cfg.reset_sampler)
        if self.cfg.reset_mode == "uniform":
            sampler_cfg.sampling.strategies = [UniformScoreCfg()]
        self.reset_sampler = instantiate(sampler_cfg, self.num_envs, self.reset_buffer.num_states, self.device)
        self.generation_report = {
            "seed": collection_cfg.seed,
            "num_states": self.reset_buffer.num_states,
            "stages": {name: group["accepted"] for name, group in report.items()},
            "collection": report,
            "validation": "Finite state, low velocity, replayed physical grasp; no initial goal success.",
        }
        if self.cfg.log_dir:
            rank = os.getenv("RANK")
            filename = "reset_states.pt" if rank is None else f"reset_states_rank_{rank}.pt"
            self.save_reset_buffer(Path(self.cfg.log_dir) / filename)
