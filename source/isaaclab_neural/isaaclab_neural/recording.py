# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exploration actions, standard recorder terms and native NeRD dataset output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonBackend
from newton.solvers import SolverMuJoCo
from newton_neural.data import TrajectoryWriter
from newton_neural.systems import BindingArticulation, SystemArticulation

from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import RecorderTerm
from isaaclab.utils.datasets import DatasetFileHandlerBase, EpisodeData

from .recording_cfg import NeRDDynamicsRecorderCfg, NeRDExplorationActionCfg


class NeRDExplorationAction(JointPositionAction):
    """An exploration controller with a state sampler and a native joint-target generator.

    Task implementations construct ``states`` and ``commands``, then call :meth:`initialize_sampling` once.
    Only the reset event writes sampled physical state. The action manager owns command resets.
    """

    def __init__(self, cfg: NeRDExplorationActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.backend = env.sim.backend_manager.backend
        articulations = self._asset.root_view.articulation_ids.numpy().ravel().tolist()
        system = SystemArticulation.from_model(self.backend.model, articulations)
        self.binding = system.bind(self.backend.model, articulations)
        labels = system.spec.actuated_joint_labels
        self.joint_order = torch.tensor([labels.index(name) for name in self._asset.joint_names], device=self.device)
        self.default_joint_pos = torch.empty(len(labels), device=self.device)
        self.default_joint_pos[self.joint_order] = self._asset.data.default_joint_pos.torch[0]
        self.length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def initialize_sampling(self) -> None:
        """Bind declared samplers to the robot once, before the first episode."""
        self.states.start(self.binding.state_bounds())
        low, high = self.binding.command_bounds()["joint_target_q"]
        self.commands.start(self.num_envs, low, high, self._env.step_dt, sizes={"joint_target_q": len(low)})

    def process_actions(self, actions: torch.Tensor) -> None:
        frames = self.binding.read_state(self.backend.state_0)
        targets = self.commands.next(frames)[:, self.joint_order]
        super().process_actions(targets)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[slice(None) if env_ids is None else env_ids] = True
        self.commands.reset(mask)
        shortest, longest = self.cfg.episode_frames
        lengths = torch.randint(shortest, longest + 1, self.length.shape, device=self.device)
        self.length.copy_(torch.where(mask, lengths, self.length))


def reset_exploration_state(env: ManagerBasedEnv, env_ids: torch.Tensor | slice | None) -> None:
    """Sample robot state at an explicit environment reset event."""
    exploration = env.action_manager.get_term("joint_targets")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    frames = exploration.binding.place(exploration.states.sample(len(env_ids)), env_ids)
    joint_q, joint_qd = frames["joint_q"], frames["joint_qd"]
    robot = exploration._asset
    if exploration.binding.system.floating_base:
        pose = torch.cat((joint_q[:, :3], torch.nn.functional.normalize(joint_q[:, 3:7], dim=1)), dim=1)
        robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        robot.write_root_com_velocity_to_sim_index(root_velocity=joint_qd[:, :6], env_ids=env_ids)
        joint_q, joint_qd = joint_q[:, 7:], joint_qd[:, 6:]
    robot.write_joint_position_to_sim_index(position=joint_q[:, exploration.joint_order], env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_qd[:, exploration.joint_order], env_ids=env_ids)


def free_acceleration_indices(solver: SolverMuJoCo, binding: BindingArticulation) -> torch.Tensor:
    """Map the selected robot's Newton DOFs to MuJoCo's free-acceleration columns."""
    newton_dofs = solver.mjc_dof_to_newton_dof.numpy()
    worlds, columns = np.nonzero(newton_dofs >= 0)
    column = np.full(solver.model.joint_dof_count, -1)
    column[newton_dofs[worlds, columns]] = columns
    selected = column[binding.joint_dof_ids]
    if (selected < 0).any():
        raise ValueError("MuJoCo Warp must simulate every recorded robot DOF")
    return torch.as_tensor(selected, dtype=torch.long, device=wp.device_to_torch(solver.model.device))


def invalid_exploration_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Restart divergent worlds before their state is used for another transition."""
    exploration = env.action_manager.get_term("joint_targets")
    values = exploration.binding.read_state(exploration.backend.state_0).values()
    return ~torch.stack([(value.abs() <= 1000).flatten(1).all(dim=1) for value in values]).all(dim=0)


def exploration_timeout(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End each sampled exploration segment at its declared frame count."""
    exploration = env.action_manager.get_term("joint_targets")
    return env.episode_length_buf >= exploration.length


class NeRDDynamicsRecorder(RecorderTerm):
    """Capture complete transitions through the existing pre/post-step recorder hooks.

    The entrypoint supplies optional contact probes and free-acceleration columns after constructing the model.
    RecorderManager owns per-world episode storage and exports before automatic resets.
    """

    binding: BindingArticulation
    backend: NewtonBackend

    def __init__(self, cfg: NeRDDynamicsRecorderCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.sensor = None
        self.free_columns = None

    def record_pre_step(self) -> tuple[None, None]:
        self._env.sim.forward()
        backend = self.backend
        if backend.collision_pipeline is not None:
            horizon = min(backend.num_substeps, backend.cfg.collision_decimation or backend.num_substeps)
            backend.collision_pipeline.collide(backend.state_0, backend.contacts, dt=horizon * backend.solver_dt)
        self.frame = {name: value.clone() for name, value in self.binding.read_state(backend.state_0).items()}
        if self.sensor is not None:
            features = self.sensor.update(backend.state_0, backend.contacts)
            self.frame.update({name: value.clone() for name, value in features.items()})
        return None, None

    def record_post_step(self) -> tuple[str, dict[str, torch.Tensor]]:
        self.frame.update(self.binding.read_control(self.backend.control))
        if self.free_columns is not None:
            acceleration = wp.to_torch(self.backend.solver.mjw_data.qacc_smooth)
            self.frame["joint_qdd_free"] = acceleration.gather(1, self.free_columns)
        valid = ~self._env.reset_terminated
        for value in self.frame.values():
            valid = valid & torch.isfinite(value).flatten(1).all(dim=1)
        return "dynamics", {**self.frame, "valid": valid}


class NeRDDatasetFileHandler(DatasetFileHandlerBase):
    """Write RecorderManager episodes directly into the native NeRD trajectory format.

    Each episode is an independent native world. Padding is invalid, and a shard never continues another
    episode. The workflow supplies the native writer after constructing the model; no intermediate dataset
    or environment reconstruction is involved.
    """

    frame_count: int

    def __init__(self):
        self.writer: TrajectoryWriter | None = None
        self.episodes = []
        self.episode_count = 0
        self.complete = True
        self.env_args = None

    def create(self, file_path: str, env_name: str | None = None) -> None:
        self.path = Path(file_path)
        self.env_name = env_name

    def add_env_args(self, env_args: dict) -> None:
        if self.env_args is not None and self.env_args != env_args:
            raise ValueError("NeRD recording settings changed during capture")
        self.env_args = env_args

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None, dataset_compression: bool = True) -> None:
        if demo_id is not None:
            raise ValueError("Native NeRD datasets identify independent episodes by world index")
        self.episodes.append(episode.data["dynamics"])
        self.episode_count += 1
        if len(self.episodes) >= 4096:
            self.flush()

    def flush(self) -> None:
        if not self.episodes:
            return
        first = self.episodes[0]
        lengths = [len(episode["valid"]) for episode in self.episodes]
        shape = (len(lengths), self.frame_count)
        frames = {
            name: torch.zeros((*shape, *value.shape[1:]), dtype=value.dtype, device=value.device)
            for name, value in first.items()
        }
        for world, (episode, length) in enumerate(zip(self.episodes, lengths, strict=True)):
            for name, value in episode.items():
                frames[name][world, :length] = value
        valid = frames.pop("valid")
        starts = torch.zeros_like(valid)
        starts[:, 0] = True
        self.writer.write(frames, starts, valid)
        self.episodes.clear()

    def close(self) -> None:
        if self.writer is not None:
            self.flush()
            self.writer.close(complete=self.complete)
            self.writer = None

    def get_env_name(self) -> str | None:
        return self.env_name

    def get_num_episodes(self) -> int:
        return self.episode_count

    def open(self, file_path: str, mode: str = "r") -> None:
        raise NotImplementedError("Read native recordings with newton_neural.data.TrajectoryDataset")

    def load_episode(self, episode_name: str) -> EpisodeData:
        raise NotImplementedError("Read native recordings with newton_neural.data.TrajectoryWindows")
