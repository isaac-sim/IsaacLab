# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exploration actions and the recorder term that writes native NeRD datasets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import warp as wp
from newton.solvers import SolverMuJoCo
from newton_neural.contacts import ContactInputProbes
from newton_neural.data import TrajectoryWriter
from newton_neural.systems import BindingArticulation, SystemArticulation

from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import RecorderTerm
from isaaclab.utils import string_to_callable

from ..physics.nerd_solver import physical_contract
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
    """Capture one frame per environment step and write contiguous per-environment shards.

    The pre-step hook copies state and contact inputs; the post-step hook adds the consumed controls, the
    free-motion label and validity. Frames stay batched across environments and reach the native
    ``TrajectoryWriter`` every :attr:`~NeRDDynamicsRecorderCfg.shard_frames` steps, so recording materializes
    no per-episode data. A reset marks the next frame of its environments as an episode start.

    The term binds the configured articulation itself and records the signals that the configured contact
    input and prediction need. The dataset describes the recorded robot, so that a model of any architecture
    is built on it when fitting. The workflow closes the recorder when it ends.
    """

    def __init__(self, cfg: NeRDDynamicsRecorderCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.backend = env.sim.backend_manager.backend
        model = self.backend.model
        articulations = env.scene[cfg.asset_name].root_view.articulation_ids.numpy().ravel().tolist()
        self.system = SystemArticulation.from_model(model, articulations)
        self.binding = self.system.bind(model, articulations)
        self.contact_input = None
        self.sensor = None
        if cfg.contact_bodies:
            self.contact_input = ContactInputProbes.from_model(
                model, self.system, bodies=cfg.contact_bodies, articulation=articulations
            )
            self.sensor = self.contact_input.sensor(model, self.binding)
        prediction = string_to_callable(cfg.prediction)()
        self.free_columns = None
        if not prediction.integrates:
            self.free_columns = free_acceleration_indices(self.backend.solver, self.binding)
        contact_signals = {} if self.contact_input is None else self.contact_input.signals
        self.signals = {**self.system.signals, **contact_signals, **prediction.signals(self.system)}
        """Specifications of the recorded signals, keyed by name."""
        self.writer: TrajectoryWriter | None = None
        self._frames: list[dict[str, torch.Tensor]] = []
        self._episode_starts: list[torch.Tensor] = []
        self._valid: list[torch.Tensor] = []
        # Environments whose next frame starts an episode. The first recorded frame always does.
        self._starting = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self._continues = False

    def description(self) -> dict:
        """Describe the recorded robot as plain data, from which fitting builds a model.

        Returns:
            The system and contact input by class name and configuration, the physical-model contract that a
            checkpoint must match at deployment, and the settings that selected the recorded signals.
        """
        contact_input = self.contact_input
        return {
            "system": {"name": type(self.system).__name__, "config": self.system.config()},
            "contact_input": (
                None
                if contact_input is None
                else {"name": type(contact_input).__name__, "config": contact_input.config()}
            ),
            "physical_model": physical_contract(
                self.backend.model, self.binding, free_motion=self.free_columns is not None
            ),
            "model_inputs": {"contact_bodies": self.cfg.contact_bodies, "prediction": self.cfg.prediction},
        }

    def record_post_reset(self, env_ids: Sequence[int] | slice | None) -> tuple[None, None]:
        self._starting[slice(None) if env_ids is None else env_ids] = True
        return None, None

    def record_pre_step(self) -> tuple[None, None]:
        if self.writer is None:
            # Opened at the first step: startup events run after the managers are built and may still edit the robot.
            recorders = self._env.cfg.recorders
            generation = {**self._env.recorder_manager.get_ep_meta(), "isaaclab": self.description()}
            self.writer = TrajectoryWriter(
                Path(recorders.dataset_export_dir_path) / recorders.dataset_filename,
                self.signals,
                self._env.step_dt,
                generation=generation,
            )
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

    def record_post_step(self) -> tuple[None, None]:
        frame = self.frame
        # Buffered frames outlive the step, so the control buffers that later steps overwrite are copied.
        controls = self.binding.read_control(self.backend.control)
        frame.update({name: value.clone() for name, value in controls.items()})
        if self.free_columns is not None:
            acceleration = wp.to_torch(self.backend.solver.mjw_data.qacc_smooth)
            frame["joint_qdd_free"] = acceleration.gather(1, self.free_columns)
        valid = ~self._env.reset_terminated
        for value in frame.values():
            valid = valid & torch.isfinite(value).flatten(1).all(dim=1)
        self._frames.append(frame)
        self._episode_starts.append(self._starting.clone())
        self._valid.append(valid)
        self._starting.zero_()
        if len(self._frames) >= self.cfg.shard_frames:
            self.flush()
        return None, None

    def flush(self) -> None:
        """Write the buffered frames of all environments as one shard that continues the previous one."""
        if not self._frames:
            return
        signals = {name: torch.stack([frame[name] for frame in self._frames], dim=1) for name in self._frames[0]}
        episode_start, valid = torch.stack(self._episode_starts, dim=1), torch.stack(self._valid, dim=1)
        self.writer.write(signals, episode_start, valid, continues=self._continues)
        self._continues = True
        self._frames.clear()
        self._episode_starts.clear()
        self._valid.clear()

    def close(self, *, complete: bool = True) -> None:
        """Write the remaining frames and publish the dataset index.

        Args:
            complete: Whether the recording ran to its end. An interrupted recording keeps the shards that
                were already written and is marked incomplete.
        """
        if self.writer is None:
            return
        if complete:
            self.flush()
        writer, self.writer = self.writer, None
        writer.close(complete=complete)
