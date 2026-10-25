# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native recorder terms for generalized-coordinate learned dynamics."""

from functools import partial

import torch
import warp as wp

from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


@configclass
class NeRDRecorderCfg(RecorderTermCfg):
    """Select the transition representation independently of its consumer or storage."""

    class_type: str = "{DIR}.recorders:NeRDRecorder"
    asset_name: str = "robot"
    control: str = "joint_f"
    """Explicit applied efforts (fixed implicit damping retained) or position targets, in Newton order."""
    heading: bool = False
    contact_bodies: tuple[str, ...] = ()
    root_body: str = ""
    record_to_episodes: bool = False
    """Retain transitions through RecorderManager; streaming consumers read ``latest`` either way."""


@configclass
class NeRDRecorderManagerCfg(RecorderManagerBaseCfg):
    """Optional native dynamics capture; task playback has no capture cost by default."""

    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_NONE
    transition: NeRDRecorderCfg | None = None


class NeRDRecorder(RecorderTerm):
    """Capture the terminal successor before autoreset into persistent device tensors.

    ``latest`` is valid until the next step. Replay and curiosity consumers copy the fields they
    retain; this term owns neither a dataset writer nor an episode/replay buffer.
    """

    requires_post_step_observations = False

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from isaaclab_newton.physics.nerd import JointDynamics

        self.manager = env.sim.backend_manager
        self.robot = env.scene[cfg.asset_name]
        if cfg.control == "joint_f" and (env.cfg.decimation != 1 or env.cfg.sim.physics.num_substeps != 1):
            raise ValueError("Effort recording requires one physics step per environment step.")
        model = self.manager.get_model()
        self.dynamics = JointDynamics(
            model=model,
            frame_dt=env.step_dt,
            control=cfg.control,
            heading=cfg.heading,
            contact_bodies=cfg.contact_bodies,
            root_body=cfg.root_body,
        )
        self.manager.register_state_force_callback(
            partial(self.dynamics.check_inputs, control=self.manager.get_control())
        )
        columns = {
            name.rsplit("/", 1)[-1]: int(start)
            for name, start, world in zip(model.joint_label, model.joint_q_start.numpy(), model.joint_world.numpy())
            if world == 0
        }
        self._target_columns = torch.tensor([columns[name] for name in self.robot.joint_names], device=self.device)
        width = self.dynamics.codec.state_shape[0]
        self.latest = {
            "state": torch.empty(self.num_envs, width, device=self.device),
            "next_state": torch.empty(self.num_envs, width, device=self.device),
            "control": torch.empty(self.num_envs, self.dynamics.control_dim, device=self.device),
            "contacts": torch.empty(self.num_envs, self.dynamics.contact_dim, device=self.device),
            "terminated": torch.empty(self.num_envs, dtype=torch.bool, device=self.device),
            "truncated": torch.empty(self.num_envs, dtype=torch.bool, device=self.device),
        }

        self._has_transition = False

    def record_pre_step(self):
        state = self.manager.get_scene_data_backend().state
        self.dynamics.codec.read_into(state, self.latest["state"])
        self.latest["contacts"].copy_(self.dynamics.contacts(state))
        return None, None

    def record_post_step(self):
        self.dynamics.codec.read_into(self.manager.get_state_0(), self.latest["next_state"])
        control = self.manager.get_control()
        value = control.joint_f if self.cfg.control == "joint_f" else control.joint_target_q
        self.latest["control"].copy_(wp.to_torch(value).reshape(self.num_envs, -1))
        if self.cfg.control == "joint_target_q":
            # Explicit actuator networks consume targets without publishing them in Newton Control.
            self.latest["control"][:, self._target_columns] = self.robot.actuators.target_command.position.torch
        self.latest["terminated"].copy_(self._env.reset_terminated)
        self.latest["truncated"].copy_(self._env.reset_time_outs)
        self._has_transition = True
        return ("transition", self.latest) if self.cfg.record_to_episodes else (None, None)

    def record_pre_reset(self, env_ids):
        if self._has_transition:
            ids = slice(None) if env_ids is None else env_ids
            self.latest["truncated"][ids] |= ~self.latest["terminated"][ids]
        return None, None
