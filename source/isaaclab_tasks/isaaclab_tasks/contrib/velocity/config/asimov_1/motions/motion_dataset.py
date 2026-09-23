# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul


class MotionDataset:
    """Expert motion transitions used by the AMP discriminator."""

    def __init__(
        self,
        env,
        device: str = "cpu",
        *,
        motion_files: list[str] | None = None,
        body_names: list[str] | None = None,
        anchor_name: str | None = None,
        amp_obs_terms: list[str] | None = None,
        joint_names: list[str] | None = None,
        asset_name: str = "robot",
    ) -> None:
        self.device = device
        self.robot = env.scene[asset_name]
        self.motion_files = list(motion_files)
        self.observation_terms = list(amp_obs_terms)
        self.joint_names = list(joint_names) if joint_names is not None else None

        sim_body_ids, self._body_names_resolved = self.robot.find_bodies(list(body_names), preserve_order=True)
        sim_anchor_ids, self._anchor_names_resolved = self.robot.find_bodies(anchor_name, preserve_order=True)
        self.body_indexes = torch.tensor(sim_body_ids, dtype=torch.long, device=device)
        self.anchor_index = torch.tensor(sim_anchor_ids, dtype=torch.long, device=device)

        self.load_motions()
        self.init_observation_dims()

    def load_motions(self) -> None:
        joint_pos_list = []
        joint_vel_list = []
        body_pos_w_list = []
        body_quat_w_list = []
        body_lin_vel_w_list = []
        body_ang_vel_w_list = []
        fps_list = []
        traj_lengths = []
        joint_pos_offset = None
        if self.joint_names is not None:
            sel_ids = self.robot.find_joints(self.joint_names, preserve_order=True)[0]
            default_joint_pos = getattr(self.robot.data, "nominal_default_joint_pos", None)
            if default_joint_pos is None:
                default_joint_pos = self.robot.data.default_joint_pos.torch
            joint_pos_offset = default_joint_pos[0, sel_ids].detach().cpu().numpy()

        file_body_names = None
        for f in self.motion_files:
            assert os.path.isfile(f), f"Invalid motion file: {f}"
            data = np.load(f)
            if "body_names" in data.files:
                names = [str(n) for n in np.asarray(data["body_names"]).tolist()]
                if file_body_names is None:
                    file_body_names = names
                    wanted = list(self._body_names_resolved) + list(self._anchor_names_resolved)
                    missing = [n for n in wanted if n not in names]
                    if missing:
                        raise ValueError(f"Motion file {f} missing bodies: {missing}")
                    self.body_indexes = torch.tensor(
                        [names.index(n) for n in self._body_names_resolved],
                        dtype=torch.long,
                        device=self.device,
                    )
                    self.anchor_index = torch.tensor(
                        [names.index(n) for n in self._anchor_names_resolved],
                        dtype=torch.long,
                        device=self.device,
                    )
                elif names != file_body_names:
                    raise ValueError(
                        f"Motion file {f} body_names differ from the other files; "
                        "all clips must share one body ordering."
                    )

            fps = np.asarray(data["fps"])
            if fps.size != 1:
                raise ValueError(f"Expected scalar fps in {f}, got shape {fps.shape}")
            fps_list.append(float(fps.item()))
            traj_lengths.append(data["joint_pos"].shape[0])

            joint_pos = data["joint_pos"]
            joint_vel = data["joint_vel"]
            if self.joint_names is not None:
                if "joint_names" not in data.files:
                    raise ValueError(f"Motion file {f} must include joint_names to select cfg joint_names.")
                file_joint_names = [str(name) for name in np.asarray(data["joint_names"]).tolist()]
                missing = [name for name in self.joint_names if name not in file_joint_names]
                if missing:
                    raise ValueError(f"Motion file {f} missing joints: {missing}")
                joint_ids = [file_joint_names.index(name) for name in self.joint_names]
                joint_pos = joint_pos[:, joint_ids]
                joint_vel = joint_vel[:, joint_ids]
                assert joint_pos_offset is not None
                joint_pos = joint_pos - joint_pos_offset

            joint_pos_list.append(torch.tensor(joint_pos, dtype=torch.float32))
            joint_vel_list.append(torch.tensor(joint_vel, dtype=torch.float32))
            body_pos_w_list.append(torch.tensor(data["body_pos_w"], dtype=torch.float32))
            body_quat_w_list.append(torch.tensor(data["body_quat_w"], dtype=torch.float32))
            body_lin_vel_w_list.append(torch.tensor(data["body_lin_vel_w"], dtype=torch.float32))
            body_ang_vel_w_list.append(torch.tensor(data["body_ang_vel_w"], dtype=torch.float32))

        self.joint_pos = torch.cat(joint_pos_list, dim=0).to(self.device)
        self.joint_vel = torch.cat(joint_vel_list, dim=0).to(self.device)
        self.body_pos_w_all = torch.cat(body_pos_w_list, dim=0).to(self.device)
        self.body_quat_w_all = torch.cat(body_quat_w_list, dim=0).to(self.device)
        self.body_lin_vel_w_all = torch.cat(body_lin_vel_w_list, dim=0).to(self.device)
        self.body_ang_vel_w_all = torch.cat(body_ang_vel_w_list, dim=0).to(self.device)

        self.total_dataset_size = sum(traj_lengths)
        self.fps_list = fps_list
        self.index_t, self.index_tp1 = self._build_transition_indices(traj_lengths, self.device)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.body_pos_w_all[:, self.body_indexes].reshape(self.total_dataset_size, -1)

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.body_quat_w_all[:, self.body_indexes].reshape(self.total_dataset_size, -1)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w_all[:, self.body_indexes].reshape(self.total_dataset_size, -1)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w_all[:, self.body_indexes].reshape(self.total_dataset_size, -1)

    @property
    def body_pos_b(self) -> torch.Tensor:
        pos_w = self.body_pos_w_all[:, self.body_indexes]
        num_bodies = pos_w.shape[1]
        anchor_pos = self.anchor_pos_w.unsqueeze(1)
        anchor_quat = self.anchor_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)
        rel_local = quat_apply_inverse(anchor_quat, pos_w - anchor_pos)
        return rel_local.reshape(self.total_dataset_size, -1)

    @property
    def body_quat_b(self) -> torch.Tensor:
        q_body = self.body_quat_w_all[:, self.body_indexes]
        num_bodies = q_body.shape[1]
        q_anchor_inv = quat_conjugate(self.anchor_quat_w).unsqueeze(1).expand(-1, num_bodies, -1)
        return quat_mul(q_anchor_inv, q_body).reshape(self.total_dataset_size, -1)

    @property
    def body_lin_vel_b(self) -> torch.Tensor:
        v_body = self.body_lin_vel_w_all[:, self.body_indexes]
        num_bodies = v_body.shape[1]
        rel = v_body - self.anchor_lin_vel_w.unsqueeze(1)
        anchor_quat = self.anchor_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)
        return quat_apply_inverse(anchor_quat, rel).reshape(self.total_dataset_size, -1)

    @property
    def body_ang_vel_b(self) -> torch.Tensor:
        w_body = self.body_ang_vel_w_all[:, self.body_indexes]
        num_bodies = w_body.shape[1]
        rel = w_body - self.anchor_ang_vel_w.unsqueeze(1)
        anchor_quat = self.anchor_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)
        return quat_apply_inverse(anchor_quat, rel).reshape(self.total_dataset_size, -1)

    @property
    def anchor_height(self) -> torch.Tensor:
        return self.anchor_pos_w[:, -1]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w_all[:, self.anchor_index].reshape(self.total_dataset_size, -1)

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w_all[:, self.anchor_index].reshape(self.total_dataset_size, -1)

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w_all[:, self.anchor_index].reshape(self.total_dataset_size, -1)

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w_all[:, self.anchor_index].reshape(self.total_dataset_size, -1)

    @property
    def base_lin_vel(self) -> torch.Tensor:
        return quat_apply_inverse(self.anchor_quat_w, self.anchor_lin_vel_w)

    @property
    def base_ang_vel(self) -> torch.Tensor:
        return quat_apply_inverse(self.anchor_quat_w, self.anchor_ang_vel_w)

    def observation_dim_cast(self, name: str) -> int:
        obs_term = getattr(self, name, None)
        if not isinstance(obs_term, torch.Tensor):
            raise NotImplementedError(f"Invalid AMP observation term: {name}")
        return obs_term.shape[-1]

    def init_observation_dims(self) -> None:
        self.observation_dims = [self.observation_dim_cast(term) for term in self.observation_terms]
        self.observation_dim = sum(self.observation_dims)

    def _build_transition_indices(self, traj_lengths: list[int], device: str):
        idx_t = []
        idx_tp1 = []
        offset = 0
        for length in traj_lengths:
            if length < 2:
                offset += length
                continue
            t = torch.arange(offset, offset + length - 1)
            idx_t.append(t)
            idx_tp1.append(t + 1)
            offset += length
        return torch.cat(idx_t).to(device), torch.cat(idx_tp1).to(device)

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, len(self.index_t), (batch_size,), device=self.device)
        return self.index_t[idx], self.index_tp1[idx]

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for _ in range(num_mini_batch):
            t, tp1 = self.sample_batch(mini_batch_size)
            state_features = []
            next_state_features = []
            for term in self.observation_terms:
                data = getattr(self, term)
                state_features.append(data[t])
                next_state_features.append(data[tp1])
            yield torch.cat(state_features, dim=-1), torch.cat(next_state_features, dim=-1)


@configclass
class MotionDatasetCfg:
    """Configuration for loading task-local AMP expert motions."""

    class_type: type[MotionDataset] = MotionDataset
    asset_name: str = "robot"
    motion_files: list[str] = MISSING
    joint_names: list[str] | None = None
    body_names: list[str] = MISSING
    amp_obs_terms: list[str] = MISSING
    anchor_name: str = MISSING
