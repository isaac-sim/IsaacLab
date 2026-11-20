# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics backend that replaces a physics engine with a learned world model."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend, SceneDataFormat
from isaaclab.sim.views import FrameView, UsdFrameView
from isaaclab.utils.configclass import instantiate

from ..world_model import ResetRecord, load_reset_records

if TYPE_CHECKING:
    from ..world_model import WorldModel, WorldModelSpec
    from .world_model_manager_cfg import WorldModelCfg

# There is no physics engine to move prims, so frame prims such as cameras are read from the USD stage. Registering
# the class here lets ``FrameView`` resolve it for this backend without a backend-specific views package.
FrameView.register("neural", UsdFrameView)


class WorldModelManager(PhysicsManager):
    """Own a world model and each environment's joint state.

    The joint state is the reset record's state at reset and the last commanded joint target after that; the model
    never outputs joint state. The model runs once per chunk: every :meth:`step` buffers one joint target per
    environment, and the step that completes a chunk predicts all of the chunk's frames at once. Environments can
    therefore only reset between chunks.

    The articulation shares the joint state and target tensors on the simulation device. Joint targets are buffered
    there too, and are transferred to the existing NumPy model interface only when a chunk is complete.
    """

    backend_name = "neural"
    _cfg: WorldModelCfg

    def __init__(self, cfg: WorldModelCfg):
        super().__init__(cfg)
        self.model: WorldModel | None = None
        self.spec: WorldModelSpec | None = None
        self.num_envs = 0
        self.num_joints = 0
        self.joint_state: torch.Tensor
        """Joint positions [m or rad, depending on joint type], shape ``(N, J)``."""
        self.joint_target: torch.Tensor
        """Joint position targets applied by the next :meth:`step`, shape ``(N, J)``."""
        self.frame_history: np.ndarray
        """RGB frames of the last completed chunk, shape ``(N, T, H, W, 3)``. After a reset, every frame is the start
        frame."""
        self.frame_serial: np.ndarray
        """Per-environment counter that increases whenever the environment's frames change."""
        self.new_frames = False
        """Whether the last :meth:`step` completed a chunk."""
        self._records = load_reset_records(cfg.reset_manifest)
        self._split_records = [r for r in self._records if r.split == cfg.split]
        self._record_images: np.ndarray
        self._env_record: np.ndarray
        self._env_seed: np.ndarray
        self._restart: np.ndarray
        self._rng = np.random.default_rng(cfg.seed)
        self._cursor = 0
        self._draws = 0
        self._targets: torch.Tensor
        self._chunk_start: torch.Tensor
        self._steps_in_chunk = 0
        self._scene_data: SceneDataBackend

    def bind(self, num_envs: int, num_joints: int) -> None:
        """Size the manager for a scene. Called by the articulation when it initializes."""
        self.model = instantiate(self._cfg.model, num_envs)
        self.spec = self.model.spec()
        if not math.isclose(self._sim.cfg.dt, 1.0 / self.spec.fps):
            raise ValueError(f"sim.dt must equal 1 / fps = {1.0 / self.spec.fps} s, got {self._sim.cfg.dt}")
        self.num_envs, self.num_joints = num_envs, num_joints
        self.joint_state = torch.zeros((num_envs, num_joints), dtype=torch.float32, device=self._device)
        self.joint_target = torch.zeros_like(self.joint_state)
        self.frame_history = np.zeros((num_envs, self.spec.chunk_length, *self.spec.image_size, 3), np.uint8)
        self.frame_serial = np.zeros(num_envs, np.int64)
        self._targets = torch.zeros(
            (num_envs, self.spec.chunk_length, num_joints), dtype=torch.float32, device=self._device
        )
        self._chunk_start = torch.zeros_like(self.joint_state)
        self._record_images = _load_images(self._split_records, self.spec.image_size)
        self._env_record = np.zeros(num_envs, dtype=np.int64)
        self._env_seed = np.zeros(num_envs, dtype=np.int64)
        self._restart = np.zeros(num_envs, dtype=bool)

    def initialize(self, sim_context) -> None:
        super().initialize(sim_context)
        # World-model frames are not rigid-body poses, so there is no scene data to publish.
        self._scene_data = _EmptySceneData(self._device)

    def reset(self, soft: bool = False) -> None:
        """Dispatch the lifecycle events. Episodes start with :meth:`reset_envs`, called by a reset event."""
        if not soft:
            self.dispatch_event(PhysicsEvent.MODEL_INIT)
        self.dispatch_event(PhysicsEvent.PHYSICS_READY)

    def reset_envs(self, env_ids: np.ndarray) -> None:
        """Start new episodes in the selected environments from sampled reset records.

        Publish the record's state and start image immediately. The model restarts from the selected records just
        before generating the next chunk, after its joint targets have been collected.

        Raises:
            RuntimeError: If called in the middle of a chunk.
        """
        if self._steps_in_chunk:
            raise RuntimeError(
                f"Environments can only reset on chunk boundaries, but {self._steps_in_chunk} of "
                f"{self.spec.chunk_length} steps of the chunk have been taken."
            )
        indices = self._sample(env_ids)
        records = [self._split_records[index] for index in indices]
        self._draws += 1
        self._env_record[env_ids] = indices
        self._env_seed[env_ids] = (self._cfg.seed + 997 * self._draws + env_ids) % 2**32
        self._restart[env_ids] = True
        self.frame_history[env_ids] = self._record_images[indices, None]
        self.frame_serial[env_ids] += 1
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device)
        state = torch.as_tensor(np.stack([r.joint_state for r in records]), device=self._device)
        self.joint_state[ids] = state
        self.joint_target[ids] = state
        self.new_frames = False

    def set_joint_target(self, target: np.ndarray | torch.Tensor) -> None:
        """Set the joint position targets, shape ``(N, J)``, that the next :meth:`step` applies."""
        self.joint_target.copy_(torch.as_tensor(target, dtype=self.joint_target.dtype, device=self._device))

    def step(self) -> None:
        """Advance one frame; on the last frame of a chunk, predict the chunk with the model."""
        if not self._steps_in_chunk:
            self._chunk_start.copy_(self.joint_state)
        self._targets[:, self._steps_in_chunk] = self.joint_target
        self.joint_state.copy_(self.joint_target)
        self._steps_in_chunk += 1
        self.new_frames = self._steps_in_chunk == self.spec.chunk_length
        if self.new_frames:
            self._restart_model()
            # Keep the existing NumPy model interface; transfer joint inputs only when a chunk is complete.
            self.frame_history = self.model.step(
                self._targets.detach().cpu().numpy(), self._chunk_start.detach().cpu().numpy()
            )
            self.frame_serial += 1
            self._steps_in_chunk = 0
        self._sim_time += self.get_physics_dt()

    def forward(self) -> None:
        pass

    def get_scene_data_backend(self) -> SceneDataBackend:
        return self._scene_data

    def close(self) -> None:
        super().close()
        if self.model is not None:
            self.model.close()

    def _restart_model(self) -> None:
        """Restart only the environments reset since the previous model call, using their latest records."""
        env_ids = np.flatnonzero(self._restart)
        if not len(env_ids):
            return
        records = [self._split_records[index] for index in self._env_record[env_ids]]
        self.model.reset(env_ids, [r.image for r in records], [r.prompt for r in records], self._env_seed[env_ids])
        self._restart[env_ids] = False

    def _sample(self, env_ids: np.ndarray) -> np.ndarray:
        """Pick a reset record per environment. Environments of one reset-record group share a draw, so a group-relative
        algorithm compares rollouts of the same scene. Training draws at random; other splits are visited in order."""
        if self.num_envs % self._cfg.envs_per_reset_record:
            raise ValueError(
                f"envs_per_reset_record={self._cfg.envs_per_reset_record} must divide num_envs={self.num_envs}"
            )
        group_draws: dict[int, int] = {}
        for group in np.unique(env_ids // self._cfg.envs_per_reset_record):
            if self._cfg.split == "train":
                index = self._rng.integers(len(self._split_records))
            else:
                index = self._cursor % len(self._split_records)
                self._cursor += 1
            group_draws[group] = index
        return np.asarray([group_draws[env // self._cfg.envs_per_reset_record] for env in env_ids], dtype=np.int64)


def _load_images(records: list[ResetRecord], image_size: tuple[int, int]) -> np.ndarray:
    """Read reset images as uint8 RGB and check their size against the model's camera output."""
    from PIL import Image

    if not records:
        raise ValueError("The selected reset split has no records.")
    images = []
    for record in records:
        with Image.open(record.image) as image:
            pixels = np.array(image.convert("RGB"))
        if pixels.shape[:2] != tuple(image_size):
            raise ValueError(
                f"The start image {record.image} is {pixels.shape[:2]}, but the world model expects {image_size}."
            )
        images.append(pixels)
    return np.stack(images)


class _EmptySceneData(SceneDataBackend):
    """Scene data without transforms."""

    def __init__(self, device: str):
        self.transforms_timestamp = 0
        self._transforms = SceneDataFormat.Transform()
        self._transforms.transforms = wp.empty(0, dtype=wp.transformf, device=device)

    @property
    def transforms(self) -> SceneDataFormat.Transform:
        return self._transforms

    @property
    def transform_count(self) -> int:
        return 0

    @property
    def transform_paths(self) -> list[str]:
        return []
