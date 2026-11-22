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

from ..world_model import WorldModelAttribute, load_reset_records
from ..world_model_frame_buffer import WorldModelFrameBuffer

if TYPE_CHECKING:
    from pathlib import Path

    from ..world_model import ResetRecord, WorldModel, WorldModelSpec
    from .neural_manager_cfg import NeuralCfg

# There is no physics engine to move prims, so frame prims such as cameras are read from the USD stage. Registering
# the class here lets ``FrameView`` resolve it for this backend without a backend-specific views package.
FrameView.register("neural", UsdFrameView)


class NeuralManager(PhysicsManager):
    """Own a world model and each environment's joint state.

    The joint state is the reset record's state at reset and the last commanded joint target after that; the model
    never outputs joint state. The model runs once per chunk: every :meth:`step` buffers one joint target per
    environment, and the step that completes a chunk predicts all of the chunk's frames at once. Environments can
    therefore only reset between chunks.

    Joint state, targets and frames are torch tensors on the simulation device. The articulation reads and writes
    :attr:`joint_state` and :attr:`joint_target` in place, so an environment step copies nothing to the host. A reset
    stays on the device too: :meth:`reset_envs` starts the episode at once from the record's state and start image,
    and the step that ends the next chunk, which waits for the model anyway, restarts the model's environments.
    """

    backend_name = "neural"
    _cfg: NeuralCfg

    def __init__(self, cfg: NeuralCfg):
        super().__init__(cfg)
        self.model: WorldModel | None = None
        self.spec: WorldModelSpec | None = None
        self.num_envs = 0
        self.num_joints = 0
        self.joint_state: torch.Tensor
        """Joint positions [m or rad, depending on joint type], shape ``(N, J)``."""
        self.joint_target: torch.Tensor
        """Joint position targets applied by the next :meth:`step`, shape ``(N, J)``."""
        self._frames: dict[tuple[str, str], WorldModelFrameBuffer] = {}
        self.new_frames = False
        """Whether the last :meth:`step` completed a chunk."""
        self.chunk_boundary: torch.Tensor
        """:attr:`new_frames` for every environment, shape ``(N,)``, ``bool``. Updated in place by :meth:`step`;
        read-only for callers."""
        self._records = load_reset_records(cfg.reset_manifest)
        self._split_records = [r for r in self._records if r.split == cfg.split]
        if not self._split_records:
            raise ValueError(f"The reset manifest {cfg.reset_manifest} has no records in split '{cfg.split}'.")
        self._bound = False
        self._draws = 0
        self._reset_called = False
        self._record_states: torch.Tensor
        self._record_images: dict[tuple[str, str], torch.Tensor] = {}
        self._targets: torch.Tensor
        self._chunk_start: torch.Tensor
        self._inputs: dict[str, torch.Tensor] = {}
        self._steps_in_chunk = 0
        # Per environment, on the device: the record and draw of its last reset, whether the model has yet to restart
        # it, and whether the reset fell inside a chunk.
        self._env_record: torch.Tensor
        self._env_draw: torch.Tensor
        self._restart: torch.Tensor
        self._mid_chunk: torch.Tensor
        self._generator: torch.Generator
        self._cursor: torch.Tensor
        self._scene_data: SceneDataBackend

    def bind(self, num_envs: int, num_joints: int) -> None:
        """Size the manager for a scene and build the model. Called by the articulation when it initializes.

        Raises:
            ValueError: If a second articulation binds in the same initialization, or the scene does not match the
                configuration or the reset records.
        """
        if self._bound:
            raise ValueError("The world-model backend drives one articulation, but the scene has more than one.")
        if num_envs % self._cfg.envs_per_reset_record:
            raise ValueError(f"envs_per_reset_record={self._cfg.envs_per_reset_record} must divide num_envs={num_envs}")
        lengths = {len(record.joint_state) for record in self._split_records}
        if lengths != {num_joints}:
            raise ValueError(
                f"The reset records in {self._cfg.reset_manifest} have {sorted(lengths)} joint positions, but the"
                f" articulation has {num_joints} joints."
            )
        self._bound = True
        # A new initialization, after a full simulation reset, replaces the model of the previous one.
        if self.model is not None:
            self.model.close()
        self.model = instantiate(self._cfg.model_cfg, num_envs, self._device)
        self.spec = self.model.spec()
        if self.spec.max_envs is not None and num_envs > self.spec.max_envs:
            raise ValueError(
                f"The world model serves at most {self.spec.max_envs} environments, but the scene has {num_envs}."
            )
        if not math.isclose(self._sim.cfg.dt, 1.0 / self.spec.fps):
            raise ValueError(f"sim.dt must equal 1 / fps = {1.0 / self.spec.fps} s, got {self._sim.cfg.dt}")
        for names, what in ((self.spec.views, "views"), (self.spec.outputs, "outputs")):
            if not names or len(set(names)) != len(names):
                raise ValueError(f"A world model names one or more distinct {what}, got {names}.")
        provided = [attribute.value for attribute in WorldModelAttribute]
        if not set(self.spec.inputs) <= set(provided):
            raise ValueError(
                f"The neural backend provides the inputs {provided}, but the world model needs"
                f" {list(self.spec.inputs)}."
            )
        self.num_envs, self.num_joints = num_envs, num_joints
        device = self._device
        self.joint_state = torch.zeros((num_envs, num_joints), device=device)
        self.joint_target = torch.zeros_like(self.joint_state)
        self.chunk_boundary = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._record_states = torch.from_numpy(np.stack([r.joint_state for r in self._split_records])).to(device)
        # The start frames set each output's layout, which the model's frames must match.
        self._record_images = {
            (view, output): _load_frames(self._split_records, view, output, self.spec).to(device)
            for view in self.spec.views
            for output in self.spec.outputs
        }
        self._frames = {
            key: WorldModelFrameBuffer(num_envs, self.spec.chunk_length, images.shape[1:], images.dtype, device)
            for key, images in self._record_images.items()
        }
        self._targets = torch.zeros((num_envs, self.spec.chunk_length, num_joints), device=device)
        self._chunk_start = torch.zeros_like(self.joint_state)
        # The buffers behind the model's inputs, written before every step.
        buffers = {
            WorldModelAttribute.JOINT_TARGETS: self._targets,
            WorldModelAttribute.START_JOINT_STATE: self._chunk_start,
        }
        self._inputs = {name: buffers[name] for name in self.spec.inputs}
        self._env_record = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self._env_draw = torch.zeros_like(self._env_record)
        self._restart = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._mid_chunk = torch.zeros_like(self._restart)
        self._generator = torch.Generator(device).manual_seed(self._cfg.seed)
        self._cursor = torch.zeros((), dtype=torch.int64, device=device)
        self._reset_called = False

    def initialize(self, sim_context) -> None:
        super().initialize(sim_context)
        # World-model frames are not rigid-body poses, so there is no scene data to publish.
        self._scene_data = _EmptySceneData(self._device)

    def frames(self, view: str | None = None, output: str = "rgb") -> WorldModelFrameBuffer:
        """Return the predicted frames of one output of one view; a camera reads them through
        :meth:`WorldModelFrameBuffer.reader`.

        Args:
            view: One of the model's :attr:`~isaaclab_neural.world_model.WorldModelSpec.views`, or ``None`` if the
                model predicts one view.
            output: One of the model's :attr:`~isaaclab_neural.world_model.WorldModelSpec.outputs`.

        Raises:
            ValueError: If the model has no such view or output, or ``view`` is ``None`` and the model predicts
                several views.
        """
        views = list(dict.fromkeys(view for view, _ in self._frames))
        if view is None and len(views) == 1:
            view = views[0]
        if view not in views:
            raise ValueError(
                f"The world model predicts the views {views}; a world-model camera selects one of them with"
                f" NeuralRendererCfg.view, got {view!r}."
            )
        if (view, output) not in self._frames:
            raise ValueError(f"The world model predicts the outputs {list(self.spec.outputs)}, not {output!r}.")
        return self._frames[(view, output)]

    def reset(self, soft: bool = False) -> None:
        """Dispatch the lifecycle events. Episodes start with :meth:`reset_envs`, called by the articulation's reset."""
        if not soft:
            # The articulation binds again while it initializes.
            self._bound = False
            self.dispatch_event(PhysicsEvent.MODEL_INIT)
        self.dispatch_event(PhysicsEvent.PHYSICS_READY)

    def reset_envs(self, env_mask: torch.Tensor) -> None:
        """Start new episodes in the selected environments from sampled reset records, without reading the device.

        The joint state, joint target and frames become the record's at once. The model restarts the environments at
        the end of the next chunk, before it predicts that chunk.

        Args:
            env_mask: Environments to reset, shape ``(N,)``, ``bool``, on the simulation device.
        """
        if self._steps_in_chunk:
            # Reported when the chunk ends, where the device is read anyway.
            self._mid_chunk |= env_mask
        records = self._sample(env_mask)
        self._draws += 1
        torch.where(env_mask, records, self._env_record, out=self._env_record)
        self._env_draw.masked_fill_(env_mask, self._draws)
        self._restart |= env_mask
        self._reset_called = True
        state = self._record_states[records]
        torch.where(env_mask[:, None], state, self.joint_state, out=self.joint_state)
        torch.where(env_mask[:, None], state, self.joint_target, out=self.joint_target)
        for key, buffer in self._frames.items():
            buffer.publish(self._record_images[key][records][:, None], env_mask)

    def step(self) -> None:
        """Advance one frame; on the last frame of a chunk, predict the chunk with the model."""
        if not self._steps_in_chunk:
            self._chunk_start.copy_(self.joint_state)
        self._targets[:, self._steps_in_chunk] = self.joint_target
        self.joint_state.copy_(self.joint_target)
        self._steps_in_chunk += 1
        self.new_frames = self._steps_in_chunk == self.spec.chunk_length
        self.chunk_boundary.fill_(self.new_frames)
        if self.new_frames:
            if self._reset_called:
                self._restart_model()
            for name, value in self._inputs.items():
                self.model.write(name, value)
            frames = _flatten(self.model.step())
            if frames.keys() != self._frames.keys():
                raise RuntimeError(
                    f"The world model returned the views and outputs {list(frames)}, not {list(self._frames)}."
                )
            for key, buffer in self._frames.items():
                buffer.publish(frames[key])
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
        """Restart the model's environments that reset since the last chunk.

        The records' prompts are host strings, so this reads the resets on the host, once per chunk with resets.

        Raises:
            RuntimeError: If an environment reset in the middle of the chunk.
        """
        self._reset_called = False
        restart, mid_chunk, records, draws = torch.stack(
            (self._restart, self._mid_chunk, self._env_record, self._env_draw)
        ).tolist()
        self._restart.zero_()
        if any(mid_chunk):
            self._mid_chunk.zero_()
            raise RuntimeError(
                f"Environments {[env for env, late in enumerate(mid_chunk) if late]} reset inside a chunk, but"
                " environments can only reset on chunk boundaries. Make episode_length_s a multiple of the chunk"
                f" duration ({self.spec.chunk_length / self.spec.fps} s), use a decimation that divides"
                f" {self.spec.chunk_length}, and wrap terminations with at_chunk_boundary."
            )
        env_ids = [env for env, flag in enumerate(restart) if flag]
        if not env_ids:
            return
        indices = [records[env] for env in env_ids]
        rows = torch.tensor(indices, device=self._device)
        self.model.reset(
            env_ids,
            _nest({key: images[rows] for key, images in self._record_images.items()}),
            [self._split_records[i].prompt for i in indices],
            [(self._cfg.seed + 997 * draws[env] + env) % 2**32 for env in env_ids],
        )

    def _sample(self, env_mask: torch.Tensor) -> torch.Tensor:
        """Pick a reset record per environment, on the device, and return their indices, shape ``(N,)``.

        Environments of one reset-record group share a draw, so a group-relative algorithm compares rollouts of the
        same scene. Training draws at random; other splits visit the records in order, one per group that resets.
        """
        group_size = self._cfg.envs_per_reset_record
        count = len(self._split_records)
        groups = env_mask.view(-1, group_size).any(dim=1)
        if self._cfg.split == "train":
            draws = torch.randint(count, groups.shape, generator=self._generator, device=self._device)
        else:
            draws = (self._cursor + torch.cumsum(groups, 0) - 1) % count
            self._cursor += groups.sum()
        return draws.repeat_interleave(group_size)


def _flatten(frames: dict[str, dict[str, torch.Tensor]]) -> dict[tuple[str, str], torch.Tensor]:
    """``{view: {output: tensor}}`` as ``{(view, output): tensor}``."""
    return {(view, output): tensor for view, outputs in frames.items() for output, tensor in outputs.items()}


def _nest(frames: dict[tuple[str, str], torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """``{(view, output): tensor}`` as ``{view: {output: tensor}}``."""
    nested: dict[str, dict[str, torch.Tensor]] = {}
    for (view, output), tensor in frames.items():
        nested.setdefault(view, {})[output] = tensor
    return nested


def _load_frames(records: list[ResetRecord], view: str, output: str, spec: WorldModelSpec) -> torch.Tensor:
    """Read the records' start frames of one output of one view, shape ``(R, H, W, C)``.

    RGB start images are image files, read as ``uint8``; other outputs are ``.npy`` arrays.

    Raises:
        ValueError: If a record has no start frame for the view or output, or a frame is not the model's image size.
    """
    from PIL import Image

    frames = []
    for record in records:
        path = _start_frame_path(record, view, output, spec)
        if path.suffix == ".npy":
            frame = np.load(path)
        elif output == "rgb":
            frame = np.asarray(Image.open(path).convert("RGB"))
        else:
            raise ValueError(f"The start frame {path} of the output '{output}' must be a .npy array (H, W, C).")
        if frame.shape[:2] != tuple(spec.image_size):
            raise ValueError(
                f"The start frame {path} is {frame.shape[0]} x {frame.shape[1]}, but the world model predicts"
                f" {spec.image_size[0]} x {spec.image_size[1]} frames."
            )
        frames.append(frame if frame.ndim == 3 else frame[..., None])
    return torch.from_numpy(np.stack(frames))


def _start_frame_path(record: ResetRecord, view: str, output: str, spec: WorldModelSpec) -> Path:
    """The path of a record's start frame of one output of one view; see :attr:`ResetRecord.image`."""
    image = record.image
    if isinstance(image, dict):
        image = image.get(view)
        if image is None:
            raise ValueError(f"The reset record with {record.image} has no start image for the view '{view}'.")
    elif len(spec.views) > 1:
        raise ValueError(
            f"The world model predicts the views {list(spec.views)}, so each reset record needs one start image"
            f" per view, as {{view: path}}; got {record.image}."
        )
    if isinstance(image, dict):
        path = image.get(output)
        if path is None:
            raise ValueError(f"The reset record with {record.image} has no start frame of '{output}' for '{view}'.")
        return path
    if output != "rgb":
        raise ValueError(
            f"The world model predicts the outputs {list(spec.outputs)}, so each reset record needs one start frame"
            f" per view and output, as {{view: {{output: path}}}}; got {record.image}."
        )
    return image


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
