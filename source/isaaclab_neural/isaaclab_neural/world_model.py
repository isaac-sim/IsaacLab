# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interface implemented by every learned world model."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from isaaclab.utils import configclass


@configclass
class ModelCfg:
    """Base configuration of a world model. Concrete models subclass it and set :attr:`class_type`."""

    class_type: type[WorldModel] | str = "{DIR}.world_model:WorldModel"
    """The :class:`WorldModel` implementation, built as ``class_type(cfg, num_envs, device)``."""


@dataclass(frozen=True)
class WorldModelSpec:
    """Static properties a model declares to the manager."""

    chunk_length: int
    """Frames predicted by one :meth:`WorldModel.step`, and joint targets it consumes."""

    fps: int
    """Frame rate of the predicted frames [Hz]."""

    image_size: tuple[int, int]
    """Height and width of each predicted RGB frame [pixels]."""

    max_envs: int | None = None
    """Largest number of environments one model instance serves, or ``None`` for no limit."""


@dataclass(frozen=True)
class ResetRecord:
    """Initial condition of an episode: a start image, the joint state at that image and a task prompt."""

    image: Path
    """RGB start image."""

    joint_state: np.ndarray
    """Joint positions [m or rad, depending on joint type] at the start image, shape ``(num_joints,)``."""

    prompt: str
    """Natural-language task description."""

    split: str = "train"
    """Dataset split of the record, such as ``"train"`` or ``"validation"``."""


def load_reset_records(manifest: str | Path) -> list[ResetRecord]:
    """Load reset records from a JSON list of ``{"image", "state", "prompt", "split"}`` objects.

    Relative image paths resolve against the manifest's directory.
    """
    manifest = Path(manifest)
    return [
        ResetRecord(
            manifest.parent / row["image"],
            np.asarray(row["state"], np.float32),
            row["prompt"],
            row.get("split", "train"),
        )
        for row in json.loads(manifest.read_text())
    ]


class WorldModel(ABC):
    """Interface of a learned model that predicts camera frames from joint targets.

    The model owns only visual dynamics; it never outputs joint state. An implementation is built as
    ``class_type(cfg, num_envs, device)`` and keeps what it needs of those. Construction is cheap and weights load on
    the first call. Joint targets and frames are torch tensors on the simulation device, so a model that runs on the
    GPU keeps them there.
    """

    @abstractmethod
    def spec(self) -> WorldModelSpec:
        """Return the model's static properties."""

    @abstractmethod
    def reset(self, env_ids: list[int], images: torch.Tensor, prompts: list[str], seeds: list[int]) -> None:
        """Restart the selected environments from their start images.

        The manager calls it right before :meth:`step`, for the environments that reset since the last chunk. The
        start image is the environment's frame until that chunk's frames arrive.

        Args:
            env_ids: Environments to restart.
            images: Start RGB image of each selected environment, shape ``(K, H, W, 3)``, ``uint8``, on the
                simulation device.
            prompts: Task description of each selected environment.
            seeds: Sampling seed of each selected environment.
        """

    @abstractmethod
    def step(self, joint_targets: torch.Tensor, start_joint_state: torch.Tensor) -> torch.Tensor:
        """Predict the next chunk of frames for all environments.

        Args:
            joint_targets: Joint position targets [m or rad], shape ``(N, chunk_length, num_joints)``.
            start_joint_state: Joint positions before the chunk, shape ``(N, num_joints)``.

        Returns:
            RGB frames, shape ``(N, chunk_length, H, W, 3)``, ``uint8``, on the simulation device.
        """

    @abstractmethod
    def close(self) -> None:
        """Release model resources."""
