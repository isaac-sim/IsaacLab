# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interface implemented by every learned world model."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import torch

from isaaclab.utils import configclass


@configclass
class ModelCfg:
    """Base configuration of a world model. Concrete models subclass it and set :attr:`class_type`."""

    class_type: type[WorldModel] | str = "{DIR}.world_model:WorldModel"
    """The :class:`WorldModel` implementation, built as ``class_type(cfg, num_envs, device)``."""


class WorldModelAttribute(StrEnum):
    """Inputs the neural backend can write to a world model before each :meth:`WorldModel.step`."""

    JOINT_TARGETS = "joint_targets"
    """Joint position targets of the chunk [m or rad], shape ``(N, chunk_length, num_joints)``."""

    START_JOINT_STATE = "start_joint_state"
    """Joint positions before the chunk [m or rad], shape ``(N, num_joints)``."""


@dataclass(frozen=True)
class WorldModelSpec:
    """Static properties a model declares to the manager."""

    chunk_length: int
    """Frames predicted by one :meth:`WorldModel.step`, and joint targets it consumes."""

    fps: int
    """Frame rate of the predicted frames [Hz]."""

    image_size: tuple[int, int]
    """Height and width of each predicted frame [pixels]."""

    max_envs: int | None = None
    """Largest number of environments one model instance serves, or ``None`` for no limit."""

    views: tuple[str, ...] = ("camera",)
    """Names of the camera views the model predicts. Each world-model camera shows one of them, selected with
    :attr:`~isaaclab_neural.renderers.NeuralRendererCfg.view`."""

    inputs: tuple[str, ...] = (WorldModelAttribute.JOINT_TARGETS, WorldModelAttribute.START_JOINT_STATE)
    """Inputs the model needs, from :class:`WorldModelAttribute`. The manager writes each with
    :meth:`WorldModel.write` before every step, and rejects a model that needs an input the backend does not provide."""

    outputs: tuple[str, ...] = ("rgb",)
    """Camera outputs the model predicts for each view, named as camera data types
    (:class:`~isaaclab.renderers.RenderBufferKind`), such as ``"rgb"`` or ``"depth"``."""


@dataclass(frozen=True)
class ResetRecord:
    """Initial condition of an episode: a start image, the joint state at that image and a task prompt."""

    image: Path | dict[str, Path] | dict[str, dict[str, Path]]
    """Start image: one path for a model of one view that predicts RGB, one path per view, or one path per view and
    output. RGB start images are image files; other outputs are ``.npy`` arrays of shape ``(H, W, C)``."""

    joint_state: np.ndarray
    """Joint positions [m or rad, depending on joint type] at the start image, shape ``(num_joints,)``."""

    prompt: str
    """Natural-language task description."""

    split: str = "train"
    """Dataset split of the record, such as ``"train"`` or ``"validation"``."""


def load_reset_records(manifest: str | Path) -> list[ResetRecord]:
    """Load reset records from a JSON list of ``{"image", "state", "prompt", "split"}`` objects.

    ``image`` is a path, an object mapping each view to a path, or an object mapping each view to an object that maps
    each output to a path; see :attr:`ResetRecord.image`. Relative paths resolve against the manifest's directory.
    """
    manifest = Path(manifest)

    def resolve(image):
        return (
            {key: resolve(value) for key, value in image.items()}
            if isinstance(image, dict)
            else manifest.parent / image
        )

    records = []
    for row in json.loads(manifest.read_text()):
        image = resolve(row["image"])
        records.append(
            ResetRecord(image, np.asarray(row["state"], np.float32), row["prompt"], row.get("split", "train"))
        )
    return records


class WorldModel(ABC):
    """Interface of a learned model that predicts camera outputs from an articulation's joint targets.

    It covers action-conditioned models whose predictions are camera outputs, such as video world models; each
    implementation maps the joint targets to its own action space. Models that predict state instead, and policies,
    are not world models of this kind. The model owns only visual dynamics; it never outputs joint state.

    An implementation is built as ``class_type(cfg, num_envs, device)`` and keeps what it needs of those.
    Construction is cheap and weights load on the first call. Joint targets and frames are torch tensors on the
    simulation device, so a model that runs on the GPU keeps them there.
    """

    @abstractmethod
    def spec(self) -> WorldModelSpec:
        """Return the model's static properties."""

    @abstractmethod
    def reset(
        self, env_ids: list[int], images: dict[str, dict[str, torch.Tensor]], prompts: list[str], seeds: list[int]
    ) -> None:
        """Restart the selected environments from their start images.

        The manager calls it right before :meth:`step`, for the environments that reset since the last chunk. The
        start images are the environment's frames until that chunk's frames arrive.

        Args:
            env_ids: Environments to restart.
            images: For each of the model's views and outputs, the start frame of each selected environment, shape
                ``(K, H, W, C)``, on the simulation device; RGB is ``uint8`` with ``C = 3``.
            prompts: Task description of each selected environment.
            seeds: Sampling seed of each selected environment.
        """

    @abstractmethod
    def write(self, name: str, value: torch.Tensor) -> None:
        """Set one of the model's :attr:`~WorldModelSpec.inputs` for the next :meth:`step`.

        Args:
            name: A :class:`WorldModelAttribute` the model lists in its spec.
            value: The input of every environment, on the simulation device. It is the manager's buffer, valid until
                the step returns.
        """

    @abstractmethod
    def step(self) -> dict[str, dict[str, torch.Tensor]]:
        """Predict the next chunk of frames for all environments from the inputs written since the last step.

        Returns:
            For each of the model's views and outputs, frames of shape ``(N, chunk_length, H, W, C)`` with the layout
            of the start frames, on the simulation device.
        """

    @abstractmethod
    def close(self) -> None:
        """Release model resources."""
