# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construction of the model that a :class:`~isaaclab_neural.NeRDCfg` defines, and how it runs by default."""

from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING

import torch
from newton_neural.backbones import BackboneFlash
from newton_neural.contacts import ContactInputBase
from newton_neural.models import FlashNeRD
from newton_neural.systems import SystemBase

from isaaclab.utils import string_to_callable, to_dict

if TYPE_CHECKING:
    from newton_neural.data import TrajectoryDataset
    from newton_neural.models import NeRDBase

    from .nerd_cfg import NeRDCfg

logger = logging.getLogger(__name__)


def default_compile(device: torch.device | str) -> bool:
    """Return whether a model on ``device`` runs through ``torch.compile`` when no setting says otherwise.

    ``torch.compile`` generates CUDA code with Triton, which PyTorch installs on Linux only. Where Triton is
    missing, such as on Windows, the model runs without ``torch.compile``, which is slower and otherwise
    equivalent.
    """
    if torch.device(device).type != "cuda":
        return False
    if importlib.util.find_spec("triton") is not None:
        return True
    logger.warning(
        "Triton is not installed, so torch.compile cannot generate CUDA code and the neural dynamics model runs"
        " without it. This is slower and otherwise equivalent. PyTorch installs Triton on Linux only."
    )
    return False


def build_model(dataset: TrajectoryDataset, cfg: NeRDCfg) -> NeRDBase:
    """Build the model that ``cfg`` defines on the robot that ``dataset`` describes.

    Raises:
        ValueError: If the dataset describes no robot, or if it was recorded for other inputs or labels than
            ``cfg`` asks for.
    """
    recorded = dataset.generation.get("isaaclab")
    if recorded is None:
        raise ValueError(f"{dataset.path} does not describe its robot; record it with scripts/neural/record.py")
    inputs = recorded["model_inputs"]
    changed = [name for name, value in inputs.items() if getattr(cfg, name) != value]
    if changed:
        recorded_values = ", ".join(f"model.{name}={inputs[name]!r}" for name in changed)
        raise ValueError(
            f"the dataset was recorded with {recorded_values}, and the model definition sets other values. These "
            "settings select the recorded signals. Keep the recorded values, or record the dataset again"
        )
    contact_input = recorded["contact_input"]
    nerd = FlashNeRD(
        SystemBase.create(**recorded["system"]),
        backbone=BackboneFlash(spans=cfg.backbone_spans),
        contact_input=None if contact_input is None else ContactInputBase.create(**contact_input),
        prediction=string_to_callable(cfg.prediction)(),
    )
    nerd.metadata["isaaclab"] = {"physical_model": recorded["physical_model"], "model_cfg": to_dict(cfg)}
    return nerd
