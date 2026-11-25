# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construction of the model that a :class:`~isaaclab_neural.NeRDCfg` defines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from newton_neural.backbones import BackboneFlash
from newton_neural.contacts import ContactInputBase
from newton_neural.models import FlashNeRD
from newton_neural.systems import SystemBase

from isaaclab.utils import string_to_callable, to_dict

if TYPE_CHECKING:
    from newton_neural.data import TrajectoryDataset
    from newton_neural.models import NeRDBase

    from .nerd_cfg import NeRDCfg


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
