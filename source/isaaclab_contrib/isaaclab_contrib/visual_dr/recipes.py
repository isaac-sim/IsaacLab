# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""YAML overrides shared by visual DR demos and task configurations."""

from dataclasses import fields, replace
from pathlib import Path

import yaml

from .cfg import CosmosBackendCfg, RemoteCosmosBackendCfg, VisualDRCfg


def apply_visual_dr_recipe(cfg: VisualDRCfg, path: str | Path) -> VisualDRCfg:
    """Return a copy with backend and common camera overrides from a safe YAML file.

    Task-owned prompts, semantic classes and camera names are retained. Recipe
    fields cannot select Python implementations; configure those in the task.
    """
    recipe = yaml.safe_load(Path(path).read_text())
    if not isinstance(recipe, dict) or recipe.keys() - {"backend", "camera"}:
        raise ValueError("A visual DR recipe accepts only backend and camera mappings")
    for name, values in recipe.items():
        if not isinstance(values, dict) or {"class_type", "prompts"} & values.keys():
            raise ValueError(f"Invalid {name} recipe overrides")
    return replace(
        cfg,
        backend=replace(cfg.backend, **recipe.get("backend", {})),
        cameras={name: replace(camera, **recipe.get("camera", {})) for name, camera in cfg.cameras.items()},
    )


def remote_cosmos_cfg(cfg: CosmosBackendCfg, devices: tuple[int, ...]) -> RemoteCosmosBackendCfg:
    """Copy every Cosmos setting to a worker configuration."""
    from .remote import RemoteCosmosBackend

    values = {field.name: getattr(cfg, field.name) for field in fields(CosmosBackendCfg)}
    values.update(class_type=RemoteCosmosBackend, max_batch=len(devices))
    return RemoteCosmosBackendCfg(**values, devices=devices)
