# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actuator model config loaders used by deployment tasks."""

from pathlib import Path

_ACTUATOR_MODELS_DIR = Path(__file__).resolve().parent
_IMPLICIT_ACTUATOR_FIELDS = (
    "effort_limit_sim",
    "velocity_limit_sim",
    "stiffness",
    "damping",
    "armature",
    "friction",
    "dynamic_friction",
    "viscous_friction",
)


def load_implicit_actuator_cfg(yaml_file: str, joint_names_expr: list[str]):
    """Load an Isaac Lab ``ImplicitActuatorCfg`` from an actuator-parameter YAML."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.utils.io import load_yaml

    yaml_path = Path(yaml_file)
    if not yaml_path.is_absolute():
        yaml_path = _ACTUATOR_MODELS_DIR / yaml_path

    params = load_yaml(str(yaml_path))
    actuator_kwargs = {field: params[field] for field in _IMPLICIT_ACTUATOR_FIELDS if field in params}
    return ImplicitActuatorCfg(joint_names_expr=joint_names_expr, **actuator_kwargs)
