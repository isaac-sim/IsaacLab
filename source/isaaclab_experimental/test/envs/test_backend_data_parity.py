# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend data-parity coverage for the warp MDP twins.

The warp MDP twins read simulation state exclusively through ``data.<field>.warp``
views on the factory-dispatched asset and sensor data classes. This test pins
that every field a twin reads exists on **each warp-capable backend** (Newton,
OVPhysX), so a backend gap fails here rather than deep inside a training run.

The field lists mirror the twins' actual reads (``grep -o 'data\\.[a-z_]+\\.warp'``
over :mod:`isaaclab_experimental.envs.mdp` and the task twin packages); extend
them when a twin starts reading a new field.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

# Articulation data fields read by the warp MDP twins. Direct warp envs are
# intentionally excluded: they are declared per backend via ``warp_entry_point``
# and may use backend-specific extensions (e.g. the Newton-native Allegro env
# reads the split ``joint_pos_limits_lower``/``_upper`` fields).
_ARTICULATION_FIELDS = [
    "applied_torque",
    "body_com_pos_b",
    "body_lin_vel_w",
    "body_pos_w",
    "body_quat_w",
    "default_joint_pos",
    "default_joint_vel",
    "default_root_pose",
    "default_root_vel",
    "joint_acc",
    "joint_pos",
    "joint_vel",
    "root_ang_vel_w",
    "root_com_vel_w",
    "root_lin_vel_w",
    "root_link_pose_w",
    "root_pos_w",
    "root_quat_w",
    "root_vel_w",
    "soft_joint_pos_limits",
    "soft_joint_vel_limits",
]

# Contact-sensor data fields read by the warp MDP twins.
_CONTACT_SENSOR_FIELDS = [
    "current_air_time",
    "current_contact_time",
    "last_air_time",
    "net_forces_w_history",
]

# Joint-wrench-sensor data fields read by the warp MDP twins.
_JOINT_WRENCH_FIELDS = [
    "force",
    "torque",
]

# Warp-capable backends: package root -> data-class module paths.
_BACKENDS = {
    "newton": "isaaclab_newton",
    "ovphysx": "isaaclab_ovphysx",
}

_DATA_CLASSES = {
    "articulation": ("{pkg}.assets.articulation.articulation_data", "ArticulationData", _ARTICULATION_FIELDS),
    "contact_sensor": ("{pkg}.sensors.contact_sensor.contact_sensor_data", "ContactSensorData", _CONTACT_SENSOR_FIELDS),
    "joint_wrench": (
        "{pkg}.sensors.joint_wrench.joint_wrench_sensor_data",
        "JointWrenchSensorData",
        _JOINT_WRENCH_FIELDS,
    ),
}


def _exposes(cls: type, name: str) -> bool:
    """Whether ``cls`` declares ``name`` as a descriptor/attribute or annotation."""
    try:
        inspect.getattr_static(cls, name)
        return True
    except AttributeError:
        pass
    return any(name in getattr(klass, "__annotations__", {}) for klass in cls.__mro__)


@pytest.mark.parametrize("backend", sorted(_BACKENDS), ids=sorted(_BACKENDS))
@pytest.mark.parametrize("data_kind", sorted(_DATA_CLASSES), ids=sorted(_DATA_CLASSES))
def test_backend_exposes_twin_data_fields(backend: str, data_kind: str):
    """Every field the warp twins read exists on the backend's data class."""
    module_tpl, class_name, fields = _DATA_CLASSES[data_kind]
    module = importlib.import_module(module_tpl.format(pkg=_BACKENDS[backend]))
    data_class = getattr(module, class_name)

    missing = [field for field in fields if not _exposes(data_class, field)]
    assert not missing, f"{backend} {class_name} lacks fields read by warp twins: {missing}"
