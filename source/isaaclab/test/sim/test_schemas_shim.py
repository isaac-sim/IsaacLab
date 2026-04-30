# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that the forwarding shim in :mod:`isaaclab.sim.schemas` resolves the four PhysX schema
cfgs that were relocated to :mod:`isaaclab_physx.sim.schemas`.

These tests do not require Isaac Sim — only Python import semantics.
"""

import warnings

import pytest

import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as schemas
from isaaclab_physx.sim.schemas import schemas_cfg as physx_cfg

FORWARDED_NAMES = [
    "RigidBodyPropertiesCfg",
    "JointDrivePropertiesCfg",
    "PhysxRigidBodyPropertiesCfg",
    "PhysxJointDrivePropertiesCfg",
]


@pytest.mark.parametrize("name", FORWARDED_NAMES)
def test_schemas_shim_resolves_to_physx_class(name):
    """``isaaclab.sim.schemas.<name>`` resolves to the same class object as the one in
    ``isaaclab_physx.sim.schemas.schemas_cfg``."""
    assert getattr(schemas, name) is getattr(physx_cfg, name)


@pytest.mark.parametrize("name", FORWARDED_NAMES)
def test_sim_namespace_shim_resolves_to_physx_class(name):
    """``isaaclab.sim.<name>`` (i.e. ``sim_utils.<name>``) resolves to the same class object."""
    assert getattr(sim_utils, name) is getattr(physx_cfg, name)


def test_deprecated_alias_emits_deprecation_warning():
    """Instantiating ``RigidBodyPropertiesCfg`` via the shim still emits ``DeprecationWarning``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        schemas.RigidBodyPropertiesCfg()
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_new_class_does_not_emit_deprecation_warning():
    """Instantiating ``PhysxRigidBodyPropertiesCfg`` directly does NOT emit ``DeprecationWarning``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        schemas.PhysxRigidBodyPropertiesCfg()
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_dir_lists_forwarded_names():
    """``dir(isaaclab.sim.schemas)`` includes the forwarded names so IDE autocomplete works."""
    listing = dir(schemas)
    for name in FORWARDED_NAMES:
        assert name in listing
