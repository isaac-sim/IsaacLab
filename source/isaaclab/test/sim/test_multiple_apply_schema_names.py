# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Namespace resolution for multiple-apply USD schemas.

Kitless on purpose: the mapping comes from the USD schema registry, so it needs no simulation.
"""

from typing import ClassVar

import pytest


def test_multiple_apply_property_name_resolves_each_instance():
    """USD reports the namespace prefix, which is not derivable from the schema name.

    ``PhysicsDriveAPI`` writes under ``drive:``, so the prefix has to come from the registry.
    Core UsdPhysics, so this needs no backend plugin.
    """
    from isaaclab.sim.schemas import multiple_apply_property_name

    # the instance is not the last namespace here -- 'physics:' follows it, so the name cannot
    # be composed as prefix:instance:property
    assert multiple_apply_property_name("PhysicsDriveAPI", "angular", "damping") == "drive:angular:physics:damping"
    # templates are cached per schema; the instance must not be
    assert multiple_apply_property_name("PhysicsDriveAPI", "linear", "damping") == "drive:linear:physics:damping"


def test_multiple_apply_property_name_rejects_single_apply_schemas():
    """An empty mapping means 'plugin not loaded'; a single-apply schema is a caller error.

    Without this guard both cases return ``{}``, so a caller silently falls back to a
    hard-coded prefix and writes the attribute under the wrong namespace.
    """
    from isaaclab.sim.schemas import multiple_apply_property_name

    with pytest.raises(ValueError, match="not a multiple-apply schema"):
        multiple_apply_property_name("PhysicsMassAPI", "inst", "mass")


def test_multiple_apply_property_name_raises_for_unregistered_schemas():
    """Backends register their schema plugins before stage creation, so a miss is a hard error.

    Returning an empty mapping instead let callers fall back to a hand-spelled prefix -- the
    failure mode that let tendon properties be authored under the wrong namespace unnoticed.
    """
    from isaaclab.sim.schemas import multiple_apply_property_name

    with pytest.raises(LookupError, match="is not registered"):
        multiple_apply_property_name("NotARealSchema", "inst", "anything")


def test_multiple_apply_templates_are_cached_per_schema():
    """The registry lookup is the costly part and its answer is fixed."""
    from isaaclab.sim.schemas.schemas import _multiple_apply_templates

    _multiple_apply_templates.cache_clear()
    _multiple_apply_templates("PhysicsDriveAPI")
    assert _multiple_apply_templates.cache_info().currsize == 1
    _multiple_apply_templates("PhysicsDriveAPI")
    assert _multiple_apply_templates.cache_info().hits == 1


def test_multiple_apply_property_name_rejects_unknown_properties():
    """A property the schema does not declare is a caller error, not a name to compose."""
    from isaaclab.sim.schemas import multiple_apply_property_name

    with pytest.raises(KeyError):
        multiple_apply_property_name("PhysicsDriveAPI", "angular", "notAProperty")


def test_apply_namespaced_writes_every_instance_of_a_multiple_apply_schema():
    """A fragment declaring ``_usd_multi_apply_schema`` is written across each applied instance.

    Uses ``PhysicsDriveAPI`` (core UsdPhysics, multiple-apply) so the generic path is covered
    without a backend plugin or a running simulation.
    """
    from pxr import Usd, UsdPhysics

    from isaaclab.sim.schemas import apply_namespaced, schemas_cfg
    from isaaclab.utils.configclass import configclass

    @configclass
    class _DriveFragment(schemas_cfg.SchemaFragment):
        _usd_multi_apply_schema: ClassVar[str | None] = "PhysicsDriveAPI"
        damping: float | None = None

    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/World/Joint", "PhysicsRevoluteJoint")
    for instance in ("angular", "linear"):
        UsdPhysics.DriveAPI.Apply(prim, instance)

    assert apply_namespaced(_DriveFragment(damping=7.0), "/World/Joint", stage) is True
    # the instance sits mid-namespace: 'physics:' follows it, so the name comes from USD
    assert prim.GetAttribute("drive:angular:physics:damping").Get() == pytest.approx(7.0)
    assert prim.GetAttribute("drive:linear:physics:damping").Get() == pytest.approx(7.0)


def test_apply_namespaced_reports_a_prim_without_the_declared_schema():
    """No applied instance means nothing was tuned, which the caller distinguishes from success."""
    from pxr import Usd

    from isaaclab.sim.schemas import apply_namespaced, schemas_cfg
    from isaaclab.utils.configclass import configclass

    @configclass
    class _DriveFragment(schemas_cfg.SchemaFragment):
        _usd_multi_apply_schema: ClassVar[str | None] = "PhysicsDriveAPI"
        damping: float | None = None

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/Joint", "PhysicsRevoluteJoint")

    assert apply_namespaced(_DriveFragment(damping=7.0), "/World/Joint", stage) is False
