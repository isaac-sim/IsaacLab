# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Instance selection and attribute spelling of multiple-apply fragments.

Kitless on purpose: ``PhysicsDriveAPI`` ships with core UsdPhysics, so no backend plugin is needed.
Its attribute layout also puts the instance mid-name (``drive:<inst>:physics:<prop>``), which is
why the fragment carries a template rather than a namespace.
"""

from typing import ClassVar

import pytest


def _drive_fragment_cls():
    from collections.abc import Callable

    from isaaclab.sim.schemas import schemas_cfg
    from isaaclab.utils.configclass import configclass

    @configclass
    class _DriveFragment(schemas_cfg.MultiApplyFragment):
        _usd_multi_apply_schema: ClassVar[str] = "PhysicsDriveAPI"
        _usd_multi_apply_template: ClassVar[str] = "drive:{instance}:physics:{prop}"
        func: Callable | str = "isaaclab.sim.schemas:apply_multi_apply"
        damping: float | None = None

    return _DriveFragment


def _joint_with_drives():
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/World/Joint", "PhysicsRevoluteJoint")
    for instance in ("angular", "linear"):
        UsdPhysics.DriveAPI.Apply(prim, instance)
    return stage, prim


def test_multi_apply_writes_only_the_named_instance():
    """The instance is part of the schema's address, so naming one leaves the other untouched."""
    from isaaclab.sim.schemas import apply_multi_apply

    stage, prim = _joint_with_drives()
    assert apply_multi_apply(_drive_fragment_cls()(instance_names="angular", damping=7.0), "/World/Joint", stage)
    assert prim.GetAttribute("drive:angular:physics:damping").Get() == pytest.approx(7.0)
    assert not prim.GetAttribute("drive:linear:physics:damping").HasAuthoredValue()


def test_multi_apply_writes_each_listed_instance():
    """A list addresses exactly those instances."""
    from isaaclab.sim.schemas import apply_multi_apply

    stage, prim = _joint_with_drives()
    apply_multi_apply(_drive_fragment_cls()(instance_names=["angular", "linear"], damping=3.0), "/World/Joint", stage)
    for instance in ("angular", "linear"):
        assert prim.GetAttribute(f"drive:{instance}:physics:damping").Get() == pytest.approx(3.0)


def test_multi_apply_broadcasts_only_when_asked():
    """``None`` writes every applied instance, which the caller chooses rather than inherits."""
    from isaaclab.sim.schemas import apply_multi_apply

    stage, prim = _joint_with_drives()
    apply_multi_apply(_drive_fragment_cls()(instance_names=None, damping=5.0), "/World/Joint", stage)
    for instance in ("angular", "linear"):
        assert prim.GetAttribute(f"drive:{instance}:physics:damping").Get() == pytest.approx(5.0)


def test_multi_apply_requires_an_instance_selection():
    """Omitting the selection is an error, so it can never silently mean broadcast."""
    from isaaclab.sim.schemas import apply_multi_apply

    stage, _ = _joint_with_drives()
    with pytest.raises(ValueError, match="instance_names"):
        apply_multi_apply(_drive_fragment_cls()(damping=7.0), "/World/Joint", stage)


def test_multi_apply_rejects_an_empty_selection():
    """An empty list is a configuration error rather than a silent no-op."""
    from isaaclab.sim.schemas import apply_multi_apply

    stage, _ = _joint_with_drives()
    with pytest.raises(ValueError, match="instance_names"):
        apply_multi_apply(_drive_fragment_cls()(instance_names=[], damping=7.0), "/World/Joint", stage)


def test_multi_apply_skips_a_prim_without_the_named_instance():
    """A prim carrying the schema but not the named instance is not addressed.

    The family writer visits every prim under its expression, so the name may belong to a sibling;
    the applier reports "nothing written" and authors nothing rather than raising.
    """
    from isaaclab.sim.schemas import apply_multi_apply

    stage, prim = _joint_with_drives()
    assert (
        apply_multi_apply(_drive_fragment_cls()(instance_names="rotary", damping=7.0), "/World/Joint", stage) is False
    )
    for instance in ("angular", "linear"):
        assert not prim.GetAttribute(f"drive:{instance}:physics:damping").HasAuthoredValue()


def test_multi_apply_reports_a_prim_without_the_schema():
    """No applied instance means nothing was tuned, which the caller distinguishes from success."""
    from pxr import Usd

    from isaaclab.sim.schemas import apply_multi_apply

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/Joint", "PhysicsRevoluteJoint")
    assert apply_multi_apply(_drive_fragment_cls()(instance_names=None, damping=7.0), "/World/Joint", stage) is False


def test_multi_apply_raises_on_an_invalid_prim():
    """A missing prim fails loudly, matching the other appliers."""
    from pxr import Usd

    from isaaclab.sim.schemas import apply_multi_apply

    with pytest.raises(ValueError, match="not valid"):
        apply_multi_apply(_drive_fragment_cls()(instance_names=None, damping=7.0), "/Nope", Usd.Stage.CreateInMemory())


def test_multi_apply_fragment_func_resolves_to_the_applier():
    """The fragment's ``func`` string is how the family writers reach the applier."""
    from isaaclab.sim.schemas import apply_multi_apply
    from isaaclab.utils.string import string_to_callable

    assert string_to_callable(_drive_fragment_cls()(instance_names=None).func) is apply_multi_apply
