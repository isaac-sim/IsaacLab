# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Instance selection and attribute spelling of multiple-apply fragments.

Kitless on purpose: ``CollectionAPI`` ships with core USD and lays its attributes out as
``collection:<instance>:<property>``, the same shape as the PhysX tendon schemas.
"""

from typing import ClassVar

import pytest


def _collection_fragment_cls():
    from collections.abc import Callable
    from dataclasses import MISSING

    from isaaclab.sim.schemas import schemas_cfg
    from isaaclab.utils.configclass import configclass

    @configclass
    class _CollectionFragment(schemas_cfg.SchemaFragment):
        _usd_namespace: ClassVar[str | None] = "collection"
        _usd_applied_schema: ClassVar[str | None] = "CollectionAPI"
        func: Callable | str = "isaaclab.sim.schemas:apply_multi_apply"
        instance_names: str | list[str] | None = MISSING
        include_root: bool | None = None

    return _CollectionFragment


def _prim_with_collections():
    from pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/World/Group", "Xform")
    for instance in ("left", "right"):
        Usd.CollectionAPI.Apply(prim, instance)
    return stage, prim


def test_multi_apply_writes_only_the_named_instance():
    """The instance is part of the schema's address, so naming one leaves the other untouched."""
    from isaaclab.sim.schemas import apply_multi_apply

    stage, prim = _prim_with_collections()
    assert apply_multi_apply(
        _collection_fragment_cls()(instance_names="left", include_root=True), "/World/Group", stage
    )
    assert prim.GetAttribute("collection:left:includeRoot").Get() is True
    assert not prim.GetAttribute("collection:right:includeRoot").HasAuthoredValue()


def test_multi_apply_writes_each_listed_instance():
    """A list addresses exactly those instances."""
    from isaaclab.sim.schemas import apply_multi_apply

    stage, prim = _prim_with_collections()
    cfg = _collection_fragment_cls()(instance_names=["left", "right"], include_root=False)
    apply_multi_apply(cfg, "/World/Group", stage)
    for instance in ("left", "right"):
        assert prim.GetAttribute(f"collection:{instance}:includeRoot").Get() is False


def test_multi_apply_broadcasts_only_when_asked():
    """``None`` writes every applied instance, which the caller chooses rather than inherits."""
    from isaaclab.sim.schemas import apply_multi_apply

    stage, prim = _prim_with_collections()
    apply_multi_apply(_collection_fragment_cls()(instance_names=None, include_root=False), "/World/Group", stage)
    for instance in ("left", "right"):
        assert prim.GetAttribute(f"collection:{instance}:includeRoot").Get() is False


def test_multi_apply_requires_an_instance_selection():
    """Omitting the selection is an error, so it can never silently mean broadcast."""
    from isaaclab.sim.schemas import apply_multi_apply

    stage, _ = _prim_with_collections()
    with pytest.raises(ValueError, match="instance_names"):
        apply_multi_apply(_collection_fragment_cls()(include_root=True), "/World/Group", stage)


def test_multi_apply_rejects_an_empty_selection():
    """An empty list is a configuration error rather than a silent no-op."""
    from isaaclab.sim.schemas import apply_multi_apply

    stage, _ = _prim_with_collections()
    with pytest.raises(ValueError, match="instance_names"):
        apply_multi_apply(_collection_fragment_cls()(instance_names=[], include_root=True), "/World/Group", stage)


def test_multi_apply_skips_a_prim_without_the_named_instance():
    """A prim carrying the schema but not the named instance is not addressed.

    The family writer visits every prim under its expression, so the name may belong to a sibling;
    the applier reports "nothing written" and authors nothing rather than raising.
    """
    from isaaclab.sim.schemas import apply_multi_apply

    stage, prim = _prim_with_collections()
    cfg = _collection_fragment_cls()(instance_names="middle", include_root=True)
    assert apply_multi_apply(cfg, "/World/Group", stage) is False
    for instance in ("left", "right"):
        assert not prim.GetAttribute(f"collection:{instance}:includeRoot").HasAuthoredValue()


def test_multi_apply_reports_a_prim_without_the_schema():
    """No applied instance means nothing was tuned, which the caller distinguishes from success."""
    from pxr import Usd

    from isaaclab.sim.schemas import apply_multi_apply

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/Group", "Xform")
    cfg = _collection_fragment_cls()(instance_names=None, include_root=True)
    assert apply_multi_apply(cfg, "/World/Group", stage) is False


def test_multi_apply_raises_on_an_invalid_prim():
    """A missing prim fails loudly, matching the other appliers."""
    from pxr import Usd

    from isaaclab.sim.schemas import apply_multi_apply

    cfg = _collection_fragment_cls()(instance_names=None, include_root=True)
    with pytest.raises(ValueError, match="not valid"):
        apply_multi_apply(cfg, "/Nope", Usd.Stage.CreateInMemory())


def test_multi_apply_fragment_func_resolves_to_the_applier():
    """The fragment's ``func`` string is how the family writers reach the applier."""
    from isaaclab.sim.schemas import apply_multi_apply
    from isaaclab.utils.string import string_to_callable

    assert string_to_callable(_collection_fragment_cls()(instance_names=None).func) is apply_multi_apply
