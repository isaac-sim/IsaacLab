# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`isaaclab.utils.module` — focused on :func:`lazy_imports`."""

import sys
import types

import pytest

from isaaclab.utils.module import _LazyModule


def _make_owner_module(name: str) -> types.ModuleType:
    """Construct a real module object and register it in :data:`sys.modules`.

    The proxy needs ``sys.modules[owner]`` to exist so it can rebind itself
    on first attribute access.  Tests create a throwaway module to act as the
    owner instead of polluting the test module's own globals.
    """
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _drop_module(name: str) -> None:
    sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# _LazyModule directly — exercises the proxy without going through lazy_imports
# (which uses sys._getframe to identify the caller).
# ---------------------------------------------------------------------------


def test_lazy_module_does_not_load_on_construction():
    """Constructing a :class:`_LazyModule` does not import the target."""
    sys.modules.pop("pxr.Sdf", None)
    proxy = _LazyModule("pxr.Sdf", "Sdf", "__test_owner_construct__")
    assert "pxr.Sdf" not in sys.modules
    assert type(proxy).__name__ == "_LazyModule"


def test_lazy_module_repr_before_load():
    """:meth:`__repr__` indicates the proxy state before first access."""
    proxy = _LazyModule("pxr.Sdf", "Sdf", "__test_owner_repr__")
    assert repr(proxy) == "<lazy pxr.Sdf (not yet loaded)>"


def test_lazy_module_first_access_imports_and_replaces():
    """First attribute access imports the real module and rebinds in owner globals."""
    owner_name = "__test_owner_first_access__"
    owner = _make_owner_module(owner_name)
    proxy = _LazyModule("pxr.Sdf", "Sdf", owner_name)
    owner.Sdf = proxy

    # Sanity: before access, no pxr.Sdf in sys.modules, owner has the proxy.
    sys.modules.pop("pxr.Sdf", None)
    assert isinstance(owner.Sdf, _LazyModule)

    # Trigger first attribute access.
    value = proxy.ValueTypeNames

    # Real module loaded, owner rebound, value works.
    assert "pxr.Sdf" in sys.modules
    assert owner.Sdf is sys.modules["pxr.Sdf"]
    assert type(owner.Sdf) is types.ModuleType
    assert value is sys.modules["pxr.Sdf"].ValueTypeNames

    _drop_module(owner_name)


def test_lazy_module_post_load_attribute_is_native():
    """After first access, owner.X.Y is a native module-attr lookup."""
    owner_name = "__test_owner_native_attr__"
    owner = _make_owner_module(owner_name)
    proxy = _LazyModule("pxr.Sdf", "Sdf", owner_name)
    owner.Sdf = proxy

    # Trigger replace.
    _ = proxy.ValueTypeNames

    # Native attribute access — no proxy hop, identity matches the real module.
    import pxr.Sdf

    assert owner.Sdf is pxr.Sdf
    assert owner.Sdf.Path is pxr.Sdf.Path

    _drop_module(owner_name)


# ---------------------------------------------------------------------------
# lazy_imports — exercises the side-effect API and sys._getframe caller lookup
# via exec() so we can use a synthetic owner module without touching the test
# module's own globals.
# ---------------------------------------------------------------------------


def _exec_in_owner(owner_name: str, code: str) -> dict:
    """Execute *code* as if it were the top-level body of *owner_name* module.

    Returns the owner module's globals dict.
    """
    owner = _make_owner_module(owner_name)
    owner.__dict__["__name__"] = owner_name
    exec(code, owner.__dict__)
    return owner.__dict__


def test_lazy_imports_binds_in_caller_globals():
    """:func:`lazy_imports` binds each name in the caller's module globals."""
    g = _exec_in_owner(
        "__test_owner_bind__",
        "from isaaclab.utils.module import lazy_imports\nlazy_imports('pxr', ['Sdf', 'Usd'])\n",
    )
    assert "Sdf" in g
    assert "Usd" in g
    assert isinstance(g["Sdf"], _LazyModule)
    assert isinstance(g["Usd"], _LazyModule)
    _drop_module("__test_owner_bind__")


def test_lazy_imports_does_not_preload_pxr():
    """``lazy_imports`` must not add ``pxr.X`` to ``sys.modules``."""
    # Drop pxr.* first so the test is meaningful.
    for k in list(sys.modules):
        if k.startswith("pxr.UsdLux"):
            sys.modules.pop(k, None)
    sys.modules.pop("pxr.UsdLux", None)

    _exec_in_owner(
        "__test_owner_no_preload__",
        "from isaaclab.utils.module import lazy_imports\nlazy_imports('pxr', ['UsdLux'])\n",
    )
    assert "pxr.UsdLux" not in sys.modules
    _drop_module("__test_owner_no_preload__")


def test_lazy_imports_per_name_isolation():
    """Touching one bound name does not trigger import of the others."""
    # Use submodules that this test session has not yet pulled in.
    sys.modules.pop("pxr.UsdSkel", None)
    sys.modules.pop("pxr.UsdRender", None)

    g = _exec_in_owner(
        "__test_owner_isolation__",
        "from isaaclab.utils.module import lazy_imports\nlazy_imports('pxr', ['UsdSkel', 'UsdRender'])\n",
    )
    assert "pxr.UsdSkel" not in sys.modules
    assert "pxr.UsdRender" not in sys.modules

    # Touch only UsdSkel.
    _ = g["UsdSkel"].Tokens

    assert "pxr.UsdSkel" in sys.modules
    # UsdRender must still be lazy.
    assert "pxr.UsdRender" not in sys.modules
    assert isinstance(g["UsdRender"], _LazyModule)

    _drop_module("__test_owner_isolation__")


def test_lazy_imports_first_access_swaps_in_caller_globals():
    """After first attribute access the binding *is* the real module object."""
    g = _exec_in_owner(
        "__test_owner_swap__",
        "from isaaclab.utils.module import lazy_imports\nlazy_imports('pxr', ['Sdf'])\n",
    )
    assert isinstance(g["Sdf"], _LazyModule)

    # Trigger access.
    _ = g["Sdf"].Path

    import pxr.Sdf

    assert g["Sdf"] is pxr.Sdf
    assert type(g["Sdf"]) is types.ModuleType
    _drop_module("__test_owner_swap__")


def test_lazy_imports_real_semantics_after_load():
    """Once loaded, the proxy/module supports the same calls as a direct import."""
    g = _exec_in_owner(
        "__test_owner_semantics__",
        "from isaaclab.utils.module import lazy_imports\nlazy_imports('pxr', ['Sdf'])\n",
    )
    Sdf = g["Sdf"]  # noqa: N806

    path = Sdf.Path("/World/foo")
    assert str(path) == "/World/foo"
    assert isinstance(path, Sdf.Path)

    _drop_module("__test_owner_semantics__")


def test_lazy_imports_unknown_submodule_raises_on_access():
    """A non-existent submodule raises :class:`ModuleNotFoundError` on first access."""
    g = _exec_in_owner(
        "__test_owner_missing__",
        "from isaaclab.utils.module import lazy_imports\nlazy_imports('pxr', ['NotARealSubmodule'])\n",
    )
    with pytest.raises(ModuleNotFoundError):
        _ = g["NotARealSubmodule"].anything

    _drop_module("__test_owner_missing__")


def test_lazy_imports_empty_names_is_noop():
    """``lazy_imports(package, [])`` binds nothing and never imports the package."""
    sys.modules.pop("pxr_synthetic_unused", None)
    g = _exec_in_owner(
        "__test_owner_empty__",
        "from isaaclab.utils.module import lazy_imports\nlazy_imports('pxr_synthetic_unused', [])\n",
    )
    # No new names bound (only the import + call itself).
    assert "pxr_synthetic_unused" not in sys.modules
    bound = [k for k in g if not k.startswith("__") and k != "lazy_imports"]
    assert bound == []
    _drop_module("__test_owner_empty__")


def test_lazy_imports_multiple_calls_from_same_module():
    """Multiple :func:`lazy_imports` calls in one module compose correctly."""
    g = _exec_in_owner(
        "__test_owner_multi_call__",
        "from isaaclab.utils.module import lazy_imports\nlazy_imports('pxr', ['Sdf'])\nlazy_imports('pxr', ['Usd'])\n",
    )
    assert isinstance(g["Sdf"], _LazyModule)
    assert isinstance(g["Usd"], _LazyModule)
    # Different proxies.
    assert g["Sdf"] is not g["Usd"]
    _drop_module("__test_owner_multi_call__")


def test_lazy_imports_caller_isolation_between_modules():
    """Calls from different modules bind in their respective globals."""
    g1 = _exec_in_owner(
        "__test_owner_caller_a__",
        "from isaaclab.utils.module import lazy_imports\nlazy_imports('pxr', ['Sdf'])\n",
    )
    g2 = _exec_in_owner(
        "__test_owner_caller_b__",
        "from isaaclab.utils.module import lazy_imports\nlazy_imports('pxr', ['Usd'])\n",
    )
    assert "Sdf" in g1 and "Sdf" not in g2
    assert "Usd" in g2 and "Usd" not in g1
    _drop_module("__test_owner_caller_a__")
    _drop_module("__test_owner_caller_b__")
