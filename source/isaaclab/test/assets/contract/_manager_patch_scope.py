# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test-local backend-manager patching for mocked asset factories."""

from collections.abc import Iterator
from contextlib import contextmanager
import inspect


_MISSING = object()
_patch_scopes: list[dict[tuple[object, str], object]] = []


@contextmanager
def contract_manager_patch_scope() -> Iterator[None]:
    """Restore backend manager attributes changed by factories in this scope."""
    patches: dict[tuple[object, str], object] = {}
    _patch_scopes.append(patches)
    try:
        yield
    finally:
        for (owner, name), original in reversed(patches.items()):
            if original is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, original)
        popped = _patch_scopes.pop()
        assert popped is patches


def patch_contract_manager(owner: object, name: str, replacement: object) -> None:
    """Patch a manager binding until the current contract test finishes."""
    if not _patch_scopes:
        raise RuntimeError("contract factories must run inside contract_manager_patch_scope()")
    patches = _patch_scopes[-1]
    key = (owner, name)
    if key not in patches:
        try:
            patches[key] = inspect.getattr_static(owner, name)
        except AttributeError:
            patches[key] = _MISSING
    setattr(owner, name, replacement)
