# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stubs for the parts of an articulation that a world model does not have."""

from __future__ import annotations

import abc
import inspect


def _raiser(owner: str, name: str):
    def unsupported(self, *args, **kwargs):
        raise NotImplementedError(
            f"{owner}.{name} is not available with the world-model backend: it has joint state only, "
            "no bodies, root, forces or physical properties."
        )

    unsupported.__name__ = name
    return unsupported


def raise_for_unsupported(cls: type, base: type) -> type:
    """Give every abstract member of ``base`` that ``cls`` leaves unimplemented a version that raises.

    The base classes declare their whole public surface abstractly, and Python refuses to instantiate a class
    that leaves any of it open. A world model has only joint state, so the rest raises a clear error when used.
    Properties stay properties, so reading them raises instead of returning a placeholder.
    """
    for name in sorted(base.__abstractmethods__):
        if name in cls.__dict__:
            continue
        raiser = _raiser(cls.__name__, name)
        member = inspect.getattr_static(base, name)
        setattr(cls, name, property(raiser) if isinstance(member, property) else raiser)
    abc.update_abstractmethods(cls)
    return cls
