# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for versioning."""

from __future__ import annotations

import functools

from packaging.version import Version


def has_kit() -> bool:
    """Check if Kit (Omniverse Kit) is available in the current environment.

    Returns True when running inside an Omniverse Kit application (e.g. Isaac Sim).
    Returns False in kitless mode (e.g. Newton physics backend without Kit).

    Not cached with ``lru_cache`` because this may be called before ``AppLauncher``
    finishes starting Kit, which would permanently lock in a ``False`` result.
    The underlying ``get_app()`` call is cheap once the module is loaded.

    This function deliberately avoids triggering a fresh ``import omni.kit.app``
    when called before Kit has started. If ``omni.kit.app`` is not already present
    in ``sys.modules``, Kit is not running and we return ``False`` immediately without
    performing any import (which would be a forbidden side-effect during cfg-only loading).
    """
    import sys

    mod = sys.modules.get("omni.kit.app")
    if mod is None:
        return False
    try:
        return mod.get_app() is not None
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def get_isaac_sim_version() -> Version:
    """Get the Isaac Sim version as a Version object, cached for performance.

    This function wraps :func:`isaacsim.core.version.get_version()` and caches the result
    to avoid repeated file I/O operations. The underlying Isaac Sim function reads from
    a file each time it's called, which can be slow when called frequently.

    Returns:
        A :class:`packaging.version.Version` object representing the Isaac Sim version.
        This object supports rich comparison operators (<, <=, >, >=, ==, !=).

    Example:
        >>> from isaaclab.utils import get_isaac_sim_version
        >>> from packaging.version import Version
        >>>
        >>> isaac_version = get_isaac_sim_version()
        >>> print(isaac_version)
        5.0.0
        >>>
        >>> # Natural version comparisons
        >>> if isaac_version >= Version("5.0.0"):
        ...     print("Using Isaac Sim 5.0 or later")
        >>>
        >>> # Access components
        >>> print(isaac_version.major, isaac_version.minor, isaac_version.micro)
        5 0 0
    """
    from isaacsim.core.version import get_version

    version_tuple = get_version()
    # version_tuple[2] = major (year), [3] = minor (release), [4] = micro (patch)
    return Version(f"{version_tuple[2]}.{version_tuple[3]}.{version_tuple[4]}")


def compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings and return the comparison result.

    The version strings are expected to be in the format "x.y.z" where x, y,
    and z are integers. The version strings are compared lexicographically.

    .. note::
        This function is provided for backward compatibility. For new code,
        prefer using :class:`packaging.version.Version` objects directly with
        comparison operators (``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``).

    Args:
        v1: The first version string.
        v2: The second version string.

    Returns:
        An integer indicating the comparison result:

        - :attr:`1` if v1 is greater
        - :attr:`-1` if v2 is greater
        - :attr:`0` if v1 and v2 are equal

    Example:
        >>> from isaaclab.utils import compare_versions
        >>> compare_versions("5.0.0", "4.5.0")
        1
        >>> compare_versions("4.5.0", "5.0.0")
        -1
        >>> compare_versions("5.0.0", "5.0.0")
        0
        >>>
        >>> # Better: use Version objects directly
        >>> from packaging.version import Version
        >>> Version("5.0.0") > Version("4.5.0")
        True
    """
    ver1 = Version(v1)
    ver2 = Version(v2)

    if ver1 > ver2:
        return 1
    elif ver1 < ver2:
        return -1
    else:
        return 0
