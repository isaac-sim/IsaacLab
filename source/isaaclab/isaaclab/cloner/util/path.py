# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Segment-boundary-safe path/template primitives for the cloner.

A prim path is a sequence of ``/``-delimited segments, not a character string. The
stdlib string operations (:meth:`str.startswith`, :meth:`str.replace`,
:meth:`str.removeprefix`, slicing) work on characters and silently cross segment
boundaries, so this module encodes the segment-boundary semantics once.

Access these through the package, e.g.::

    import isaaclab.cloner as cloner

    cloner.path.split("/World/envs/env_{}/Robot")
    cloner.path.rebase(prim_path, "/World/envs/env_0", "/World/envs/env_5")

Here a *template* is a destination string carrying a single ``"{}"`` clone slot (for
example ``"/World/envs/env_{}/Robot"``), while a *root* is a concrete prefix path.
"""

from __future__ import annotations

import re


def split(template: str) -> tuple[str, str]:
    """Split a clone destination template around its ``"{}"`` clone slot.

    The clone slot represents one concrete environment/instance path segment.

    Args:
        template: Destination path template with ``"{}"`` for the instance id.

    Returns:
        The ``(prefix, suffix)`` strings around the clone slot. Trailing slashes are
        normalized, so an instance-root template (``".../env_{}"``) yields an empty suffix.

    Raises:
        ValueError: If ``template`` does not contain a clone slot.
    """
    template = template.rstrip("/") or "/"
    prefix, slot, suffix = template.partition("{}")
    if slot != "{}":
        raise ValueError(f"Clone destination template must contain '{{}}': {template!r}.")
    return prefix, suffix


def relativize(path_expr: str, template: str) -> str | None:
    """Return the part of ``path_expr`` below a template's env-instance root.

    The template's ``"{}"`` slot matches exactly one path segment (a concrete id like
    ``env_3`` or a wildcard like ``env_.*``). This is the chart inverse: it strips the
    template's instance root off ``path_expr``.

    Args:
        path_expr: Path or path expression on the clone (destination) side.
        template: Destination path template with ``"{}"`` for the instance id.

    Returns:
        The asset-relative suffix (starting with ``/``, or ``""`` when ``path_expr`` is
        exactly the template root), or ``None`` when ``path_expr`` is not under the root.

    Example:
        >>> tmpl = "/World/scenes/{}/Robot"
        >>> relativize("/World/scenes/env_3/Robot/base", tmpl)
        '/base'
        >>> relativize("/World/scenes/.*/Robot/base", tmpl)
        '/base'
        >>> relativize("/World/scenes/env_3/Robot", tmpl)
        ''
        >>> relativize("/World/scenes/env_3/Sensor", tmpl) is None
        True
        >>> relativize("/World/scenes/env_3/RobotArm", tmpl) is None
        True
    """
    pattern = re.compile(r"[^/]+".join(re.escape(part) for part in split(template)))
    match = pattern.match(path_expr)
    if match is None:
        return None
    suffix = path_expr[match.end() :]
    return None if suffix and not suffix.startswith("/") else suffix


def under(path: str, root: str) -> bool:
    """Return whether ``path`` lies within the subtree rooted at ``root``.

    Boundary-correct membership test: unlike :meth:`str.startswith`, it does not match
    across a segment boundary (``".../Robot"`` does not contain ``".../RobotArm"``).
    The root's trailing slash is normalized.

    Args:
        path: Candidate descendant path.
        root: Concrete subtree root.

    Returns:
        ``True`` when ``path`` equals ``root`` or is a descendant of it.
    """
    root = root.rstrip("/") or "/"
    return path == root or path.startswith(root + "/")


def relative_to(path: str, root: str) -> str | None:
    """Strip a concrete ``root`` prefix off ``path`` on a segment boundary.

    Concrete-root counterpart of :func:`relativize`. Unlike slicing or
    :meth:`str.removeprefix`, it returns ``None`` (rather than a mid-segment remainder)
    when ``path`` is not under ``root``.

    Args:
        path: Path to make relative.
        root: Concrete subtree root.

    Returns:
        The suffix below ``root`` (starting with ``/``, or ``""`` when ``path`` equals
        ``root``), or ``None`` when ``path`` is not under ``root``.
    """
    root = root.rstrip("/") or "/"
    if not path.startswith(root):
        return None
    suffix = path[len(root) :]
    return None if suffix and not suffix.startswith("/") else suffix


def rebase(path: str, src_root: str, dst_root: str) -> str:
    """Rebase ``path`` from one concrete root prefix onto another on a segment boundary.

    Unlike :meth:`str.replace`, it swaps only a boundary-aligned prefix and touches only
    the leading occurrence. Equivalent to ``dst_root + relative_to(path, src_root)``.

    Args:
        path: Path to rebase.
        src_root: Concrete source root prefix.
        dst_root: Concrete destination root prefix.

    Returns:
        The rebased path, or ``path`` unchanged when it is not under ``src_root``.
    """
    dst_root = dst_root.rstrip("/") or "/"
    suffix = relative_to(path, src_root)
    if suffix is None:
        return path
    return dst_root + suffix
