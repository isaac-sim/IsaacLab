# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Segment-boundary-safe prim-path primitives for the cloner.

A prim path is a sequence of ``/``-delimited segments, not a character string. The stdlib
string operations (:meth:`str.startswith`, :meth:`str.replace`, :meth:`str.removeprefix`,
slicing) work on characters and silently cross segment boundaries, so this module encodes
the boundary semantics once.

Two kinds of prefix appear throughout the cloner. A *root* is a concrete prefix path such
as ``"/World/envs/env_0"``. A *template* is a destination path carrying a single ``"{}"``
clone slot, such as ``"/World/envs/env_{}/Robot"``, whose slot stands for exactly one path
segment. :func:`relative_to` strips a root, :func:`relativize` strips a template.

Access these through the package::

    import isaaclab.cloner as cloner

    cloner.path.split("/World/envs/env_{}/Robot")
    cloner.path.rebase(prim_path, "/World/envs/env_0", "/World/envs/env_5")

The operations satisfy, for every ``path``, ``root``, ``dst_root`` and ``template``:

* **P1 (membership)** ``under(path, root)`` holds exactly when ``relative_to(path, root)``
  is not ``None``.
* **P2 (rebase swaps only the root)** when ``under(path, root)``,
  ``rebase(path, root, dst_root) == dst_root.rstrip("/") + relative_to(path, root)``; in
  particular ``rebase(path, root, root) == path``.
* **P3 (no special cases)** every operation accepts ``"/"`` as a root, and a trailing slash
  on a root or template is insignificant.
* **P4 (a match reassembles)** when ``match(path, template)`` returns
  ``(instance, suffix)``, ``path == template.format(instance) + suffix``, and
  ``relativize(path, template)`` is that ``suffix``.
"""

from __future__ import annotations

import re
from typing import NamedTuple


class TemplateMatch(NamedTuple):
    """A destination template matched against one path expression."""

    instance: str
    """The text the template's ``"{}"`` slot captured, e.g. ``"3"`` or the wildcard ``".*"``."""

    suffix: str
    """The part of the path below the template, starting with ``/`` (empty at the template root)."""


def split(template: str) -> tuple[str, str]:
    """Split a clone destination template around its ``"{}"`` clone slot.

    The clone slot represents one concrete environment/instance path segment.

    Args:
        template: Destination path template with exactly one ``"{}"`` for the instance id.

    Returns:
        The ``(prefix, suffix)`` strings around the clone slot. A trailing slash is
        insignificant, so an instance-root template (``".../env_{}"``) yields an empty
        suffix.

    Raises:
        ValueError: If ``template`` does not contain exactly one clone slot. A second slot
            would survive into the suffix and silently break the later ``str.format`` call
            that fills the first.
    """
    template = template.rstrip("/") or "/"
    slots = template.count("{}")
    if slots != 1:
        raise ValueError(f"Clone destination template must contain exactly one '{{}}', found {slots}: {template!r}.")
    prefix, _, suffix = template.partition("{}")
    return prefix, suffix


def match(path_expr: str, template: str) -> TemplateMatch | None:
    """Match ``path_expr`` against a destination template, capturing the instance slot.

    The template's ``"{}"`` slot matches exactly one path segment's worth of text, whether
    a concrete id (``3``) or a wildcard (``.*``). This is the primitive behind
    :func:`relativize`, and the only way to recover *which* instance a concrete clone path
    belongs to without slicing the string by hand.

    Args:
        path_expr: Path or path expression on the clone (destination) side.
        template: Destination path template with ``"{}"`` for the instance id.

    Returns:
        A :class:`TemplateMatch` with the captured instance text and the asset-relative
        suffix, or ``None`` when ``path_expr`` is not under the template's instance root.

    Example:
        >>> match("/World/envs/env_3/Robot/base", "/World/envs/env_{}/Robot")
        TemplateMatch(instance='3', suffix='/base')
        >>> match("/World/envs/env_.*/Robot", "/World/envs/env_{}/Robot")
        TemplateMatch(instance='.*', suffix='')
        >>> match("/World/envs/env_3/Sensor", "/World/envs/env_{}/Robot") is None
        True
    """
    prefix, template_suffix = split(template)
    pattern = re.compile(re.escape(prefix) + r"([^/]+)" + re.escape(template_suffix))
    matched = pattern.match(path_expr)
    if matched is None:
        return None
    suffix = path_expr[matched.end() :]
    if suffix and not suffix.startswith("/"):
        return None
    return TemplateMatch(matched.group(1), suffix)


def relative_to(path: str, root: str) -> str | None:
    """Strip a concrete ``root`` prefix off ``path`` on a segment boundary.

    Unlike slicing or :meth:`str.removeprefix`, this returns ``None`` rather than a
    mid-segment remainder when ``path`` is not under ``root``.

    Args:
        path: Path to make relative.
        root: Concrete subtree root. A trailing slash is insignificant, and ``"/"`` is the
            root of every path.

    Returns:
        The suffix below ``root`` (starting with ``/``, or ``""`` when ``path`` equals
        ``root``), or ``None`` when ``path`` is not under ``root``.

    Example:
        >>> relative_to("/World/envs/env_0/Robot", "/World/envs/env_0")
        '/Robot'
        >>> relative_to("/World/envs/env_0", "/World/envs/env_0")
        ''
        >>> relative_to("/World/envs/env_0X", "/World/envs/env_0") is None
        True
    """
    root = root.rstrip("/") or "/"
    if path == root:
        return ""
    # "/" prefixes every path but contributes no segment of its own.
    prefix = "" if root == "/" else root
    if not path.startswith(prefix):
        return None
    suffix = path[len(prefix) :]
    return suffix if suffix.startswith("/") else None


def under(path: str, root: str) -> bool:
    """Return whether ``path`` lies within the subtree rooted at ``root``.

    Boundary-correct membership test: unlike :meth:`str.startswith`, it does not match
    across a segment boundary (``".../Robot"`` does not contain ``".../RobotArm"``).

    Args:
        path: Candidate descendant path.
        root: Concrete subtree root.

    Returns:
        ``True`` when ``path`` equals ``root`` or is a descendant of it.
    """
    return relative_to(path, root) is not None


def rebase(path: str, src_root: str, dst_root: str) -> str:
    """Rebase ``path`` from one concrete root prefix onto another on a segment boundary.

    Unlike :meth:`str.replace`, this swaps only a boundary-aligned prefix and touches only
    the leading occurrence.

    Args:
        path: Path to rebase.
        src_root: Concrete source root prefix.
        dst_root: Concrete destination root prefix.

    Returns:
        The rebased path, or ``path`` unchanged when it is not under ``src_root``.
    """
    suffix = relative_to(path, src_root)
    if suffix is None:
        return path
    return (dst_root.rstrip("/") + suffix) or "/"


def relativize(path_expr: str, template: str) -> str | None:
    """Return the part of ``path_expr`` below a template's instance root.

    Template counterpart of :func:`relative_to`, and the suffix projection of
    :func:`match` for callers that do not need the captured instance.

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
    matched = match(path_expr, template)
    return None if matched is None else matched.suffix
