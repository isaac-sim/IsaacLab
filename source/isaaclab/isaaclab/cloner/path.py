# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Segment-boundary-safe prim-path primitives for the cloner.

A prim path is a sequence of ``/``-delimited segments, not a character string, and the stdlib
string operations cross those boundaries silently: :meth:`str.startswith` reports that
``".../Robot"`` contains ``".../RobotArm"``. This module encodes the boundary semantics once.

Two kinds of prefix appear in the cloner. A *root* is a concrete prefix path
(``"/World/envs/env_0"``); :func:`relative_to`, :func:`under` and :func:`rebase` work against
one. A *template* carries a single ``"{}"`` clone slot standing for one segment
(``"/World/envs/env_{}/Robot"``); :func:`split`, :func:`match` and :func:`relativize` work
against one. Reach them through the package, as ``cloner.path.rebase(...)``.
"""

from __future__ import annotations

import re
from typing import NamedTuple


class TemplateMatch(NamedTuple):
    """The ``"{}"`` text a template captured (``"3"``, or a wildcard ``".*"``), and the path below it."""

    instance: str
    suffix: str


def split(template: str) -> tuple[str, str]:
    """Split a clone destination template around its ``"{}"`` clone slot.

    The clone slot represents one concrete environment/instance path segment.

    Args:
        template: Destination path template with exactly one ``"{}"`` for the instance id.

    Returns:
        The ``(prefix, suffix)`` strings around the clone slot. A trailing slash is
        insignificant, so an instance-root template (``".../env_{}"``) yields an empty suffix.

    Raises:
        ValueError: If ``template`` does not hold exactly one clone slot. A second slot would
            survive into the suffix and break the later ``str.format`` that fills the first.
    """
    template = template.rstrip("/") or "/"
    slots = template.count("{}")
    if slots != 1:
        raise ValueError(f"Clone destination template must contain exactly one '{{}}', found {slots}: {template!r}.")
    prefix, _, suffix = template.partition("{}")
    return prefix, suffix


def match(path_expr: str, template: str) -> TemplateMatch | None:
    """Match ``path_expr`` against a destination template, capturing the instance slot.

    The ``"{}"`` slot matches one path segment's worth of text, whether a concrete id (``3``)
    or a wildcard (``.*``). Recovering that text is the only way to tell which instance a
    concrete clone path belongs to without slicing the string by hand.

    Args:
        path_expr: Path or path expression on the clone (destination) side.
        template: Destination path template with ``"{}"`` for the instance id.

    Returns:
        A :class:`TemplateMatch` with the captured instance text and the asset-relative
        suffix, or ``None`` when ``path_expr`` is not under the template's instance root.

    Example:
        >>> match("/World/envs/env_3/Robot/base", "/World/envs/env_{}/Robot")
        TemplateMatch(instance='3', suffix='/base')
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

    The suffix half of :func:`match`, for callers that do not need the captured instance.

    Args:
        path_expr: Path or path expression on the clone (destination) side.
        template: Destination path template with ``"{}"`` for the instance id.

    Returns:
        The asset-relative suffix (starting with ``/``, or ``""`` when ``path_expr`` is
        exactly the template root), or ``None`` when ``path_expr`` is not under the root.

    """
    matched = match(path_expr, template)
    return None if matched is None else matched.suffix
