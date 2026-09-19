# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Version values and semantic version bumps."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Version:
    """A semver-style version string ``X.Y.Z`` (optionally suffixed with ``.devN``).

    Models a version as a value object: immutable, comparable by its text,
    knows how to produce a bumped successor. PEP 440 ``.devN`` suffixes
    are tolerated on the way *in* (stripped before bumping) but never
    written back out — :meth:`bumped` always returns a clean ``X.Y.Z``.

    Construction validates the format up front so that an invalid
    ``--version`` flag from the CLI fails fast instead of silently writing
    a malformed entry to ``CHANGELOG.rst``.
    """

    # ``X.Y.Z`` with an optional PEP 440 ``.devN`` suffix. The suffix is
    # tolerated on the way *in* (e.g. when reading a stale dev version out
    # of an existing version metadata file) but :meth:`bumped` always strips
    # it before producing a successor.
    _SEMVER_RE: ClassVar[re.Pattern[str]] = re.compile(r"^\d+\.\d+\.\d+(\.dev\d+)?$")

    text: str

    def __post_init__(self) -> None:
        if not self._SEMVER_RE.match(self.text):
            raise ValueError(f"Invalid version {self.text!r}; expected X.Y.Z (optionally suffixed with .devN)")

    def bumped(self, tier: str) -> Version:
        """Return a new Version one tier ahead of this one.

        ``tier`` is ``'major'``, ``'minor'``, or ``'patch'``. Major zeros
        the minor and patch components; minor zeros patch. Any ``.devN``
        suffix on the current version is stripped before bumping.
        """
        # __post_init__ guarantees the format, so this split is safe.
        parts = self.text.split(".dev")[0].split(".")
        if tier == "major":
            return Version(f"{int(parts[0]) + 1}.0.0")
        if tier == "minor":
            return Version(f"{parts[0]}.{int(parts[1]) + 1}.0")
        return Version(f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}")

    def __str__(self) -> str:
        return self.text
