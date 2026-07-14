# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pinned Isaac for Healthcare assets and physical constants for needle pass.

The assets are referenced directly from the public Isaac for Healthcare 0.6.0
catalogue under the Apache License 2.0.  No USD or mesh content is redistributed
with Isaac Lab.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cache
from urllib.error import URLError
from urllib.request import Request, urlopen

I4H_CATALOGUE_RELEASE = "0.6.0"
"""Isaac for Healthcare asset catalogue release."""

I4H_CATALOGUE_COMMIT = "bee7e9314bb8f1c78f7e178a7840d708eda9ffb1"
"""Commit referenced by the ``v0.6.0`` catalogue tag."""

I4H_CATALOGUE_LICENCE = "Apache-2.0"
"""SPDX identifier of the pinned catalogue's licence."""

I4H_CATALOGUE_LICENCE_URL = "https://github.com/isaac-for-healthcare/i4h-asset-catalog/blob/v0.6.0/LICENSE"
"""Licence text for the pinned catalogue tag."""

I4H_CONTENT_REVISION = "c189487"
"""Immutable content revision embedded in the public asset URLs."""

I4H_CONTENT_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    f"Assets/Isaac/Healthcare/{I4H_CATALOGUE_RELEASE}/{I4H_CONTENT_REVISION}"
)


@dataclass(frozen=True, slots=True)
class I4HAssetPin:
    """Remote I4H asset with a digest for the explicit online preflight."""

    key: str
    sha256: str

    @property
    def url(self) -> str:
        """Return the revisioned public catalogue URL."""

        return f"{I4H_CONTENT_ROOT}/{self.key}"


@cache
def verify_remote_asset_sha256(url: str, expected_sha256: str) -> None:
    """Fail closed unless a remote USD response matches its declared SHA-256.

    The preflight is cached per process, so cloned environments do not repeat
    the request.  It verifies response bytes rather than trusting the revision
    text embedded in the remote path.
    """

    expected_sha256 = expected_sha256.lower()
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise ValueError("expected asset digest must be a lowercase SHA-256 hex string")
    digest = hashlib.sha256()
    try:
        with urlopen(Request(url, headers={"User-Agent": "IsaacLab-dVRK-asset-preflight"}), timeout=30) as response:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                digest.update(chunk)
    except URLError as error:
        raise RuntimeError(f"unable to preflight pinned dVRK asset {url!r}") from error
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"pinned dVRK asset digest mismatch for {url!r}: expected {expected_sha256}, got {actual_sha256}"
        )


NEEDLE_ASSET = I4HAssetPin(
    key="Props/SutureNeedle/needle_sdf.usd",
    sha256="2b317a61f93631a7192e7ed2839ef20f7a75c05aa5f84a3905696134a64f36d7",
)
"""Preferred dynamic SDF needle used by I4H surgical tasks."""

NEEDLE_CONVEX_FALLBACK_ASSET = I4HAssetPin(
    key="Props/SutureNeedle/needle.usd",
    sha256="dd6910d4e3b8cede984e66d1772c2a16aa21c5cd2f41c5e6f7c4dd8f6d754620",
)
"""Pinned fallback; the task does not select it without measured simulator evidence."""

SUTURE_PAD_ASSET = I4HAssetPin(
    key="Props/SuturePad/suture_pad.usd",
    sha256="1c6e4624097fbf8ffc49131539e9eec72d96c5cf68916fbe04eefff1e9522a51",
)

# The 0.4 scale is the scale used by the pinned I4H surgical handover scene.
NEEDLE_SCALE = (0.4, 0.4, 0.4)

# The aligned source bounds include the nested +0.05 m X transform authored
# beneath the rigid-body root. They were computed from the exact pinned USD's
# extents and provide a reviewable size oracle independent of a live episode.
NEEDLE_SOURCE_AABB_MIN_M = (-0.002297283822972465, -0.049757134369845066, -0.001107099094351224)
NEEDLE_SOURCE_AABB_MAX_M = (0.04786571530379667, 0.0491908637664369, 0.0030219009137550075)
NEEDLE_BODY_LOCAL_AABB_MIN_M = tuple(
    coordinate * scale for coordinate, scale in zip(NEEDLE_SOURCE_AABB_MIN_M, NEEDLE_SCALE, strict=True)
)
NEEDLE_BODY_LOCAL_AABB_MAX_M = tuple(
    coordinate * scale for coordinate, scale in zip(NEEDLE_SOURCE_AABB_MAX_M, NEEDLE_SCALE, strict=True)
)
NEEDLE_BODY_LOCAL_EXTENT_M = tuple(
    maximum - minimum
    for minimum, maximum in zip(NEEDLE_BODY_LOCAL_AABB_MIN_M, NEEDLE_BODY_LOCAL_AABB_MAX_M, strict=True)
)

# The source USD does not author these values.  They are immutable task inputs,
# declared before the analytical grasp calculation and never adapted from a
# live episode.  The source volume was computed from the exact pinned/hash-
# checked SDF USD mesh at /Needle/Needle/Needle: 934 consistently oriented
# triangles, the authored parent +0.05 m X transform, and the signed tetrahedron
# (divergence-theorem) sum.  Uniform scale changes volume by the scale product.
# A 316L surgical-steel density of 8000 kg/m^3 is an explicit task reference
# assumption; it is not authored I4H metadata or a measured task value.
NEEDLE_SOURCE_VOLUME_M3 = 2.085934204311373e-6
NEEDLE_REFERENCE_DENSITY_KG_M3 = 8000.0
NEEDLE_MASS_KG = (
    NEEDLE_SOURCE_VOLUME_M3 * NEEDLE_SCALE[0] * NEEDLE_SCALE[1] * NEEDLE_SCALE[2] * (NEEDLE_REFERENCE_DENSITY_KG_M3)
)

# The same pinned mesh integration gives the source centre of mass below.  It
# is scaled with the asset, but is not tuned or updated from a running episode.
# The supported Isaac Sim 5.1 lane separately verifies the resolved PhysX COM.
NEEDLE_SOURCE_CENTRE_OF_MASS_M = (0.016004362454017, 0.001034805027302, 0.000955963914748)
NEEDLE_CENTRE_OF_MASS_BODY_LOCAL_M = tuple(
    coordinate * scale for coordinate, scale in zip(NEEDLE_SOURCE_CENTRE_OF_MASS_M, NEEDLE_SCALE, strict=True)
)

# The dry steel/steel friction coefficients are likewise declared reference
# assumptions, not values authored by I4H and not tuned or measured against
# this task.  Restitution zero encodes a non-bouncing surgical tool assumption.
NEEDLE_STATIC_FRICTION = 0.74
NEEDLE_DYNAMIC_FRICTION = 0.57
NEEDLE_RESTITUTION = 0.0
# The pinned PSM jaw material contains an anomalous dynamic coefficient of
# 10.0.  Resolving the pair with ``max`` would make retention depend on that
# value rather than on the declared dry steel/steel model above.  ``min`` has
# higher PhysX precedence than the jaw material's unauthored/default
# ``average`` mode and therefore resolves the pair to 0.74 static / 0.57
# dynamic friction without modifying the shared robot asset.
NEEDLE_FRICTION_COMBINE_MODE = "min"
NEEDLE_RESTITUTION_COMBINE_MODE = "min"

__all__ = [
    "I4H_CATALOGUE_COMMIT",
    "I4H_CATALOGUE_LICENCE",
    "I4H_CATALOGUE_LICENCE_URL",
    "I4H_CATALOGUE_RELEASE",
    "I4H_CONTENT_REVISION",
    "I4H_CONTENT_ROOT",
    "I4HAssetPin",
    "NEEDLE_ASSET",
    "NEEDLE_BODY_LOCAL_AABB_MAX_M",
    "NEEDLE_BODY_LOCAL_AABB_MIN_M",
    "NEEDLE_BODY_LOCAL_EXTENT_M",
    "NEEDLE_CENTRE_OF_MASS_BODY_LOCAL_M",
    "NEEDLE_CONVEX_FALLBACK_ASSET",
    "NEEDLE_DYNAMIC_FRICTION",
    "NEEDLE_FRICTION_COMBINE_MODE",
    "NEEDLE_MASS_KG",
    "NEEDLE_REFERENCE_DENSITY_KG_M3",
    "NEEDLE_RESTITUTION",
    "NEEDLE_RESTITUTION_COMBINE_MODE",
    "NEEDLE_SCALE",
    "NEEDLE_STATIC_FRICTION",
    "NEEDLE_SOURCE_AABB_MAX_M",
    "NEEDLE_SOURCE_AABB_MIN_M",
    "NEEDLE_SOURCE_CENTRE_OF_MASS_M",
    "NEEDLE_SOURCE_VOLUME_M3",
    "SUTURE_PAD_ASSET",
    "verify_remote_asset_sha256",
]
