# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drop the stored credential for one image's registry from a docker config.

A registry can refuse a credential scoped to another organization a pull token
for a public repository instead of downgrading the request to anonymous, which
makes a public base image unreadable. Removing just that registry's entry lets
the pull proceed anonymously while every other registry in the same config
keeps working.

Usage: strip_registry_auth.py <config.json> <image-ref>
Exits 0 when a credential was dropped, 3 when nothing matched.
"""

from __future__ import annotations

import json
import sys

# Docker Hub is stored under several spellings; treat them as one registry.
_HUB_ALIASES = frozenset(
    {
        "docker.io",
        "index.docker.io",
        "registry-1.docker.io",
    }
)


def registry_host(image_ref):
    """Return the canonical registry host for a docker image reference."""
    first = image_ref.split("/")[0]
    if "/" in image_ref and ("." in first or ":" in first or first == "localhost"):
        # A registry may carry a port, and hosts are case-insensitive.
        host = first.lower().rsplit(":", 1)[0] if ":" in first else first.lower()
    else:
        # No registry component means Docker Hub.
        host = "index.docker.io"
    if host in _HUB_ALIASES:
        return "index.docker.io"
    return host


def _normalize(auth_key):
    """Reduce a config.json auths key to a comparable registry host."""
    trimmed = auth_key.split("://", 1)[-1]
    host = trimmed.split("/", 1)[0].lower()
    if ":" in host:
        host = host.rsplit(":", 1)[0]
    if host in _HUB_ALIASES:
        return "index.docker.io"
    return host


def main(argv):
    config_path, image_ref = argv[1], argv[2]
    target = registry_host(image_ref)

    with open(config_path) as handle:
        config = json.load(handle)

    auths = config.get("auths") or {}
    removed = sorted(key for key in auths if _normalize(key) == target)

    # A credHelpers entry would re-supply the credential from an external helper
    # even after its `auths` entry is gone, so the mapping for this registry is
    # removed too. A credential held only by a global credsStore is outside this
    # file, which is why the caller treats exit 3 as "nothing was dropped".
    helpers = config.get("credHelpers") or {}
    helper_keys = sorted(key for key in helpers if _normalize(key) == target)
    for key in helper_keys:
        del helpers[key]
    if helper_keys:
        config["credHelpers"] = helpers

    if not removed and not helper_keys:
        print(f"No stored credential for {target} was found in the docker config")
        return 3

    for key in removed:
        del auths[key]
    config["auths"] = auths
    with open(config_path, "w") as handle:
        json.dump(config, handle)

    dropped = ", ".join(removed + helper_keys)
    print(f"Dropped stored credential(s) for {target}: {dropped}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
