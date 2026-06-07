# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single source of truth for intersphinx inventory locations.

External documentation servers (notably the NVIDIA Omniverse/Isaac Sim sites)
intermittently drop requests or return transient ``404``s. Because the docs
build runs with ``-W`` (see ``docs/Makefile``), a single failed inventory fetch
turns a warning into a fatal error and flakes CI.

To make the build resilient, ``conf.py`` reads the inventory from a locally
cached copy first and only falls back to the live server if the cache is
missing. CI warms that cache by running this module as a script
(``python docs/_extensions/intersphinx_inventories.py``), downloading each
``objects.inv`` with retries before invoking Sphinx.
"""

from __future__ import annotations

import os
import posixpath

# Mapping of intersphinx name -> documentation root URI used for cross-reference
# links. This is the only place inventory locations are declared.
INVENTORY_URIS: dict[str, str] = {
    "python": "https://docs.python.org/3",
    "numpy": "https://numpy.org/doc/stable/",
    "trimesh": "https://trimesh.org/",
    # NOTE: pinned to /docs/2.11/ because /docs/stable/objects.inv currently 404s
    "torch": "https://docs.pytorch.org/docs/2.11/",
    "isaacsim": "https://docs.isaacsim.omniverse.nvidia.com/6.0.0/py/",
    "gymnasium": "https://gymnasium.farama.org/",
    # NOTE: pinned to /stable/ because /objects.inv at the root currently 404s
    "warp": "https://nvidia.github.io/warp/stable/",
    "omniverse": "https://docs.omniverse.nvidia.com/dev-guide/latest",
}

# Directory (relative to the docs source dir) holding the pre-fetched fallback
# inventories. Kept out of version control; warmed by CI before the build.
CACHE_DIR = "_intersphinx"

# CI sets this to an absolute path so the cache resolves even from the temporary
# per-ref checkouts that ``sphinx-multiversion`` builds in (where a path relative
# to the source dir would not exist).
CACHE_DIR_ENV = "ISAACLAB_INTERSPHINX_CACHE"


def cache_dir() -> str:
    """Return the inventory cache directory (absolute via env var, else relative)."""
    return os.environ.get(CACHE_DIR_ENV, CACHE_DIR)


def local_inventory_path(name: str) -> str:
    """Return the cached fallback inventory path for ``name``.

    Absolute when :data:`CACHE_DIR_ENV` is set, otherwise relative to docs/.
    """
    return posixpath.join(cache_dir(), f"{name}.inv")


def inventory_url(uri: str) -> str:
    """Return the ``objects.inv`` URL for a documentation root ``uri``."""
    return uri.rstrip("/") + "/objects.inv"


def build_intersphinx_mapping() -> dict[str, tuple[str, tuple[str, None]]]:
    """Build the ``intersphinx_mapping`` consumed by ``conf.py``.

    The inventory location is the cached local file first, then the live server
    (``None``). Cross-reference links always point at the public URI.
    """
    return {name: (uri, (local_inventory_path(name), None)) for name, uri in INVENTORY_URIS.items()}


def _fetch_all() -> int:
    """Download every ``objects.inv`` into the cache dir with retries (best effort)."""
    import time
    import urllib.error
    import urllib.request

    docs_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = cache_dir()
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(docs_dir, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    failures = []
    for name, uri in INVENTORY_URIS.items():
        url = inventory_url(uri)
        dest = os.path.join(out_dir, f"{name}.inv")
        # Some doc CDNs reject the default ``Python-urllib`` agent with a 403;
        # send a browser-like User-Agent so the cache matches what Sphinx fetches.
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (IsaacLab docs build)"})
        for attempt in range(1, 6):
            try:
                with urllib.request.urlopen(request, timeout=30) as resp:  # noqa: S310 (trusted doc URLs)
                    data = resp.read()
                with open(dest, "wb") as handle:
                    handle.write(data)
                print(f"[ok] {name}: {url} -> {dest} ({len(data)} bytes)")
                break
            except (urllib.error.URLError, OSError) as exc:
                print(f"[retry {attempt}/5] {name}: {url}: {exc}")
                time.sleep(2 * attempt)
        else:
            failures.append(name)

    if failures:
        # Do not fail CI here: the build still tries the live server as a
        # fallback, and a persistent outage will surface as a build warning.
        print(f"[warn] could not pre-fetch: {', '.join(failures)} (build will try the live server)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_fetch_all())
