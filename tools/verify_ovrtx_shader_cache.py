# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assert that the OVRTX shader cache directories CI mounted are present and writable.

Run inside the test container before tests start. Unlike ``verify_warp_cache``,
this cannot confirm the redirect took effect: the ovrtx settings extension has no
read-back, and the redirect happens in the pytest process, not this one. So the
check is deliberately limited to the mounts.

That still covers the fragile part. kit/ arrives as a nested bind mount that has
to land on top of the enclosing kit/cache tmpdir mount, and it has no source-side
redirect to fall back on - the mount is the whole mechanism. When it silently
fails the restore step still reports a hit, and the only symptom is a rendering
job several minutes slower.

Lives under ``tools/`` for the same reason as ``verify_warp_cache``:
``.dockerignore`` excludes ``.github/``, so a copy there would be missing from any
container that is not bind-mounted.
"""

from __future__ import annotations

import os
import sys

# Each tree is named by the env var carrying its container path; see the mount
# layout in .github/actions/run-tests/run_tests.sh. A tree whose variable is
# unset is not mounted for this job and is skipped.
CACHE_PATH_ENVS = {
    "kitless": "OVRTX_SHADER_CACHE_PATH",
    "kit": "OVRTX_KIT_SHADER_CACHE_PATH",
}


def check_tree(tree: str, cache_path: str) -> bool:
    """Report whether ``cache_path`` exists and can be written to."""
    probe = os.path.join(cache_path, ".ovrtx_cache_probe")
    try:
        with open(probe, "w") as handle:
            handle.write("ok")
        os.remove(probe)
    except OSError as exc:
        print(f"[verify_ovrtx_shader_cache] {tree}/ cache is not usable: {cache_path!r} - {exc}", file=sys.stderr)
        return False

    print(f"[verify_ovrtx_shader_cache] {tree}/ cache OK - path={cache_path!r}")
    return True


def main() -> int:
    mounted = {tree: os.environ[env] for tree, env in CACHE_PATH_ENVS.items() if os.environ.get(env)}
    if not mounted:
        print("[verify_ovrtx_shader_cache] no OVRTX shader cache paths are set; skipping.", file=sys.stderr)
        return 0

    # A list, not a generator: every tree gets checked and reported even after one
    # fails, so a single run shows every broken mount rather than the first.
    results = [check_tree(tree, path) for tree, path in sorted(mounted.items())]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
