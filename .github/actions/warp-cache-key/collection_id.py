# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print the Warp cache collection identifier for the current lockfile.

The Warp kernel cache is produced by Warp, Newton, MuJoCo Warp, and Isaac Lab's
own kernels. Only the first three belong in the cache key: Warp's module hash
already separates changed Isaac Lab source while leaving unchanged modules
reusable, so keying on Isaac Lab source would discard hits for no benefit.

GPU architecture is deliberately absent. Warp puts the target in the artifact
filename (``<module>.sm120.ptx``), so variants for several architectures coexist
in one module directory and a shared collection accumulates whatever the runner
fleet needs.
"""

import re
import sys

LOCKFILE = "uv.lock"


def read_packages(text: str) -> dict[str, str]:
    """Map each ``[[package]]`` name in the lockfile to the raw text of its block.

    ``tomllib`` is unavailable on the runners' system Python, and uv writes
    ``name``/``version``/``source`` as plain single-line pairs, so the few fields
    needed here are read with a regex instead of a TOML parser.
    """
    packages = {}
    for block in re.split(r"^\[\[package\]\]$", text, flags=re.MULTILINE)[1:]:
        # The block ends at the next table header starting at column zero.
        block = re.split(r"^\[", block, flags=re.MULTILINE)[0]
        name = re.search(r'^name = "(.*)"$', block, flags=re.MULTILINE)
        if name:
            packages[name.group(1)] = block
    return packages


def main() -> int:
    with open(LOCKFILE, encoding="utf-8") as handle:
        packages = read_packages(handle.read())

    try:
        warp = re.search(r'^version = "(.*)"$', packages["warp-lang"], flags=re.MULTILINE).group(1)
        mjwarp = re.search(r'^version = "(.*)"$', packages["mujoco-warp"], flags=re.MULTILINE).group(1)
        newton = packages["newton"]
        newton_version = re.search(r'^version = "(.*)"$', newton, flags=re.MULTILINE).group(1)
    except (AttributeError, KeyError) as exc:
        print(f"::error::{LOCKFILE} has no version entry for {exc}", file=sys.stderr)
        return 1

    # Git builds can share a placeholder version across revisions, so identify
    # them by commit. Registry releases are immutable and use their version.
    source = re.search(r"^source = (.*)$", newton, flags=re.MULTILINE)
    source = source.group(1) if source else ""
    match = re.search(r"(?:rev=|#)([0-9a-f]{7,40})", source)
    if match:
        newton_id = match.group(1)[:8]
    elif "registry" in source:
        newton_id = newton_version
    else:
        print(f"::error::Could not identify the newton package from {source!r}", file=sys.stderr)
        return 1

    print(f"wp{warp}-newton{newton_id}-mjwarp{mjwarp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
