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
import tomllib

LOCKFILE = "uv.lock"


def main() -> int:
    with open(LOCKFILE, "rb") as handle:
        packages = {pkg["name"]: pkg for pkg in tomllib.load(handle)["package"]}

    try:
        warp = packages["warp-lang"]["version"]
        mjwarp = packages["mujoco-warp"]["version"]
        newton = packages["newton"]
    except KeyError as exc:
        print(f"::error::{LOCKFILE} has no entry for {exc}", file=sys.stderr)
        return 1

    # Newton is a git dependency, so its resolved version is a placeholder like
    # 1.5.0.dev0 that does not move between revisions. Use the pinned revision.
    source = newton.get("source", {}).get("git", "")
    match = re.search(r"(?:rev=|#)([0-9a-f]{7,40})", source)
    if not match:
        print(f"::error::Could not read the pinned newton revision from {source!r}", file=sys.stderr)
        return 1

    print(f"wp{warp}-newton{match.group(1)[:8]}-mjwarp{mjwarp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
