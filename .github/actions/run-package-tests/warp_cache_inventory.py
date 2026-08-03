# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Report what a Warp kernel cache directory actually contains.

Runs on the runner against the host cache directory. Reporting only: it never
fails a job and never gates publication. The point is to replace guesses about
cache size and retention with measurements - module counts, artifact mix, and
the spread of GPU targets a shared collection has accumulated.

Usage: warp_cache_inventory.py <cache-dir> [label]
"""

import collections
import os
import pathlib
import re
import sys

SM_PATTERN = re.compile(r"\.(sm\d+[a-z]?)\.")


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else "Warp cache"

    if not root.is_dir():
        print(f"{label}: directory is missing")
        return 0

    total_bytes = 0
    extensions: collections.Counter[str] = collections.Counter()
    sm_targets: collections.Counter[str] = collections.Counter()
    module_dirs = 0

    # Warp namespaces by its own version, so each child of the root is a version
    # and each grandchild is one module's compiled artifacts.
    for version_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        module_dirs += sum(1 for p in version_dir.iterdir() if p.is_dir())

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            try:
                total_bytes += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                continue
            extensions[pathlib.Path(name).suffix.lstrip(".") or "(none)"] += 1
            found = SM_PATTERN.search(name)
            if found:
                sm_targets[found.group(1)] += 1

    versions = sorted(p.name for p in root.iterdir() if p.is_dir())
    mix = ", ".join(f"{ext}={count}" for ext, count in extensions.most_common(6)) or "empty"
    targets = ", ".join(f"{sm}={count}" for sm, count in sorted(sm_targets.items())) or "none"

    print(f"{label}: {total_bytes / 1e6:.0f} MB across {module_dirs} module dirs")
    print(f"  warp versions: {', '.join(versions) or 'none'}")
    print(f"  artifacts:     {mix}")
    print(f"  gpu targets:   {targets}")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write(f"🔵 {label}: {total_bytes / 1e6:.0f} MB, {module_dirs} modules, targets [{targets}]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
