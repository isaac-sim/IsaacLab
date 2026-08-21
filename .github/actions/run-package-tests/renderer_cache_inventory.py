# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Report the size and stable metadata fingerprint of a renderer cache tree."""

import hashlib
import os
import pathlib
import sys


def fingerprint(root: pathlib.Path) -> str:
    """Return a digest of cache file paths, sizes, and modification times."""
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        try:
            stat = path.stat()
        except OSError:
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
    return digest.hexdigest()


def inventory(root: pathlib.Path) -> tuple[int, int, dict[str, int]]:
    """Return total bytes, file count, and bytes grouped by top-level directory."""
    total_bytes = 0
    file_count = 0
    groups: dict[str, int] = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = pathlib.Path(dirpath, name)
            try:
                size = path.stat().st_size
            except OSError:
                continue
            relative = path.relative_to(root)
            group = relative.parts[0] if relative.parts else "."
            groups[group] = groups.get(group, 0) + size
            total_bytes += size
            file_count += 1
    return total_bytes, file_count, groups


def main() -> int:
    """Print a fingerprint or human-readable renderer cache inventory."""
    root = pathlib.Path(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "--fingerprint":
        print(fingerprint(root))
        return 0

    label = sys.argv[2] if len(sys.argv) > 2 else "Renderer cache"
    if not root.is_dir():
        print(f"{label}: directory is missing")
        return 0

    total_bytes, file_count, groups = inventory(root)
    breakdown = ", ".join(f"{name}={size / 1e6:.0f} MB" for name, size in sorted(groups.items())) or "empty"
    print(f"{label}: {total_bytes / 1e6:.0f} MB across {file_count} files ({breakdown})")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"🔵 {label}: {total_bytes / 1e6:.0f} MB across {file_count} files ({breakdown})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
