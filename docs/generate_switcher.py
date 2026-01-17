#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate switcher.json for version dropdown from git tags and branches."""

import json
import os
import re
import subprocess
from pathlib import Path


def get_git_tags_and_branches():
    """Get all tags and branches matching the sphinx-multiversion patterns."""
    # Get tag whitelist pattern from environment or use default
    tag_pattern = os.getenv("SMV_TAG_WHITELIST", r"^v[1-9]\d*\.\d+\.\d+$")
    branch_pattern = os.getenv("SMV_BRANCH_WHITELIST", r"^(main)$")

    versions = []

    # Get all tags
    try:
        result = subprocess.run(["git", "tag", "-l"], capture_output=True, text=True, check=True)
        tags = result.stdout.strip().split("\n")
        tags = [t for t in tags if t and re.match(tag_pattern, t)]

        # Sort tags by version (newest first)
        def version_key(tag):
            # Extract version numbers from tag like v1.2.3
            match = re.match(r"^v?(\d+)\.(\d+)\.(\d+)", tag)
            if match:
                return tuple(int(x) for x in match.groups())
            return (0, 0, 0)

        tags.sort(key=version_key, reverse=True)

        # remove tags older than version 2.0.2
        tags = [t for t in tags if version_key(t) >= (2, 0, 2)]

        for tag in tags:
            version_num = tag.lstrip("v")
            versions.append({
                "version": version_num,
                "name": f"v{version_num}",
                "url": f"https://isaac-sim.github.io/IsaacLab/{tag}/",
            })

    except subprocess.CalledProcessError:
        print("Warning: Could not get git tags")

    # Mark the latest tag as stable/preferred
    if versions:
        versions[0]["name"] = f"{versions[0]['name']} (stable)"
        versions[0]["preferred"] = True

    # Get all branches
    try:
        result = subprocess.run(["git", "branch", "-r"], capture_output=True, text=True, check=True)
        branches = result.stdout.strip().split("\n")
        # Clean up branch names (remove origin/ prefix and whitespace)
        branches = [b.strip().replace("origin/", "") for b in branches if b.strip()]
        # Filter by pattern and exclude HEAD
        branches = [b for b in branches if b and re.match(branch_pattern, b) and "HEAD" not in b]

        for branch in branches:
            if branch == "main":
                versions.insert(
                    0,
                    {
                        "version": "main",
                        "name": "main (latest)",
                        "url": "https://isaac-sim.github.io/IsaacLab/main/",
                    },
                )
            elif branch == "devel":
                versions.insert(
                    1 if "main" in [v["version"] for v in versions] else 0,
                    {
                        "version": "dev",
                        "name": "devel (development)",
                        "url": "https://isaac-sim.github.io/IsaacLab/devel/",
                    },
                )
            else:
                versions.append({
                    "version": branch,
                    "name": branch,
                    "url": f"https://isaac-sim.github.io/IsaacLab/{branch}/",
                })

    except subprocess.CalledProcessError:
        print("Warning: Could not get git branches")

    return versions


def main():
    """Generate switcher.json file."""
    # Get the docs directory
    docs_dir = Path(__file__).parent
    static_dir = docs_dir / "source" / "_static"
    static_dir.mkdir(parents=True, exist_ok=True)

    # Generate version list
    versions = get_git_tags_and_branches()

    if not versions:
        print("Warning: No versions found!")
        versions = [{
            "version": "main",
            "name": "main (latest)",
            "url": "https://isaac-sim.github.io/IsaacLab/main/",
            "preferred": True,
        }]

    # Write switcher.json
    switcher_file = static_dir / "switcher.json"
    with open(switcher_file, "w") as f:
        json.dump(versions, f, indent=4)

    print(f"Generated {switcher_file} with {len(versions)} versions:")
    for v in versions:
        print(f"  - {v['name']} ({v['version']})")


if __name__ == "__main__":
    main()
