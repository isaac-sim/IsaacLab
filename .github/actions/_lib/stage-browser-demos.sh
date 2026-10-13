#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

# Keep the archive and checksum pinned together; retain published version directories in each release.
readonly archive_url="https://github.com/isaac-sim/IsaacLab/releases/download/browser-demos-v1/isaaclab-browser-demos-v1.tar.gz"
readonly archive_sha256="a2f8e9c5b1845a3be189739e530c26853fce0d2be01b8039f3fd094928bf4394"

archive=$(mktemp)
trap 'rm -f "$archive"' EXIT

curl --fail --silent --show-error --location --retry 3 --connect-timeout 15 --max-time 180 \
  "$archive_url" --output "$archive"
printf '%s  %s\n' "$archive_sha256" "$archive" | sha256sum --check

mkdir -p docs/_build/demos
tar --extract --gzip --file "$archive" --directory docs/_build/demos --no-same-owner
