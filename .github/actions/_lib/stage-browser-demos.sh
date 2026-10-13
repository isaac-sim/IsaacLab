#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

# Keep the archive and checksum pinned together; retain published version directories in each release.
readonly archive_url="https://github.com/isaac-sim/IsaacLab/releases/download/browser-demos-v2/isaaclab-browser-demos-v2.tar.gz"
readonly archive_sha256="3f7a4e1d2cd83b4cd699b399c59cb66841517cdc672e0f0b26432d0f10946c50"

archive=$(mktemp)
trap 'rm -f "$archive"' EXIT

curl --fail --silent --show-error --location --retry 3 --connect-timeout 15 --max-time 180 \
  "$archive_url" --output "$archive"
printf '%s  %s\n' "$archive_sha256" "$archive" | sha256sum --check

mkdir -p docs/_build/demos
tar --extract --gzip --file "$archive" --directory docs/_build/demos --no-same-owner

# Resolve links per version so local artifacts and nested release branches use the same files.
uv run --no-sync python - <<'PY'
import os
from pathlib import Path

build = Path("docs/_build")
for page in build.glob("**/source/setup/interactive_demos.html"):
    relative = Path(os.path.relpath(build / "demos", page.parent)).as_posix()
    content = page.read_text()
    page.write_text(content.replace('href="../../../demos/', f'href="{relative}/'))
PY

if [[ -d docs/_build/current ]]; then
  cat > docs/_build/PREVIEW.txt <<'TXT'
Extract the complete docs-html artifact, then run this command in its root directory:

    python -m http.server 8000

Documentation: http://localhost:8000/current/index.html
Interactive demos: http://localhost:8000/demos/v2/

Use HTTP to preview the demos; opening the HTML files directly does not allow
the browser to load their WebAssembly modules and data through fetch().
TXT
fi
