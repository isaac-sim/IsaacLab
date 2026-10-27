#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

# Pin the current runtime archive from the permanent interactive-demos release.
readonly archive_commit="1c3fdc57043a60d3c1b65d8d682ce85c0e544ca2"
readonly archive_url="https://github.com/isaac-sim/IsaacLab/releases/download/interactive-demos/isaaclab-browser-demos-${archive_commit}.tar.gz"
readonly archive_sha256="9577928fabc0115a5f60856c4dc80da266e489136ad6efe2dc5e2d70b701666f"

archive=$(mktemp)
trap 'rm -f "$archive"' EXIT

curl --fail --silent --show-error --location --retry 3 --connect-timeout 15 --max-time 180 \
  "$archive_url" --output "$archive"
printf '%s  %s\n' "$archive_sha256" "$archive" | sha256sum --check

rm -rf docs/_build/demos
mkdir -p docs/_build/demos
tar --extract --gzip --file "$archive" --directory docs/_build/demos --no-same-owner \
  --strip-components=1 site/

# Retain earlier gallery entry points and resolve older documentation links per version.
uv run --no-sync python - <<'PY'
import os
from pathlib import Path

build = Path("docs/_build")
demos = build / "demos"

for directory in [demos, *(demos / f"v{version}" for version in range(1, 7))]:
    directory.mkdir(parents=True, exist_ok=True)
    target = Path(os.path.relpath(demos / "interactive", directory)).as_posix() + "/"
    redirect = f"""<!doctype html>
<meta charset="utf-8">
<title>Isaac Lab interactive demos</title>
<script>location.replace('{target}' + location.search + location.hash);</script>
<p><a href="{target}">Open demo gallery</a></p>
"""
    (directory / "index.html").write_text(redirect)

pages = list(build.glob("**/source/isaaclab/interactive_demos.html"))
pages.extend(build.glob("**/source/setup/interactive_demos.html"))
for page in pages:
    relative = Path(os.path.relpath(build / "demos", page.parent)).as_posix()
    content = page.read_text()
    page.write_text(content.replace('href="../../../demos/', f'href="{relative}/'))
PY

if [[ -d docs/_build/current ]]; then
  cat > docs/_build/PREVIEW.txt <<'TXT'
Extract the complete docs-html artifact, then run this command in its root directory:

    python -m http.server 8000

Documentation: http://localhost:8000/current/index.html
Interactive demos: http://localhost:8000/demos/interactive/

Use HTTP to preview the demos; opening the HTML files directly does not allow
the browser to load their WebAssembly modules and data through fetch().
TXT
fi
