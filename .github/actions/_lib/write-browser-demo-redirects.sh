#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

# Remove previously staged runtimes before writing the legacy entry points.
rm -rf docs/_build/demos

# Retain earlier gallery entry points and resolve older documentation links per version.
uv run --no-sync python - <<'PY'
import os
from pathlib import Path

build = Path("docs/_build")
demos = build / "demos"
target = "https://isaac-sim.github.io/IsaacLab-Web-Demos/"

for directory in [demos, demos / "interactive", *(demos / f"v{version}" for version in range(1, 7))]:
    directory.mkdir(parents=True, exist_ok=True)
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

Interactive demos are hosted externally and require an internet connection:
https://isaac-sim.github.io/IsaacLab-Web-Demos/
TXT
fi
