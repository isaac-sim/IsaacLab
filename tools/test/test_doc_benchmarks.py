# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for self-contained benchmark snapshots in built documentation."""

import csv
import json
import re
import subprocess
import sys
from pathlib import Path


def test_benchmark_snapshots_are_embedded_and_refresh_on_csv_changes(tmp_path):
    """HTML preserves both CSV snapshots safely and tracks changes during incremental builds."""
    extensions = Path(__file__).resolve().parents[2] / "docs/_extensions"
    source = tmp_path / "docs"
    snapshots = source / "source/_static/benchmarks"
    snapshots.mkdir(parents=True)
    (source / "conf.py").write_text(
        f"import sys\nsys.path.insert(0, {str(extensions)!r})\nextensions = ['isaaclab_docs']\n",
        encoding="utf-8",
    )
    (source / "index.rst").write_text("Benchmarks\n==========\n\n.. isaaclab-benchmark-data::\n", encoding="utf-8")
    rows = {
        "release": [{"task": "Isaac-Cartpole", "collection_fps_mean": "100", "notes": '</script>\nquoted "value",'}],
        "develop": [{"task": "Isaac-Cartpole", "collection_fps_mean": "200", "notes": "historical snapshot"}],
    }

    def write_snapshot(channel):
        with (snapshots / f"environment-performance-{channel}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["task", "collection_fps_mean", "notes"])
            writer.writeheader()
            writer.writerows(rows[channel])

    def build_snapshots():
        output = tmp_path / "html"
        subprocess.run(
            [sys.executable, "-m", "sphinx", "-W", "-q", str(source), str(output)],
            check=True,
            capture_output=True,
            text=True,
        )
        html = (output / "index.html").read_text(encoding="utf-8")
        payload = re.search(r'<script type="application/json" data-environment-benchmark-rows>(.*?)</script>', html)
        assert payload is not None
        return json.loads(payload.group(1))

    for channel in rows:
        write_snapshot(channel)
    assert build_snapshots() == rows

    rows["develop"][0]["collection_fps_mean"] = "300"
    write_snapshot("develop")
    assert build_snapshots() == rows
