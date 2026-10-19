# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for installation commands shared with historical documentation versions."""

import html
import re
import subprocess
import sys
from pathlib import Path


def test_historical_installation_directives_render_commands(tmp_path):
    """The current extension renders every install directive used by release/3.0.0."""
    extensions = Path(__file__).resolve().parents[2] / "docs/_extensions"
    source = tmp_path / "docs"
    source.mkdir()
    (source / "conf.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(extensions)!r})\n"
        "extensions = ['isaaclab_docs']\n"
        "isaacsim_version = '6.2.0.0'\n"
        "torch_version = '2.11.0'\n"
        "torchvision_version = '0.26.0'\n",
        encoding="utf-8",
    )
    (source / "index.rst").write_text(
        "Commands\n========\n\n"
        ".. isaaclab-isaacsim-install::\n\n"
        ".. isaaclab-isaacsim-install:: pip\n\n"
        ".. isaaclab-torch-install:: cu130\n\n"
        ".. isaaclab-torch-install:: cu130 pip\n",
        encoding="utf-8",
    )
    output = tmp_path / "html"

    subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "-q", str(source), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    rendered = html.unescape(re.sub(r"<[^>]+>", "", (output / "index.html").read_text(encoding="utf-8")))
    assert 'uv pip install "isaacsim[all,extscache]==6.2.0.0"' in rendered
    assert 'python -m pip install "isaacsim[all,extscache]==6.2.0.0"' in rendered
    assert "uv pip install -U torch==2.11.0 torchvision==0.26.0" in rendered
    assert "python -m pip install -U torch==2.11.0 torchvision==0.26.0" in rendered
