# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for tools/setup_usd_libs.py.

Covers two Isaac Sim install layouts:

* **Symlink / binary install** — ``ISAACLAB_PATH/_isaac_sim/extscache/omni.usd.libs-<ver>/``
* **Wheel / pip install** — ``isaacsim`` package on ``sys.path``; ``extscache`` lives next
  to the package ``__init__.py``.

Each test runs the script as a subprocess so the module-level code executes in a
clean interpreter.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _script_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "tools" / "setup_usd_libs.py"
        if candidate.is_file():
            return candidate
    raise RuntimeError("Could not find tools/setup_usd_libs.py")


def _run(env_overrides: dict[str, str], tmp_path: Path) -> tuple[int, str, str]:
    """Run setup_usd_libs.py as a subprocess and return (returncode, stdout, stderr)."""
    import subprocess

    env = {**os.environ, **env_overrides}
    result = subprocess.run(
        [sys.executable, str(_script_path())],
        capture_output=True,
        text=True,
        env=env,
    )
    return result.returncode, result.stdout, result.stderr


def _make_usd_libs(base: Path, version: str = "26.05.0") -> Path:
    """Create a minimal omni.usd.libs extension directory under *base*/extscache."""
    usd_libs = base / "extscache" / f"omni.usd.libs-{version}"
    (usd_libs / "pxr").mkdir(parents=True)
    return usd_libs


# ---------------------------------------------------------------------------
# Symlink / binary install
# ---------------------------------------------------------------------------


class TestSymlinkInstall:
    """ISAACLAB_PATH/_isaac_sim/extscache layout (binary / developer install)."""

    def _isaaclab_env(self, tmp_path: Path) -> dict[str, str]:
        return {"ISAACLAB_PATH": str(tmp_path), "PYTHONPATH": ""}

    def test_prints_usd_libs_dir(self, tmp_path: Path) -> None:
        """Script prints the usd_libs_dir path when omni.usd.libs exists."""
        usd_libs = _make_usd_libs(tmp_path / "_isaac_sim")

        rc, stdout, _ = _run(self._isaaclab_env(tmp_path), tmp_path)

        assert rc == 0
        assert stdout == str(usd_libs)

    def test_creates_pxr_init_when_missing(self, tmp_path: Path) -> None:
        """Script creates pxr/__init__.py so the directory becomes an importable package."""
        usd_libs = _make_usd_libs(tmp_path / "_isaac_sim")

        _run(self._isaaclab_env(tmp_path), tmp_path)

        assert (usd_libs / "pxr" / "__init__.py").is_file()

    def test_leaves_existing_pxr_init_intact(self, tmp_path: Path) -> None:
        """Script does not overwrite an existing pxr/__init__.py."""
        usd_libs = _make_usd_libs(tmp_path / "_isaac_sim")
        init_py = usd_libs / "pxr" / "__init__.py"
        init_py.write_text("# sentinel")

        _run(self._isaaclab_env(tmp_path), tmp_path)

        assert init_py.read_text() == "# sentinel"

    def test_picks_latest_version(self, tmp_path: Path) -> None:
        """Script selects the lexicographically last omni.usd.libs version."""
        _make_usd_libs(tmp_path / "_isaac_sim", "25.11.0")
        newer = _make_usd_libs(tmp_path / "_isaac_sim", "26.05.0")

        _, stdout, _ = _run(self._isaaclab_env(tmp_path), tmp_path)

        assert stdout == str(newer)

    def test_no_usd_libs_produces_no_output(self, tmp_path: Path) -> None:
        """Script exits 0 with no stdout when extscache has no omni.usd.libs dirs."""
        (tmp_path / "_isaac_sim" / "extscache").mkdir(parents=True)

        rc, stdout, _ = _run(self._isaaclab_env(tmp_path), tmp_path)

        assert rc == 0
        assert stdout == ""

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod semantics differ on Windows")
    def test_readonly_pxr_dir_warns_and_exits_cleanly(self, tmp_path: Path) -> None:
        """Script emits [WARNING] to stderr and exits 0 when pxr/ is not writable."""
        usd_libs = _make_usd_libs(tmp_path / "_isaac_sim")
        pxr_dir = usd_libs / "pxr"
        pxr_dir.chmod(0o555)

        try:
            rc, _, stderr = _run(self._isaaclab_env(tmp_path), tmp_path)
        finally:
            pxr_dir.chmod(0o755)

        assert rc == 0
        assert "[WARNING]" in stderr


# ---------------------------------------------------------------------------
# Wheel / pip install
# ---------------------------------------------------------------------------


class TestWheelInstall:
    """Wheel install layout: isaacsim on sys.path, extscache next to __init__.py."""

    def _wheel_env(self, isaacsim_site: Path) -> dict[str, str]:
        """Environment that hides the symlink path and exposes a fake isaacsim package."""
        existing = os.environ.get("PYTHONPATH", "")
        pythonpath = str(isaacsim_site) + (os.pathsep + existing if existing else "")
        return {
            "ISAACLAB_PATH": "",  # no _isaac_sim symlink
            "PYTHONPATH": pythonpath,
        }

    def _make_fake_isaacsim(self, base: Path, version: str = "26.05.0") -> Path:
        """Create a minimal fake isaacsim package with an extscache directory."""
        pkg = base / "isaacsim"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").write_text("")
        return _make_usd_libs(pkg, version)

    def test_prints_usd_libs_dir(self, tmp_path: Path) -> None:
        """Script finds omni.usd.libs via importlib.util.find_spec when no symlink exists."""
        usd_libs = self._make_fake_isaacsim(tmp_path)

        rc, stdout, _ = _run(self._wheel_env(tmp_path), tmp_path)

        assert rc == 0
        assert stdout == str(usd_libs)

    def test_creates_pxr_init_when_missing(self, tmp_path: Path) -> None:
        """Script promotes pxr/ to a package in the wheel-install layout too."""
        usd_libs = self._make_fake_isaacsim(tmp_path)

        _run(self._wheel_env(tmp_path), tmp_path)

        assert (usd_libs / "pxr" / "__init__.py").is_file()

    def test_picks_latest_version(self, tmp_path: Path) -> None:
        """Script selects the latest omni.usd.libs in the wheel extscache."""
        pkg = tmp_path / "isaacsim"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        _make_usd_libs(pkg, "25.11.0")
        newer = _make_usd_libs(pkg, "26.05.0")

        _, stdout, _ = _run(self._wheel_env(tmp_path), tmp_path)

        assert stdout == str(newer)
