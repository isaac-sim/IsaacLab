# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Uv virtual-environment helpers for test classes."""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
from pathlib import Path

from utils import run_cmd

_IS_WINDOWS = platform.system() == "Windows"


class UV_Mixin:
    """Mixin providing uv virtual-environment helpers for test classes.""

    env_path: Path
    python: Path
    cli_script: Path

    def create_uv_env(self, isaaclab_root: Path, env_name: str = "") -> None:
        """Create a uv environment and store info on self.

        Sets ``self.env_path``, ``self.python``, and ``self.cli_script``.

        Args:
            isaaclab_root: Path to the IsaacLab repository root.
            env_name: Name for the venv directory. A random name is
                generated when empty.
        """
        env_name = env_name if env_name else f"_isaaclab_install_ci_{os.urandom(4).hex()}"

        self.env_path = isaaclab_root / env_name
        self.cli_script = isaaclab_root / ("isaaclab.bat" if _IS_WINDOWS else "isaaclab.sh")

        if self.env_path.exists():
            shutil.rmtree(self.env_path)

        result = run_cmd([str(self.cli_script), "-u", env_name], cwd=isaaclab_root, check=False)
        assert result.returncode == 0, f"uv env creation failed:\n{result.stdout}\n{result.stderr}"
        assert self.env_path.exists(), f"Expected env directory {self.env_path} was not created"

        self.python = (self.env_path / "Scripts" / "python.exe") if _IS_WINDOWS else (self.env_path / "bin" / "python")
        assert self.python.exists(), f"Python executable not found at {self.python}"

    def destroy_uv_env(self) -> None:
        """Remove the uv environment directory if it exists."""
        if hasattr(self, "env_path") and self.env_path.exists():
            shutil.rmtree(self.env_path)

    def run_in_env(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        """Run a command inside the activated venv by sourcing the activate script.

        Args:
            cmd: Command and arguments to run inside the venv.
            **kwargs: Extra keyword arguments forwarded to :func:`run_cmd`.
        """
        escaped = " ".join(shlex.quote(str(a)) for a in cmd)
        if _IS_WINDOWS:
            activate = str(self.env_path / "Scripts" / "activate.bat")
            shell_cmd = f'call "{activate}" && {escaped}'
            return run_cmd(["cmd", "/c", shell_cmd], **kwargs)
        else:
            activate = shlex.quote(str(self.env_path / "bin" / "activate"))
            shell_cmd = f"source {activate} && {escaped}"
            return run_cmd(["bash", "-c", shell_cmd], **kwargs)
