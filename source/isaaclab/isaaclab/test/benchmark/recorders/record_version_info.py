# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess

from isaaclab.test.benchmark.interfaces import MeasurementData, MeasurementDataRecorder
from isaaclab.test.benchmark.measurements import DictMetadata, StringMetadata


class VersionInfoRecorder(MeasurementDataRecorder):
    def __init__(self):
        self._version_info = {}
        self._dev_info = {}
        self._get_version_info()
        self._get_git_info()

    def _get_version(self, module_name: str, version_attr: str = "__version__") -> str | None:
        """Attempt to get version from a module.

        Args:
            module_name: Name of the module to import.
            version_attr: Attribute name containing the version.

        Returns:
            Version string or None if not available.
        """
        try:
            module = __import__(module_name)
            # Handle nested attributes like "config.version"
            for attr in version_attr.split("."):
                module = getattr(module, attr)
            return str(module)
        except Exception:
            return None

    def _get_version_info(self) -> None:
        # isaaclab
        version = self._get_version("isaaclab")
        if version:
            self._version_info["isaaclab"] = version

        # warp - try config.version first, then __version__
        version = self._get_version("warp", "config.version")
        if not version:
            version = self._get_version("warp")
        if version:
            self._version_info["warp"] = version

        # isaacsim
        version = self._get_version("isaacsim")
        if version:
            self._version_info["isaacsim"] = version

        # kit (from omni.kit if available)
        version = self._get_version("omni.kit", "app.get_app().get_build_version")
        if not version:
            version = self._get_version("carb", "settings.get_settings().get('/app/version')")
        if version:
            self._version_info["kit"] = version

        # torch
        version = self._get_version("torch")
        if version:
            self._version_info["torch"] = version

        # numpy
        version = self._get_version("numpy")
        if version:
            self._version_info["numpy"] = version

    def _get_git_info(self) -> None:
        """Get git repository information."""
        script_dir = os.path.dirname(os.path.abspath(__file__))

        try:
            # Get full commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=script_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self._dev_info["commit_hash"] = result.stdout.strip()
                self._dev_info["commit_hash_short"] = result.stdout.strip()[:8]

            # Get branch name
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=script_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self._dev_info["branch"] = result.stdout.strip()

            # Get commit date
            result = subprocess.run(
                ["git", "log", "-1", "--format=%ci"],
                cwd=script_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self._dev_info["commit_date"] = result.stdout.strip()

            # Check if working directory is dirty
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=script_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self._dev_info["dirty"] = len(result.stdout.strip()) > 0

        except Exception:
            pass

    def update(self) -> None:
        """No-op for version info as it doesn't change during runtime."""
        pass

    def get_initial_data(self) -> dict:
        return {
            "version_metadata": self._version_info,
            "dev": self._dev_info,
        }

    def get_runtime_data(self) -> dict:
        return {}

    def get_data(self) -> MeasurementData:
        metadata = []

        # Add version metadata
        for package, version in self._version_info.items():
            metadata.append(StringMetadata(name=f"{package}_version", data=version))

        # Add dev/git info as a dict metadata entry
        if self._dev_info:
            metadata.append(DictMetadata(name="dev", data=self._dev_info))

        return MeasurementData(measurements=[], metadata=metadata)
