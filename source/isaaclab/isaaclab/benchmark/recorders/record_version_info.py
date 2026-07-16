# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.metadata
import os
import subprocess
import sys

from isaaclab.benchmark.interfaces import MeasurementData, MeasurementDataRecorder
from isaaclab.benchmark.measurements import DictMetadata, StringMetadata

# Path to the repository root.
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 6))


class VersionInfoRecorder(MeasurementDataRecorder):
    def __init__(self):
        self._version_info: dict[str, str | None] = {}
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

    def _get_pkg_version(self, pip_name: str) -> str | None:
        """Get version via importlib.metadata (pip package name, no module import)."""
        try:
            return importlib.metadata.version(pip_name)
        except Exception:
            return None

    def _get_kit_version(self) -> str | None:
        """Get the version from the active Kit application."""
        app_module = sys.modules.get("omni.kit.app")
        if app_module is None:
            return None
        try:
            app = app_module.get_app()
            get_version = getattr(app, "get_kit_version", None)
            if not callable(get_version):
                get_version = getattr(app, "get_build_version", None)
            return str(get_version()) if callable(get_version) else None
        except Exception:
            return None

    def _get_isaacsim_version(self) -> str | None:
        """Get the Isaac Sim version from an install or active Kit runtime."""
        try:
            with open(os.path.join(os.environ["ISAAC_PATH"], "VERSION")) as file:
                return file.read().strip()
        except Exception:
            pass
        try:
            from isaacsim.core.version import get_version

            core, prerelease, _, _, _, _, _, buildtag = get_version()
            if core:
                version = str(core)
                if prerelease:
                    version += f"-{prerelease}"
                if buildtag:
                    version += f"+{buildtag}"
                return version
        except Exception:
            pass
        return self._get_pkg_version("isaacsim")

    def _record(self, key: str, version: str | None, *, nullable: bool = False) -> None:
        """Store a version entry, preserving null for explicitly nullable keys."""
        if version or nullable:
            self._version_info[key] = version

    def _get_version_info(self) -> None:
        # isaaclab
        self._record("isaaclab", self._get_version("isaaclab"))

        # warp - try config.version first, then __version__
        version = self._get_version("warp", "config.version") or self._get_version("warp")
        self._record("warp", version)

        # Kit and Isaac Sim are meaningful only for an active Kit runtime.
        version = self._get_kit_version()
        self._record("kit", version, nullable=True)
        self._record("isaacsim", self._get_isaacsim_version() if version else None, nullable=True)

        # torch
        self._record("torch", self._get_version("torch"))

        # numpy
        self._record("numpy", self._get_version("numpy"))

        # IsaacLab sub-packages
        self._record("isaaclab_newton", self._get_pkg_version("isaaclab_newton"))
        self._record("isaaclab_physx", self._get_pkg_version("isaaclab_physx"))
        self._record("isaaclab_ov", self._get_pkg_version("isaaclab_ov"))
        self._record("isaaclab_tasks", self._get_pkg_version("isaaclab_tasks"))
        self._record("isaaclab_rl", self._get_pkg_version("isaaclab_rl"))

        # Renderers & physics engines
        self._record("ovrtx", self._get_pkg_version("ovrtx"), nullable=True)
        self._record("ovphysx", self._get_pkg_version("isaaclab_ovphysx"), nullable=True)
        self._record("newton", self._get_pkg_version("newton"))
        self._record("mujoco", self._get_pkg_version("mujoco"))
        self._record("mujoco_warp", self._get_pkg_version("mujoco-warp"))

        # RL frameworks
        self._record("rl_games", self._get_pkg_version("rl_games"))
        self._record("rsl_rl", self._get_pkg_version("rsl-rl-lib"))
        self._record("stable_baselines3", self._get_pkg_version("stable_baselines3"))
        self._record("skrl", self._get_pkg_version("skrl"))

        # Key dependencies
        self._record("gymnasium", self._get_pkg_version("gymnasium"))
        self._record("cuda_bindings", self._get_pkg_version("cuda-bindings"))
        self._record("usd_core", self._get_pkg_version("usd-core"))

        # Release version from root VERSION file
        version_file = os.path.join(_REPO_ROOT, "VERSION")
        try:
            with open(version_file) as f:
                self._record("isaaclab_release", f.read().strip())
        except Exception:
            pass

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
