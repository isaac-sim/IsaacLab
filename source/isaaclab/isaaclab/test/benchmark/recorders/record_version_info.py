# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import importlib.metadata
import os
import subprocess
import sys

from isaaclab.test.benchmark.interfaces import MeasurementData, MeasurementDataRecorder
from isaaclab.test.benchmark.measurements import DictMetadata, StringMetadata

# Path to the repository root.
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 6))


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
            module = importlib.import_module(module_name)
            # Handle nested attributes like "config.version"
            for attr in version_attr.split("."):
                module = getattr(module, attr)
            if callable(module):
                module = module()
            return str(module)
        except Exception:
            return None

    def _get_pkg_version(self, pip_name: str) -> str | None:
        """Get version via importlib.metadata (pip package name, no module import)."""
        try:
            return importlib.metadata.version(pip_name)
        except Exception:
            return None

    def _record(self, key: str, version: str | None) -> None:
        """Store a version entry only if version is non-empty."""
        if version:
            self._version_info[key] = version

    def _get_isaac_sim_version(self) -> str | None:
        """Get the Isaac Sim runtime version."""
        try:
            from isaaclab.utils.version import get_isaac_sim_version

            return str(get_isaac_sim_version())
        except Exception:
            return self._get_version("isaacsim") or self._get_pkg_version("isaacsim")

    def _get_kit_version(self) -> str | None:
        """Get the Omniverse Kit runtime build version when Kit is running."""
        version = self._get_kit_app_version()
        if version:
            return version
        return self._get_kit_settings_version()

    def _get_kit_app_version(self) -> str | None:
        """Get Kit version from the running ``omni.kit.app`` instance."""
        kit_app = sys.modules.get("omni.kit.app")
        if kit_app is None:
            return None

        try:
            app = kit_app.get_app()
        except Exception:
            return None
        if app is None:
            return None

        for getter_name in ("get_kit_version", "get_build_version", "get_kernel_version"):
            try:
                getter = getattr(app, getter_name, None)
                version = getter() if callable(getter) else getter
                if version:
                    return str(version)
            except Exception:
                pass
        return None

    def _get_kit_settings_version(self) -> str | None:
        """Get Kit version from Carb settings when available."""
        carb = sys.modules.get("carb")
        carb_settings = sys.modules.get("carb.settings")
        if carb_settings is None and carb is not None:
            carb_settings = getattr(carb, "settings", None)
        if carb_settings is None:
            return None

        try:
            settings = carb_settings.get_settings()
        except Exception:
            return None

        for key in ("/app/kit/version", "/app/buildVersion"):
            try:
                version = settings.get(key)
            except Exception:
                continue
            if version:
                return str(version)
        return None

    def _get_version_info(self) -> None:
        # isaaclab
        self._record("isaaclab", self._get_version("isaaclab"))

        # warp - try config.version first, then __version__
        version = self._get_version("warp", "config.version") or self._get_version("warp")
        self._record("warp", version)

        # isaacsim
        self._record("isaacsim", self._get_isaac_sim_version())

        # kit (from the running Omniverse Kit app if available)
        self._record("kit", self._get_kit_version())

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
        self._record("ovrtx", self._get_pkg_version("ovrtx"))
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
