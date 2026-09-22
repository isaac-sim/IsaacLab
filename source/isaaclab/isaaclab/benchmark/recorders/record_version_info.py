# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.metadata
import os
import subprocess
import sys

from isaaclab.paths import ISAACLAB_ROOT

from ..interfaces import MeasurementData, MeasurementDataRecorder
from ..measurements import DictMetadata, StringMetadata

_GIT_QUERIES = (
    ("commit_hash", ["git", "rev-parse", "HEAD"]),
    ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    ("commit_date", ["git", "log", "-1", "--format=%ci"]),
    ("status", ["git", "status", "--porcelain"]),
)


class VersionInfoRecorder(MeasurementDataRecorder):
    """Record the versions of Isaac Lab, its runtimes and RL libraries, and the git checkout state."""

    def __init__(self):
        self._version_info: dict[str, str | None] = {}
        self._dev_info: dict[str, str | bool] = {}
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
            value = __import__(module_name)
            for attr in version_attr.split("."):
                value = getattr(value, attr)
            return str(value)
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
        self._record("isaaclab", self._get_version("isaaclab"))
        self._record("warp", self._get_version("warp", "config.version") or self._get_version("warp"))

        # Kit and Isaac Sim are meaningful only for an active Kit runtime.
        kit_version = self._get_kit_version()
        self._record("kit", kit_version, nullable=True)
        self._record("isaacsim", self._get_isaacsim_version() if kit_version else None, nullable=True)

        self._record("torch", self._get_version("torch"))
        self._record("numpy", self._get_version("numpy"))

        for key, pip_name in (
            ("isaaclab_newton", "isaaclab_newton"),
            ("isaaclab_physx", "isaaclab_physx"),
            ("isaaclab_ov", "isaaclab_ov"),
            ("isaaclab_tasks", "isaaclab_tasks"),
            ("isaaclab_rl", "isaaclab_rl"),
        ):
            self._record(key, self._get_pkg_version(pip_name))

        # Optional renderers and physics engines are recorded as null when absent.
        self._record("ovrtx", self._get_pkg_version("ovrtx"), nullable=True)
        self._record("ovphysx", self._get_pkg_version("ovphysx"), nullable=True)
        for key, pip_name in (
            ("newton", "newton"),
            ("mujoco", "mujoco"),
            ("mujoco_warp", "mujoco-warp"),
            ("rl_games", "rl_games"),
            ("rsl_rl", "rsl-rl-lib"),
            ("stable_baselines3", "stable_baselines3"),
            ("skrl", "skrl"),
            ("gymnasium", "gymnasium"),
            ("cuda_bindings", "cuda-bindings"),
            # usd-exchange is the standalone USD provider; usd-core only appears in environments
            # that predate the switch, so record whichever one is installed.
            ("usd_core", "usd-core"),
            ("usd_exchange", "usd-exchange"),
        ):
            self._record(key, self._get_pkg_version(pip_name))

        try:
            with open(os.path.join(ISAACLAB_ROOT, "VERSION")) as f:
                self._record("isaaclab_release", f.read().strip())
        except Exception:
            pass

    def _get_git_info(self) -> None:
        """Record the commit, branch, commit date, and dirty state of the checkout, when inside one."""
        cwd = os.path.dirname(os.path.abspath(__file__))
        try:
            for key, command in _GIT_QUERIES:
                result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=5)
                if result.returncode != 0:
                    continue
                output = result.stdout.strip()
                if key == "status":
                    self._dev_info["dirty"] = bool(output)
                else:
                    self._dev_info[key] = output
                if key == "commit_hash":
                    self._dev_info["commit_hash_short"] = output[:8]
        except Exception:
            pass

    def update(self) -> None:
        """Versions do not change while the benchmark runs."""

    def get_initial_data(self) -> dict:
        return {"version_metadata": self._version_info, "dev": self._dev_info}

    def get_runtime_data(self) -> dict:
        return {}

    def get_data(self) -> MeasurementData:
        metadata = [
            StringMetadata(name=f"{package}_version", data=version) for package, version in self._version_info.items()
        ]
        if self._dev_info:
            metadata.append(DictMetadata(name="dev", data=self._dev_info))
        return MeasurementData(measurements=[], metadata=metadata)
