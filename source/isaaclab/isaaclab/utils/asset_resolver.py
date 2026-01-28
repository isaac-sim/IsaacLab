# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Offline Asset Resolver for Isaac Lab.

This module provides utilities to transparently redirect asset paths from Nucleus/S3
to local storage when running in offline mode.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class OfflineAssetResolver:
    """Singleton class to manage offline asset path resolution."""

    _instance: OfflineAssetResolver | None = None
    _enabled: bool = False
    _initialized: bool = False
    _spawn_hooks_installed: bool = False
    _env_hooks_installed: bool = False
    _offline_assets_dir: str | None = None
    _isaac_nucleus_dir: str | None = None
    _isaaclab_nucleus_dir: str | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _initialize(self):
        """Initialize the resolver with environment paths."""
        if self._initialized:
            return

        try:
            import carb.settings

            settings = carb.settings.get_settings()
            nucleus_root = settings.get("/persistent/isaac/asset_root/default")
        except (ImportError, RuntimeError):
            nucleus_root = None

        self.isaaclab_path = os.environ.get("ISAACLAB_PATH", os.getcwd())
        self._offline_assets_dir = os.path.join(self.isaaclab_path, "offline_assets")

        if nucleus_root:
            self._isaac_nucleus_dir = f"{nucleus_root}/Isaac"
            self._isaaclab_nucleus_dir = f"{nucleus_root}/Isaac/IsaacLab"

        self._initialized = True

        print("[OfflineAssetResolver] Initialized")
        print(f"  Offline assets dir: {self._offline_assets_dir}")
        if self._isaaclab_nucleus_dir:
            print(f"  Nucleus base:       {self._isaaclab_nucleus_dir}")

    def enable(self):
        """Enable offline asset resolution."""
        self._initialize()
        self._enabled = True
        print("[OfflineAssetResolver] Offline mode ENABLED")
        print(f"  All assets will be loaded from: {self._offline_assets_dir}")

        if not os.path.exists(self._offline_assets_dir):
            print("[OfflineAssetResolver] ⚠️  WARNING: Offline assets directory not found!")
            print("  Run: ./isaaclab.sh -p scripts/offline_setup/download_assets.py --categories all")

    def disable(self):
        """Disable offline asset resolution."""
        self._enabled = False
        print("[OfflineAssetResolver] Offline mode DISABLED")

    def is_enabled(self) -> bool:
        return self._enabled

    def are_spawn_hooks_installed(self) -> bool:
        return self._spawn_hooks_installed

    def set_spawn_hooks_installed(self, value: bool = True):
        self._spawn_hooks_installed = value

    def are_env_hooks_installed(self) -> bool:
        return self._env_hooks_installed

    def set_env_hooks_installed(self, value: bool = True):
        self._env_hooks_installed = value

    def resolve_path(self, asset_path: str) -> str:
        """Resolve an asset path to offline storage if available."""
        if not self._enabled or not isinstance(asset_path, str) or not asset_path:
            return asset_path

        path_to_convert = self._extract_relative_path(asset_path)

        if path_to_convert:
            offline_path = os.path.join(self._offline_assets_dir, path_to_convert)

            # Try exact path first
            if os.path.exists(offline_path):
                print(f"[OfflineAssetResolver] ✓ Using offline: {path_to_convert}")
                return offline_path

            # Try case-insensitive fallback (handles ANYmal-D vs anymal_d mismatches)
            resolved = self._find_case_insensitive(path_to_convert)
            if resolved:
                print(f"[OfflineAssetResolver] ✓ Using offline (case-adjusted): {resolved}")
                return os.path.join(self._offline_assets_dir, resolved)

            print(f"[OfflineAssetResolver] ⚠️  Not found locally: {path_to_convert}")
            print("[OfflineAssetResolver]    Falling back to Nucleus")

        return asset_path

    def _find_case_insensitive(self, relative_path: str) -> str | None:
        """
        Find a file using case-insensitive matching.

        Handles mismatches like:
        - ANYmal-D vs anymal_d
        - ANYmal-B vs anymal_b

        Args:
            relative_path: The relative path to find (e.g., "Robots/ANYbotics/ANYmal-D/anymal_d.usd")

        Returns:
            The actual relative path if found, None otherwise
        """
        parts = relative_path.replace("\\", "/").split("/")
        current_dir = self._offline_assets_dir
        resolved_parts = []

        for i, part in enumerate(parts):
            if not os.path.exists(current_dir):
                return None

            # For the last part (filename), do exact match first then case-insensitive
            # For directories, try to find a case-insensitive match
            try:
                entries = os.listdir(current_dir)
            except (OSError, PermissionError):
                return None

            # First try exact match
            if part in entries:
                resolved_parts.append(part)
                current_dir = os.path.join(current_dir, part)
                continue

            # Try case-insensitive match
            part_lower = part.lower()
            # Also try with common substitutions (hyphen <-> underscore)
            part_normalized = part_lower.replace("-", "_")

            found = None
            for entry in entries:
                entry_lower = entry.lower()
                entry_normalized = entry_lower.replace("-", "_")

                if entry_lower == part_lower or entry_normalized == part_normalized:
                    found = entry
                    break

            if found:
                resolved_parts.append(found)
                current_dir = os.path.join(current_dir, found)
            else:
                return None

        # Verify the final path exists
        final_path = os.path.join(self._offline_assets_dir, *resolved_parts)
        if os.path.exists(final_path):
            return "/".join(resolved_parts)

        return None

    def _extract_relative_path(self, asset_path: str) -> str | None:
        """Extract relative path from various Nucleus URL formats."""
        # Pattern 1: Isaac Lab assets with version
        match = re.search(r"/Assets/Isaac/[\d.]+/Isaac/IsaacLab/(.+)$", asset_path)
        if match:
            return match.group(1)

        # Pattern 2: General Isaac assets with version
        match = re.search(r"/Assets/Isaac/[\d.]+/Isaac/(?!IsaacLab)(.+)$", asset_path)
        if match:
            return match.group(1)

        # Pattern 3: Isaac Lab assets without version
        if self._isaaclab_nucleus_dir and asset_path.startswith(self._isaaclab_nucleus_dir):
            return asset_path[len(self._isaaclab_nucleus_dir) :].lstrip("/")

        # Pattern 4: General Isaac assets without version
        if self._isaac_nucleus_dir and asset_path.startswith(self._isaac_nucleus_dir):
            return asset_path[len(self._isaac_nucleus_dir) :].lstrip("/")

        return None

    def get_offline_assets_dir(self) -> str:
        self._initialize()
        return self._offline_assets_dir


# Global resolver instance
_resolver = OfflineAssetResolver()


# =============================================================================
# Public API Functions
# =============================================================================


def enable_offline_mode():
    """Enable offline asset resolution globally."""
    _resolver.enable()


def disable_offline_mode():
    """Disable offline asset resolution globally."""
    _resolver.disable()


def is_offline_mode_enabled() -> bool:
    """Check if offline mode is currently enabled."""
    return _resolver.is_enabled()


def resolve_asset_path(asset_path: str) -> str:
    """Resolve an asset path, redirecting to offline storage if enabled."""
    return _resolver.resolve_path(asset_path)


def get_offline_assets_dir() -> str:
    """Get the offline assets directory path."""
    return _resolver.get_offline_assets_dir()


def install_spawn_hooks():
    """Install path resolution hooks on Isaac Lab spawn config classes."""
    if _resolver.are_spawn_hooks_installed():
        return

    try:
        import isaaclab.sim as sim_utils

        # Patch UsdFileCfg
        if hasattr(sim_utils, "UsdFileCfg"):
            original_usd_init = sim_utils.UsdFileCfg.__init__

            def patched_usd_init(self, *args, **kwargs):
                original_usd_init(self, *args, **kwargs)
                if hasattr(self, "usd_path") and is_offline_mode_enabled():
                    self.usd_path = resolve_asset_path(self.usd_path)

            sim_utils.UsdFileCfg.__init__ = patched_usd_init
            print("[OfflineAssetResolver] Installed UsdFileCfg path hook")

        # Patch GroundPlaneCfg
        if hasattr(sim_utils, "GroundPlaneCfg"):
            original_ground_init = sim_utils.GroundPlaneCfg.__init__

            def patched_ground_init(self, *args, **kwargs):
                original_ground_init(self, *args, **kwargs)
                if hasattr(self, "usd_path") and is_offline_mode_enabled():
                    original_path = self.usd_path
                    self.usd_path = resolve_asset_path(self.usd_path)
                    if self.usd_path != original_path:
                        print(f"[OfflineAssetResolver] ✓ Resolved ground plane: {os.path.basename(self.usd_path)}")

            sim_utils.GroundPlaneCfg.__init__ = patched_ground_init
            print("[OfflineAssetResolver] Installed GroundPlaneCfg path hook")

        # Patch PreviewSurfaceCfg
        if hasattr(sim_utils, "PreviewSurfaceCfg"):
            original_surface_init = sim_utils.PreviewSurfaceCfg.__init__

            def patched_surface_init(self, *args, **kwargs):
                original_surface_init(self, *args, **kwargs)
                if hasattr(self, "texture_file") and is_offline_mode_enabled():
                    if self.texture_file:
                        self.texture_file = resolve_asset_path(self.texture_file)

            sim_utils.PreviewSurfaceCfg.__init__ = patched_surface_init
            print("[OfflineAssetResolver] Installed PreviewSurfaceCfg path hook")

        _resolver.set_spawn_hooks_installed(True)

    except ImportError:
        print("[OfflineAssetResolver] Could not install spawn hooks - isaaclab.sim not available")

    # Hook read_file() in isaaclab.utils.assets - this is where actuator nets are loaded
    try:
        import isaaclab.utils.assets as assets_module

        if hasattr(assets_module, "read_file"):
            original_read_file = assets_module.read_file

            def patched_read_file(path: str):
                if is_offline_mode_enabled():
                    resolved_path = resolve_asset_path(path)
                    return original_read_file(resolved_path)
                return original_read_file(path)

            assets_module.read_file = patched_read_file
            print("[OfflineAssetResolver] Installed read_file path hook")

    except ImportError:
        pass

    # Hook retrieve_file_path() - another common entry point for file loading
    try:
        import isaaclab.utils.assets as assets_module

        if hasattr(assets_module, "retrieve_file_path"):
            original_retrieve = assets_module.retrieve_file_path

            def patched_retrieve_file_path(path: str, *args, **kwargs):
                if is_offline_mode_enabled():
                    resolved_path = resolve_asset_path(path)
                    return original_retrieve(resolved_path, *args, **kwargs)
                return original_retrieve(path, *args, **kwargs)

            assets_module.retrieve_file_path = patched_retrieve_file_path
            print("[OfflineAssetResolver] Installed retrieve_file_path hook")

    except ImportError:
        pass


def install_env_hooks():
    """Install hooks on environment base classes to auto-patch configs."""
    if _resolver.are_env_hooks_installed():
        return

    try:
        from isaaclab.envs import ManagerBasedEnv

        original_manager_init = ManagerBasedEnv.__init__

        def patched_manager_init(self, cfg, *args, **kwargs):
            if is_offline_mode_enabled():
                patch_config_for_offline_mode(cfg)
            original_manager_init(self, cfg, *args, **kwargs)

        ManagerBasedEnv.__init__ = patched_manager_init
        print("[OfflineAssetResolver] Installed ManagerBasedEnv config hook")

    except ImportError:
        pass

    try:
        from isaaclab.envs import DirectRLEnv

        original_direct_init = DirectRLEnv.__init__

        def patched_direct_init(self, cfg, *args, **kwargs):
            if is_offline_mode_enabled():
                patch_config_for_offline_mode(cfg)
            original_direct_init(self, cfg, *args, **kwargs)

        DirectRLEnv.__init__ = patched_direct_init
        print("[OfflineAssetResolver] Installed DirectRLEnv config hook")

    except ImportError:
        pass

    try:
        from isaaclab.envs import DirectMARLEnv

        original_marl_init = DirectMARLEnv.__init__

        def patched_marl_init(self, cfg, *args, **kwargs):
            if is_offline_mode_enabled():
                patch_config_for_offline_mode(cfg)
            original_marl_init(self, cfg, *args, **kwargs)

        DirectMARLEnv.__init__ = patched_marl_init
        print("[OfflineAssetResolver] Installed DirectMARLEnv config hook")

    except ImportError:
        pass

    _resolver.set_env_hooks_installed(True)


def _is_nucleus_path(path: str) -> bool:
    """Check if a path is a Nucleus/S3 URL that needs resolution."""
    if not isinstance(path, str):
        return False
    return (
        "omniverse-content-production" in path
        or "nucleus" in path.lower()
        or path.startswith("omniverse://")
        or "/Assets/Isaac/" in path
    )


def _patch_object_recursive(obj, visited: set, depth: int = 0, max_depth: int = 15) -> int:
    """
    Recursively walk through an object and patch any asset paths.

    Args:
        obj: Object to patch (config, dataclass, etc.)
        visited: Set of already-visited object IDs to prevent infinite loops
        depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent stack overflow

    Returns:
        Number of paths patched
    """
    if depth > max_depth:
        return 0

    obj_id = id(obj)
    if obj_id in visited:
        return 0
    visited.add(obj_id)

    patches = 0

    # Skip primitive types and None
    if obj is None or isinstance(obj, (str, int, float, bool, bytes)):
        return 0

    # Handle dictionaries
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and _is_nucleus_path(value):
                resolved = resolve_asset_path(value)
                if resolved != value:
                    obj[key] = resolved
                    patches += 1
            else:
                patches += _patch_object_recursive(value, visited, depth + 1, max_depth)
        return patches

    # Handle lists/tuples
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            if isinstance(item, str) and _is_nucleus_path(item):
                resolved = resolve_asset_path(item)
                if resolved != item:
                    if isinstance(obj, list):
                        obj[i] = resolved
                        patches += 1
            else:
                patches += _patch_object_recursive(item, visited, depth + 1, max_depth)
        return patches

    # Handle objects with __dict__ (dataclasses, configclasses, regular objects)
    if hasattr(obj, "__dict__"):
        # Known asset path attributes to check directly
        asset_attrs = [
            "usd_path",
            "texture_file",
            "asset_path",
            "file_path",
            "mesh_file",
            "network_file",  # ActuatorNet LSTM/MLP files
            "policy_path",  # Policy checkpoint files
            "checkpoint_path",  # Model checkpoints
        ]

        for attr in asset_attrs:
            if hasattr(obj, attr):
                value = getattr(obj, attr)
                if isinstance(value, str) and _is_nucleus_path(value):
                    resolved = resolve_asset_path(value)
                    if resolved != value:
                        try:
                            setattr(obj, attr, resolved)
                            patches += 1
                        except (AttributeError, TypeError):
                            pass  # Read-only attribute

        # Recursively process all attributes
        try:
            for attr_name in dir(obj):
                # Skip private/magic attributes and methods
                if attr_name.startswith("_"):
                    continue

                try:
                    attr_value = getattr(obj, attr_name)

                    # Skip callables (methods, functions)
                    if callable(attr_value) and not hasattr(attr_value, "__dict__"):
                        continue

                    # Recurse into the attribute
                    patches += _patch_object_recursive(attr_value, visited, depth + 1, max_depth)

                except (AttributeError, TypeError, RuntimeError):
                    continue  # Skip attributes that can't be accessed

        except (TypeError, RuntimeError):
            pass  # Some objects don't support dir()

    return patches


def patch_config_for_offline_mode(env_cfg):
    """
    Patch environment configuration to use offline assets.

    This function recursively walks through the ENTIRE environment config tree
    and patches ANY asset paths it finds (usd_path, texture_file, etc.).

    This handles:
    - Robot USD paths (env_cfg.scene.robot.spawn.usd_path)
    - Sky light textures (env_cfg.scene.sky_light.spawn.texture_file)
    - Ground planes (env_cfg.scene.terrain.terrain_generator.ground_plane_cfg.usd_path)
    - Visualizer markers (env_cfg.commands.*.goal_vel_visualizer_cfg.markers.*.usd_path)
    - ANY other nested asset paths

    Args:
        env_cfg: Environment configuration object
    """
    if not is_offline_mode_enabled():
        return

    visited = set()
    patches = _patch_object_recursive(env_cfg, visited)

    if patches > 0:
        print(f"[OfflineAssetResolver] Patched {patches} pre-loaded config paths")


def setup_offline_mode():
    """
    Set up offline mode with all hooks and path resolution.

    This is the main entry point for enabling offline training. It is called
    automatically by AppLauncher when the --offline flag is set.
    """
    enable_offline_mode()
    install_spawn_hooks()
    install_env_hooks()
    print("[OfflineAssetResolver] Offline mode fully configured")
    print("[OfflineAssetResolver] Environment configs will be auto-patched at creation time")


# For backwards compatibility
def install_path_hooks():
    """Alias for install_spawn_hooks() for backwards compatibility."""
    install_spawn_hooks()


# Export public API
__all__ = [
    "enable_offline_mode",
    "disable_offline_mode",
    "is_offline_mode_enabled",
    "resolve_asset_path",
    "get_offline_assets_dir",
    "patch_config_for_offline_mode",
    "install_spawn_hooks",
    "install_env_hooks",
    "install_path_hooks",
    "setup_offline_mode",
]
