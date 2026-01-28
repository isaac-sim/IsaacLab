# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Offline Asset Resolver for Isaac Lab.

This module provides utilities to transparently redirect asset paths from Nucleus/S3
to local storage when running in offline mode. It maintains the same directory structure
so that environment configs require no changes.

Key Features:
    - Automatic path resolution from Nucleus URLs to local filesystem
    - Transparent fallback to Nucleus if local asset missing
    - Monkey-patching of Isaac Lab spawn configs for automatic resolution
    - Support for versioned Nucleus URLs

Usage:
    from isaaclab.utils import setup_offline_mode, patch_config_for_offline_mode

    # Enable offline mode at start of script
    setup_offline_mode()

    # Patch environment config after loading
    patch_config_for_offline_mode(env_cfg)

    # All asset paths will now resolve to offline_assets/
"""

import os
import re
from typing import Optional

import carb.settings


class OfflineAssetResolver:
    """
    Singleton class to manage offline asset path resolution.

    When enabled, this resolver intercepts asset paths pointing to Nucleus servers
    and redirects them to the local offline_assets directory while maintaining the
    same directory structure.
    """

    _instance: Optional["OfflineAssetResolver"] = None
    _enabled: bool = False
    _offline_assets_dir: str | None = None
    _isaac_nucleus_dir: str | None = None
    _isaaclab_nucleus_dir: str | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the resolver with environment paths."""
        # Get Isaac Lab root path
        self.isaaclab_path = os.environ.get("ISAACLAB_PATH", os.getcwd())

        # Set offline assets directory
        self._offline_assets_dir = os.path.join(self.isaaclab_path, "offline_assets")

        # Get Nucleus directories from settings
        settings = carb.settings.get_settings()
        nucleus_root = settings.get("/persistent/isaac/asset_root/default")

        if nucleus_root:
            self._isaac_nucleus_dir = f"{nucleus_root}/Isaac"
            self._isaaclab_nucleus_dir = f"{nucleus_root}/Isaac/IsaacLab"

        print("[OfflineAssetResolver] Initialized")
        print(f"  Offline assets dir: {self._offline_assets_dir}")
        if self._isaaclab_nucleus_dir:
            print(f"  Nucleus base:       {self._isaaclab_nucleus_dir}")

    def enable(self):
        """Enable offline asset resolution."""
        self._enabled = True
        print("[OfflineAssetResolver] Offline mode ENABLED")
        print(f"  All assets will be loaded from: {self._offline_assets_dir}")

        # Verify offline assets directory exists
        if not os.path.exists(self._offline_assets_dir):
            print("[OfflineAssetResolver] ⚠️  WARNING: Offline assets directory not found!")
            print("  Run: ./isaaclab.sh -p scripts/offline_setup/download_assets.py")

    def disable(self):
        """Disable offline asset resolution."""
        self._enabled = False
        print("[OfflineAssetResolver] Offline mode DISABLED")

    def is_enabled(self) -> bool:
        """Check if offline mode is enabled."""
        return self._enabled

    def resolve_path(self, asset_path: str) -> str:
        """
        Resolve an asset path to either Nucleus or offline storage.

        This method handles multiple Nucleus URL formats including versioned URLs
        (e.g., .../Assets/Isaac/5.1/Isaac/...) and falls back to Nucleus if the
        local asset doesn't exist.

        Args:
            asset_path: Original asset path (may contain Nucleus URL)

        Returns:
            Resolved path (offline if enabled and exists, otherwise original)
        """
        if not self._enabled or not isinstance(asset_path, str) or not asset_path:
            return asset_path

        # Try to extract the relative path from various Nucleus URL formats
        path_to_convert = self._extract_relative_path(asset_path)

        if path_to_convert:
            offline_path = os.path.join(self._offline_assets_dir, path_to_convert)

            # Return offline path if file exists, otherwise fall back to Nucleus
            if os.path.exists(offline_path):
                print(f"[OfflineAssetResolver] ✓ Using offline: {path_to_convert}")
                return offline_path
            else:
                print(f"[OfflineAssetResolver] ⚠️  Not found locally: {path_to_convert}")
                print("[OfflineAssetResolver]    Falling back to Nucleus")

        return asset_path

    def _extract_relative_path(self, asset_path: str) -> Optional[str]:
        """
        Extract relative path from various Nucleus URL formats.

        Handles:
            - Versioned URLs: .../Assets/Isaac/5.1/Isaac/IsaacLab/Robots/...
            - Versioned URLs: .../Assets/Isaac/5.1/Isaac/Props/...
            - Non-versioned: .../Isaac/IsaacLab/Robots/...
            - Non-versioned: .../Isaac/Props/...

        Args:
            asset_path: Full Nucleus URL

        Returns:
            Relative path (e.g., "Robots/Unitree/Go2/go2.usd") or None if not a Nucleus path
        """
        # Pattern 1: Isaac Lab assets with version
        # e.g., .../Assets/Isaac/5.1/Isaac/IsaacLab/Robots/...
        match = re.search(r"/Assets/Isaac/[\d.]+/Isaac/IsaacLab/(.+)$", asset_path)
        if match:
            return match.group(1)

        # Pattern 2: General Isaac assets with version
        # e.g., .../Assets/Isaac/5.1/Isaac/Props/...
        match = re.search(r"/Assets/Isaac/[\d.]+/Isaac/(?!IsaacLab)(.+)$", asset_path)
        if match:
            return match.group(1)

        # Pattern 3: Isaac Lab assets without version (older format)
        if self._isaaclab_nucleus_dir and asset_path.startswith(self._isaaclab_nucleus_dir):
            return asset_path[len(self._isaaclab_nucleus_dir) :].lstrip("/")

        # Pattern 4: General Isaac assets without version (older format)
        if self._isaac_nucleus_dir and asset_path.startswith(self._isaac_nucleus_dir):
            return asset_path[len(self._isaac_nucleus_dir) :].lstrip("/")

        return None

    def get_offline_assets_dir(self) -> str:
        """Get the offline assets directory path."""
        return self._offline_assets_dir


# Global resolver instance
_resolver = OfflineAssetResolver()


# Public API functions
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
    """
    Resolve an asset path, redirecting to offline storage if enabled.

    Args:
        asset_path: Original asset path (may contain Nucleus URL)

    Returns:
        Resolved path (offline if mode enabled and file exists, otherwise original)
    """
    return _resolver.resolve_path(asset_path)


def get_offline_assets_dir() -> str:
    """Get the offline assets directory path."""
    return _resolver.get_offline_assets_dir()


def patch_config_for_offline_mode(env_cfg):
    """
    Patch environment configuration to use offline assets.

    This function walks through the environment config and patches known asset paths
    to use local storage when offline mode is enabled. It handles:
        - Robot USD paths
        - Terrain/ground plane paths
        - Sky light textures
        - Visualization markers

    Args:
        env_cfg: Environment configuration object (typically ManagerBasedRLEnvCfg)
    """
    if not is_offline_mode_enabled():
        return

    print("[OfflineAssetResolver] Patching configuration...")
    patches_made = 0

    # Patch robot USD path
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "robot"):
        if hasattr(env_cfg.scene.robot, "spawn") and hasattr(env_cfg.scene.robot.spawn, "usd_path"):
            original = env_cfg.scene.robot.spawn.usd_path
            resolved = resolve_asset_path(original)
            if resolved != original:
                env_cfg.scene.robot.spawn.usd_path = resolved
                patches_made += 1
                print("[OfflineAssetResolver]   ✓ Patched robot USD path")

    # Patch terrain USD path (if present)
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "terrain"):
        if hasattr(env_cfg.scene.terrain, "usd_path"):
            original = env_cfg.scene.terrain.usd_path
            resolved = resolve_asset_path(original)
            if resolved != original:
                env_cfg.scene.terrain.usd_path = resolved
                patches_made += 1
                print("[OfflineAssetResolver]   ✓ Patched terrain USD path")

    # Patch sky light texture
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "sky_light"):
        if hasattr(env_cfg.scene.sky_light, "spawn") and hasattr(env_cfg.scene.sky_light.spawn, "texture_file"):
            if env_cfg.scene.sky_light.spawn.texture_file:
                original = env_cfg.scene.sky_light.spawn.texture_file
                resolved = resolve_asset_path(original)
                if resolved != original:
                    env_cfg.scene.sky_light.spawn.texture_file = resolved
                    patches_made += 1
                    print("[OfflineAssetResolver]   ✓ Patched sky light texture")

    # Patch visualization markers
    if hasattr(env_cfg, "commands"):
        for command_name in dir(env_cfg.commands):
            if command_name.startswith("_"):
                continue

            command = getattr(env_cfg.commands, command_name, None)
            if not command:
                continue

            # Patch both current and goal velocity visualizers
            for viz_name in ["current_vel_visualizer_cfg", "goal_vel_visualizer_cfg"]:
                if not hasattr(command, viz_name):
                    continue

                visualizer = getattr(command, viz_name)
                if not hasattr(visualizer, "markers") or not isinstance(visualizer.markers, dict):
                    continue

                for marker_name, marker_cfg in visualizer.markers.items():
                    if hasattr(marker_cfg, "usd_path"):
                        original = marker_cfg.usd_path
                        resolved = resolve_asset_path(original)
                        if resolved != original:
                            marker_cfg.usd_path = resolved
                            patches_made += 1
                            print(f"[OfflineAssetResolver]   ✓ Patched {marker_name} in {viz_name}")

    if patches_made > 0:
        print(f"[OfflineAssetResolver] Patched {patches_made} asset paths")
    else:
        print("[OfflineAssetResolver] No paths needed patching (already correct)")


def install_path_hooks():
    """
    Install hooks into Isaac Lab's spawn configs for automatic path resolution.

    This function monkey-patches Isaac Lab's UsdFileCfg, GroundPlaneCfg, and
    PreviewSurfaceCfg classes to automatically resolve asset paths when they're
    instantiated. This provides transparent offline support without modifying
    environment configs.
    """
    try:
        import isaaclab.sim as sim_utils

        # Patch UsdFileCfg for general USD file spawning
        if hasattr(sim_utils, "UsdFileCfg"):
            original_usd_init = sim_utils.UsdFileCfg.__init__

            def patched_usd_init(self, *args, **kwargs):
                original_usd_init(self, *args, **kwargs)
                if hasattr(self, "usd_path") and is_offline_mode_enabled():
                    self.usd_path = resolve_asset_path(self.usd_path)

            sim_utils.UsdFileCfg.__init__ = patched_usd_init
            print("[OfflineAssetResolver] Installed UsdFileCfg path hook")

        # Patch GroundPlaneCfg for terrain/ground plane spawning
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

        # Patch PreviewSurfaceCfg for texture file resolution
        if hasattr(sim_utils, "PreviewSurfaceCfg"):
            original_surface_init = sim_utils.PreviewSurfaceCfg.__init__

            def patched_surface_init(self, *args, **kwargs):
                original_surface_init(self, *args, **kwargs)
                if hasattr(self, "texture_file") and is_offline_mode_enabled():
                    if self.texture_file:
                        self.texture_file = resolve_asset_path(self.texture_file)

            sim_utils.PreviewSurfaceCfg.__init__ = patched_surface_init
            print("[OfflineAssetResolver] Installed PreviewSurfaceCfg path hook")

    except ImportError:
        print("[OfflineAssetResolver] Could not install path hooks - isaaclab.sim not available")


def setup_offline_mode():
    """
    Set up offline mode with all hooks and path resolution.

    This is the main entry point for enabling offline training. Call this function
    at the start of your training script when the --offline flag is set.

    Example:
        if args_cli.offline:
            from isaaclab.utils import setup_offline_mode, patch_config_for_offline_mode
            setup_offline_mode()
            patch_config_for_offline_mode(env_cfg)
    """
    enable_offline_mode()
    install_path_hooks()
    print("[OfflineAssetResolver] Offline mode fully configured")


# Export public API
__all__ = [
    "enable_offline_mode",
    "disable_offline_mode",
    "is_offline_mode_enabled",
    "resolve_asset_path",
    "get_offline_assets_dir",
    "patch_config_for_offline_mode",
    "install_path_hooks",
    "setup_offline_mode",
]
