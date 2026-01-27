# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This module provides utilities to transparently redirect asset paths from Nucleus
to local storage when running in offline mode. It maintains the same directory structure
so that configs require no changes.

Usage:
    enable_offline_mode()
    
    All subsequent asset paths will be resolved to local_assets/
    path = resolve_asset_path(ISAACLAB_NUCLEUS_DIR + "/Robots/...")
    Returns: /path/to/isaaclab/local_assets/Robots/...
"""

import os
from typing import Optional

import carb.settings


class LocalAssetResolver:
    """
    Singleton class to manage local asset path resolution.
    
    When enabled, this resolver intercepts asset paths that point to Nucleus
    and redirects them to the local_assets directory.
    """
    
    _instance: Optional['LocalAssetResolver'] = None
    _enabled: bool = False
    _local_assets_dir: Optional[str] = None
    _nucleus_dir: Optional[str] = None
    _isaac_nucleus_dir: Optional[str] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalAssetResolver, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the resolver with environment paths."""
        # Get Isaac Lab root path
        self.isaaclab_path = os.environ.get('ISAACLAB_PATH', os.getcwd())
        
        # Set local assets directory
        self._local_assets_dir = os.path.join(self.isaaclab_path, "local_assets")
        
        # Get Nucleus directories from settings
        settings = carb.settings.get_settings()
        nucleus_root = settings.get("/persistent/isaac/asset_root/default")
        
        if nucleus_root:
            self._nucleus_dir = nucleus_root
            self._isaac_nucleus_dir = f"{nucleus_root}/Isaac"
            self._isaaclab_nucleus_dir = f"{nucleus_root}/Isaac/IsaacLab"
        
        print(f"[LocalAssetResolver] Initialized")
        print(f"  Local assets dir: {self._local_assets_dir}")
        if self._isaaclab_nucleus_dir:
            print(f"  Nucleus dir:      {self._isaaclab_nucleus_dir}")
    
    def enable(self):
        """Enable local asset resolution."""
        self._enabled = True
        print(f"[LocalAssetResolver] Local mode ENABLED")
        print(f"  All assets will be loaded from: {self._local_assets_dir}")
        
        # Verify local assets directory exists
        if not os.path.exists(self._local_assets_dir):
            print(f"[LocalAssetResolver] ⚠️  WARNING: Local assets directory not found!")
            print(f"  Please run: ./isaaclab.sh -p scripts/setup/download_assets.py")
    
    def disable(self):
        """Disable local asset resolution."""
        self._enabled = False
        print(f"[LocalAssetResolver] Local mode DISABLED")
    
    def is_enabled(self) -> bool:
        """Check if local mode is enabled."""
        return self._enabled
    
    def resolve_path(self, asset_path: str) -> str:
        """
        Resolve an asset path to either Nucleus or local storage.
        
        Args:
            asset_path: Original asset path (may contain Nucleus URL)
            
        Returns:
            Resolved path (local if enabled, otherwise original)
        """
        if not self._enabled:
            return asset_path
        
        # Skip if not a string or empty
        if not isinstance(asset_path, str) or not asset_path:
            return asset_path
        
        # Check if this is a Nucleus path we should redirect
        path_to_convert = None
        
        # Handle versioned paths like: .../Assets/Isaac/5.1/Isaac/IsaacLab/...
        import re
        
        # Pattern 1: Isaac Lab assets with version (e.g., .../Assets/Isaac/5.1/Isaac/IsaacLab/Robots/...)
        match = re.search(r'/Assets/Isaac/[\d.]+/Isaac/IsaacLab/(.+)$', asset_path)
        if match:
            path_to_convert = match.group(1)
        
        # Pattern 2: General Isaac assets with version (e.g., .../Assets/Isaac/5.1/Isaac/Props/...)
        if not path_to_convert:
            match = re.search(r'/Assets/Isaac/[\d.]+/Isaac/(?!IsaacLab)(.+)$', asset_path)
            if match:
                path_to_convert = match.group(1)
        
        # Pattern 3: Without version - IsaacLab specific (older format)
        if not path_to_convert and self._isaaclab_nucleus_dir and asset_path.startswith(self._isaaclab_nucleus_dir):
            path_to_convert = asset_path[len(self._isaaclab_nucleus_dir):].lstrip("/")
        
        # Pattern 4: Without version - General Isaac (older format)
        if not path_to_convert and self._isaac_nucleus_dir and asset_path.startswith(self._isaac_nucleus_dir):
            isaac_relative = asset_path[len(self._isaac_nucleus_dir):].lstrip("/")
            path_to_convert = isaac_relative
        
        # If we identified a path to convert, create the local path
        if path_to_convert:
            local_path = os.path.join(self._local_assets_dir, path_to_convert)
            
            # Verify the local file exists
            if os.path.exists(local_path):
                print(f"[LocalAssetResolver] ✓ Using local: {path_to_convert}")
                return local_path
            else:
                print(f"[LocalAssetResolver] ⚠️  Not found locally: {path_to_convert}")
                print(f"[LocalAssetResolver]    Falling back to Nucleus")
                return asset_path
        
        # If not a Nucleus path, return original
        return asset_path
    
    def get_local_assets_dir(self) -> str:
        """Get the local assets directory path."""
        return self._local_assets_dir


# Global resolver instance
_resolver = LocalAssetResolver()


def enable_local_mode():
    """Enable local asset resolution globally."""
    _resolver.enable()


def disable_local_mode():
    """Disable local asset resolution globally."""
    _resolver.disable()


def is_local_mode_enabled() -> bool:
    """Check if local mode is currently enabled."""
    return _resolver.is_enabled()


def resolve_asset_path(asset_path: str) -> str:
    """
    Resolve an asset path, redirecting to local storage if enabled.
    
    Args:
        asset_path: Original asset path (may contain Nucleus URL)
        
    Returns:
        Resolved path (local if mode is enabled and file exists, otherwise original)
    """
    return _resolver.resolve_path(asset_path)


def get_local_assets_dir() -> str:
    """Get the local assets directory path."""
    return _resolver.get_local_assets_dir()


def patch_config_for_local_mode(env_cfg):
    """
    Patch specific known paths in the environment config.
    
    Args:
        env_cfg: Environment configuration object
    """
    if not is_local_mode_enabled():
        return
    
    print("[LocalAssetResolver] Patching configuration...")
    patches_made = 0
    
    # Patch robot USD path
    if hasattr(env_cfg, 'scene') and hasattr(env_cfg.scene, 'robot'):
        if hasattr(env_cfg.scene.robot, 'spawn') and hasattr(env_cfg.scene.robot.spawn, 'usd_path'):
            original = env_cfg.scene.robot.spawn.usd_path
            resolved = resolve_asset_path(original)
            if resolved != original:
                env_cfg.scene.robot.spawn.usd_path = resolved
                patches_made += 1
                print(f"[LocalAssetResolver]   ✓ Patched robot USD path")
    
    # Patch terrain/ground plane paths
    if hasattr(env_cfg, 'scene') and hasattr(env_cfg.scene, 'terrain'):
        terrain_cfg = env_cfg.scene.terrain
        
        # Check for terrain_generator ground plane
        if hasattr(terrain_cfg, 'terrain_generator'):
            # This is typically procedural, no USD files needed
            pass
        
        # Check for direct ground plane USD
        if hasattr(terrain_cfg, 'usd_path'):
            original = terrain_cfg.usd_path
            resolved = resolve_asset_path(original)
            if resolved != original:
                terrain_cfg.usd_path = resolved
                patches_made += 1
                print(f"[LocalAssetResolver]   ✓ Patched terrain USD path")
    
    # Patch sky light textures
    if hasattr(env_cfg, 'scene') and hasattr(env_cfg.scene, 'sky_light'):
        if hasattr(env_cfg.scene.sky_light, 'spawn') and hasattr(env_cfg.scene.sky_light.spawn, 'texture_file'):
            if env_cfg.scene.sky_light.spawn.texture_file:
                original = env_cfg.scene.sky_light.spawn.texture_file
                resolved = resolve_asset_path(original)
                if resolved != original:
                    env_cfg.scene.sky_light.spawn.texture_file = resolved
                    patches_made += 1
                    print(f"[LocalAssetResolver]   ✓ Patched sky light texture")
    
    # Patch visualization markers (arrows, etc.)
    if hasattr(env_cfg, 'commands'):
        for command_name in dir(env_cfg.commands):
            if command_name.startswith('_'):
                continue
            command = getattr(env_cfg.commands, command_name, None)
            if command and hasattr(command, 'goal_vel_visualizer_cfg'):
                visualizer = command.goal_vel_visualizer_cfg
                if hasattr(visualizer, 'markers') and isinstance(visualizer.markers, dict):
                    for marker_name, marker_cfg in visualizer.markers.items():
                        if hasattr(marker_cfg, 'usd_path'):
                            original = marker_cfg.usd_path
                            resolved = resolve_asset_path(original)
                            if resolved != original:
                                marker_cfg.usd_path = resolved
                                patches_made += 1
                                print(f"[LocalAssetResolver]   ✓ Patched {marker_name} marker")

    if hasattr(env_cfg, 'commands'):
        for command_name in dir(env_cfg.commands):
            if command_name.startswith('_'):
                continue
            command = getattr(env_cfg.commands, command_name, None)
            if command:
                # Patch BOTH current and goal visualizers
                for viz_name in ['current_vel_visualizer_cfg', 'goal_vel_visualizer_cfg']:
                    if hasattr(command, viz_name):
                        visualizer = getattr(command, viz_name)
                        if hasattr(visualizer, 'markers') and isinstance(visualizer.markers, dict):
                            for marker_name, marker_cfg in visualizer.markers.items():
                                if hasattr(marker_cfg, 'usd_path'):
                                    original = marker_cfg.usd_path
                                    resolved = resolve_asset_path(original)
                                    if resolved != original:
                                        marker_cfg.usd_path = resolved
                                        patches_made += 1
                                        print(f"[LocalAssetResolver]   ✓ Patched {marker_name} in {viz_name}")
    
    if patches_made > 0:
        print(f"[LocalAssetResolver] Patched {patches_made} asset paths")
    else:
        print(f"[LocalAssetResolver] No paths needed patching (already correct)")

# Monkey patch common Isaac Lab modules to use local resolver
def install_path_hooks():
    """
    Install hooks into Isaac Lab's asset loading to automatically resolve paths.
    
    This patches the spawn configs to automatically resolve paths when local mode is enabled.
    """
    try:
        import isaaclab.sim as sim_utils
        
        # Patch UsdFileCfg
        if hasattr(sim_utils, 'UsdFileCfg'):
            original_usd_init = sim_utils.UsdFileCfg.__init__
            
            def patched_usd_init(self, *args, **kwargs):
                # Call original init
                original_usd_init(self, *args, **kwargs)
                # Resolve the usd_path if local mode is enabled
                if hasattr(self, 'usd_path') and is_local_mode_enabled():
                    self.usd_path = resolve_asset_path(self.usd_path)
            
            sim_utils.UsdFileCfg.__init__ = patched_usd_init
            print("[LocalAssetResolver] Installed UsdFileCfg path hook")
        
        # Patch GroundPlaneCfg (for terrain)
        if hasattr(sim_utils, 'GroundPlaneCfg'):
            original_ground_init = sim_utils.GroundPlaneCfg.__init__
            
            def patched_ground_init(self, *args, **kwargs):
                # Call original init
                original_ground_init(self, *args, **kwargs)
                # Resolve the usd_path if local mode is enabled
                if hasattr(self, 'usd_path') and is_local_mode_enabled():
                    original_path = self.usd_path
                    self.usd_path = resolve_asset_path(self.usd_path)
                    if self.usd_path != original_path:
                        print(f"[LocalAssetResolver] ✓ Resolved ground plane: {os.path.basename(self.usd_path)}")
            
            sim_utils.GroundPlaneCfg.__init__ = patched_ground_init
            print("[LocalAssetResolver] Installed GroundPlaneCfg path hook")
        
        # Patch PreviewSurfaceCfg for textures
        if hasattr(sim_utils, 'PreviewSurfaceCfg'):
            original_surface_init = sim_utils.PreviewSurfaceCfg.__init__
            
            def patched_surface_init(self, *args, **kwargs):
                original_surface_init(self, *args, **kwargs)
                if hasattr(self, 'texture_file') and is_local_mode_enabled():
                    if self.texture_file:
                        self.texture_file = resolve_asset_path(self.texture_file)
            
            sim_utils.PreviewSurfaceCfg.__init__ = patched_surface_init
            print("[LocalAssetResolver] Installed PreviewSurfaceCfg path hook")
            
    except ImportError:
        print("[LocalAssetResolver] Could not install path hooks - isaaclab.sim not available")


# Set up local mode with all hooks
def setup_local_mode():
    enable_local_mode()
    install_path_hooks()
    print("[LocalAssetResolver] Local mode fully configured")