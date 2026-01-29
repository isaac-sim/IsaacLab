# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Offline Asset Resolver for Isaac Lab.

This module redirects Nucleus asset paths to a local mirror of the Isaac/ directory.

Path Resolution:
    Nucleus URL: .../Assets/Isaac/5.1/Isaac/{path}
    Local Path:  offline_assets/{path}

    Examples:
    - /Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd → offline_assets/IsaacLab/Robots/Unitree/Go2/go2.usd
    - /Isaac/Props/UIElements/arrow_x.usd        → offline_assets/Props/UIElements/arrow_x.usd

Usage:
    Automatically enabled via AppLauncher --offline flag, or manually:

    from isaaclab.utils import setup_offline_mode
    setup_offline_mode()
"""

from __future__ import annotations

import os
import re


class OfflineAssetResolver:
    """Singleton class to manage offline asset path resolution."""

    _instance: OfflineAssetResolver | None = None
    _enabled: bool = False
    _strict: bool = True  # If True, fail immediately when asset not found locally
    _initialized: bool = False
    _hooks_installed: bool = False
    _offline_assets_dir: str | None = None
    _failed_assets: list[tuple[str, str]] = []
    _warned_assets: set[str] = set()  # Track assets we've already warned about

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._failed_assets = []
            cls._instance._warned_assets = set()
        return cls._instance

    def _initialize(self):
        """Initialize the resolver."""
        if self._initialized:
            return

        self.isaaclab_path = os.environ.get("ISAACLAB_PATH", os.getcwd())
        self._offline_assets_dir = os.path.join(self.isaaclab_path, "offline_assets")
        self._failed_assets = []
        self._warned_assets = set()
        self._initialized = True

        print("[OfflineAssetResolver] Initialized")
        print(f"  Offline assets: {self._offline_assets_dir}")

    def enable(self, strict: bool = True):
        """
        Enable offline asset resolution.

        Args:
            strict: If True (default), raise an error when an asset is not found locally.
                   If False, fall back to Nucleus (may cause timeouts if offline).
        """
        self._initialize()
        self._enabled = True
        self._strict = strict
        self._failed_assets = []
        self._warned_assets = set()

        mode = "STRICT" if strict else "permissive"
        print(f"[OfflineAssetResolver] Offline mode ENABLED ({mode})")
        print(f"  Local mirror of Isaac/ at: {self._offline_assets_dir}")

        if strict:
            print("  Missing assets will cause an error (no Nucleus fallback)")
        else:
            print("  Missing assets will fall back to Nucleus (may timeout if offline)")

        if not os.path.exists(self._offline_assets_dir):
            print("[OfflineAssetResolver] ⚠️  WARNING: offline_assets directory not found!")
            print("  Run: ./isaaclab.sh -p scripts/offline_setup/download_assets.py")

    def disable(self):
        """Disable offline asset resolution."""
        self._enabled = False
        print("[OfflineAssetResolver] Offline mode DISABLED")

    def is_enabled(self) -> bool:
        return self._enabled

    def is_strict(self) -> bool:
        return self._strict

    def are_hooks_installed(self) -> bool:
        return self._hooks_installed

    def set_hooks_installed(self, value: bool = True):
        self._hooks_installed = value

    def add_failed_asset(self, original_path: str, relative_path: str):
        """Track a failed asset resolution."""
        for _, rel in self._failed_assets:
            if rel == relative_path:
                return
        self._failed_assets.append((original_path, relative_path))

    def get_failed_assets(self) -> list[tuple[str, str]]:
        """Get list of (original_path, relative_path) for failed resolutions."""
        return self._failed_assets.copy()

    def resolve_path(self, asset_path: str) -> str:
        """
        Resolve an asset path to offline storage.
        """
        if not self._enabled or not isinstance(asset_path, str) or not asset_path:
            return asset_path

        # Skip if already a local path
        if os.path.exists(asset_path):
            return asset_path

        # Extract the path relative to /Isaac/
        relative_path = self._extract_relative_path(asset_path)

        if relative_path:
            offline_path = os.path.join(self._offline_assets_dir, relative_path)

            # Try exact path first
            if os.path.exists(offline_path):
                print(f"[OfflineAssetResolver] ✓ Using offline: {relative_path}")
                return offline_path

            # Try case-insensitive fallback
            resolved = self._find_case_insensitive(relative_path)
            if resolved:
                print(f"[OfflineAssetResolver] ✓ Using offline (case-adjusted): {resolved}")
                return os.path.join(self._offline_assets_dir, resolved)

            # Asset not found locally
            self._handle_missing_asset(asset_path, relative_path)

        return asset_path

    def _handle_missing_asset(self, original_path: str, relative_path: str):
        """Handle a missing asset - either error (strict) or warn (permissive)."""
        # Only warn once per asset
        if relative_path in self._warned_assets:
            return
        self._warned_assets.add(relative_path)
        self.add_failed_asset(original_path, relative_path)

        # Build download command
        parts = relative_path.split("/")
        if len(parts) >= 2 and parts[0] == "IsaacLab":
            download_cmd = f"./isaaclab.sh -p scripts/offline_setup/download_assets.py --categories IsaacLab/{parts[1]}"
        else:
            download_cmd = f"./isaaclab.sh -p scripts/offline_setup/download_assets.py --categories {parts[0]}"

        if self._strict:
            # Strict mode: raise an error
            error_msg = f"""
{"=" * 70}
[OfflineAssetResolver] ✗ ASSET NOT FOUND (offline mode)
{"=" * 70}
Missing:  {relative_path}
Expected: {os.path.join(self._offline_assets_dir, relative_path)}

To download this asset, run:
  {download_cmd}

Or to allow Nucleus fallback (may timeout if offline), use:
  --offline-permissive instead of --offline
{"=" * 70}
"""
            raise FileNotFoundError(error_msg)
        # Permissive mode: warn and fall back to Nucleus
        print(f"\n[OfflineAssetResolver] ⚠️  MISSING: {relative_path}")
        print("[OfflineAssetResolver]   Falling back to Nucleus (may timeout if offline)")
        print(f"[OfflineAssetResolver]   Download with: {download_cmd}\n")

    def _extract_relative_path(self, asset_path: str) -> str | None:
        """Extract the path relative to /Isaac/ from a Nucleus URL."""
        match = re.search(r"/Assets/Isaac/[\d.]+/Isaac/(.+)$", asset_path)
        if match:
            return match.group(1)
        return None

    def _find_case_insensitive(self, relative_path: str) -> str | None:
        """Find a file using case-insensitive matching."""
        parts = relative_path.replace("\\", "/").split("/")
        current_dir = self._offline_assets_dir
        resolved_parts = []

        for part in parts:
            if not os.path.exists(current_dir):
                return None

            try:
                entries = os.listdir(current_dir)
            except (OSError, PermissionError):
                return None

            if part in entries:
                resolved_parts.append(part)
                current_dir = os.path.join(current_dir, part)
                continue

            part_lower = part.lower()
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

        final_path = os.path.join(self._offline_assets_dir, *resolved_parts)
        if os.path.exists(final_path):
            return "/".join(resolved_parts)

        return None

    def get_offline_assets_dir(self) -> str:
        self._initialize()
        return self._offline_assets_dir


# Global resolver instance
_resolver = OfflineAssetResolver()


# =============================================================================
# Public API Functions
# =============================================================================


def enable_offline_mode(strict: bool = True):
    """
    Enable offline asset resolution.

    Args:
        strict: If True (default), raise an error when an asset is not found locally.
               If False, fall back to Nucleus (may cause timeouts if offline).
    """
    _resolver.enable(strict=strict)


def disable_offline_mode():
    _resolver.disable()


def is_offline_mode_enabled() -> bool:
    return _resolver.is_enabled()


def is_offline_mode_strict() -> bool:
    return _resolver.is_strict()


def resolve_asset_path(asset_path: str) -> str:
    return _resolver.resolve_path(asset_path)


def get_offline_assets_dir() -> str:
    return _resolver.get_offline_assets_dir()


def get_failed_assets() -> list[tuple[str, str]]:
    return _resolver.get_failed_assets()


def _install_spawn_hooks():
    """Install hooks on spawn functions (Level 1)."""
    try:
        from isaaclab.sim.spawners.from_files import from_files as from_files_module

        if hasattr(from_files_module, "spawn_from_usd"):
            original_spawn_from_usd = from_files_module.spawn_from_usd

            def patched_spawn_from_usd(prim_path, cfg, *args, **kwargs):
                if is_offline_mode_enabled() and hasattr(cfg, "usd_path"):
                    cfg.usd_path = resolve_asset_path(cfg.usd_path)
                return original_spawn_from_usd(prim_path, cfg, *args, **kwargs)

            from_files_module.spawn_from_usd = patched_spawn_from_usd
            print("[OfflineAssetResolver] Installed spawn_from_usd hook")

        if hasattr(from_files_module, "spawn_from_urdf"):
            original_spawn_from_urdf = from_files_module.spawn_from_urdf

            def patched_spawn_from_urdf(prim_path, cfg, *args, **kwargs):
                if is_offline_mode_enabled() and hasattr(cfg, "asset_path"):
                    cfg.asset_path = resolve_asset_path(cfg.asset_path)
                return original_spawn_from_urdf(prim_path, cfg, *args, **kwargs)

            from_files_module.spawn_from_urdf = patched_spawn_from_urdf
            print("[OfflineAssetResolver] Installed spawn_from_urdf hook")

    except ImportError as e:
        print(f"[OfflineAssetResolver] Could not install spawn hooks: {e}")


def _install_file_hooks():
    """Install hooks on file loading functions (Level 2)."""
    try:
        import isaaclab.utils.assets as assets_module

        if hasattr(assets_module, "read_file"):
            original_read_file = assets_module.read_file

            def patched_read_file(path: str):
                if is_offline_mode_enabled():
                    return original_read_file(resolve_asset_path(path))
                return original_read_file(path)

            assets_module.read_file = patched_read_file
            print("[OfflineAssetResolver] Installed read_file hook")

        if hasattr(assets_module, "retrieve_file_path"):
            original_retrieve = assets_module.retrieve_file_path

            def patched_retrieve_file_path(path: str, *args, **kwargs):
                if is_offline_mode_enabled():
                    return original_retrieve(resolve_asset_path(path), *args, **kwargs)
                return original_retrieve(path, *args, **kwargs)

            assets_module.retrieve_file_path = patched_retrieve_file_path
            print("[OfflineAssetResolver] Installed retrieve_file_path hook")

        if hasattr(assets_module, "check_file_path"):
            original_check_file_path = assets_module.check_file_path

            def patched_check_file_path(path: str, *args, **kwargs):
                if is_offline_mode_enabled():
                    return original_check_file_path(resolve_asset_path(path), *args, **kwargs)
                return original_check_file_path(path, *args, **kwargs)

            assets_module.check_file_path = patched_check_file_path
            print("[OfflineAssetResolver] Installed check_file_path hook")

    except ImportError:
        pass


def _install_config_hooks():
    """Install hooks on config classes (Level 3)."""
    try:
        import isaaclab.sim as sim_utils

        if hasattr(sim_utils, "UsdFileCfg"):
            original_usd_init = sim_utils.UsdFileCfg.__init__

            def patched_usd_init(self, *args, **kwargs):
                original_usd_init(self, *args, **kwargs)
                if hasattr(self, "usd_path") and is_offline_mode_enabled():
                    self.usd_path = resolve_asset_path(self.usd_path)

            sim_utils.UsdFileCfg.__init__ = patched_usd_init
            print("[OfflineAssetResolver] Installed UsdFileCfg hook")

        if hasattr(sim_utils, "GroundPlaneCfg"):
            original_ground_init = sim_utils.GroundPlaneCfg.__init__

            def patched_ground_init(self, *args, **kwargs):
                original_ground_init(self, *args, **kwargs)
                if hasattr(self, "usd_path") and is_offline_mode_enabled():
                    self.usd_path = resolve_asset_path(self.usd_path)

            sim_utils.GroundPlaneCfg.__init__ = patched_ground_init
            print("[OfflineAssetResolver] Installed GroundPlaneCfg hook")

        if hasattr(sim_utils, "PreviewSurfaceCfg"):
            original_surface_init = sim_utils.PreviewSurfaceCfg.__init__

            def patched_surface_init(self, *args, **kwargs):
                original_surface_init(self, *args, **kwargs)
                if hasattr(self, "texture_file") and is_offline_mode_enabled() and self.texture_file:
                    self.texture_file = resolve_asset_path(self.texture_file)

            sim_utils.PreviewSurfaceCfg.__init__ = patched_surface_init
            print("[OfflineAssetResolver] Installed PreviewSurfaceCfg hook")

    except ImportError:
        pass


def _install_env_hooks():
    """Install hooks on environment classes (Level 4)."""
    try:
        from isaaclab.envs import ManagerBasedEnv

        original_manager_init = ManagerBasedEnv.__init__

        def patched_manager_init(self, cfg, *args, **kwargs):
            if is_offline_mode_enabled():
                patch_config_for_offline_mode(cfg)
            original_manager_init(self, cfg, *args, **kwargs)

        ManagerBasedEnv.__init__ = patched_manager_init
        print("[OfflineAssetResolver] Installed ManagerBasedEnv hook")
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
        print("[OfflineAssetResolver] Installed DirectRLEnv hook")
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
        print("[OfflineAssetResolver] Installed DirectMARLEnv hook")
    except ImportError:
        pass


def _install_asset_hooks():
    """Install hooks on asset classes (Level 5)."""
    try:
        from isaaclab.assets import Articulation

        original_articulation_init = Articulation.__init__

        def patched_articulation_init(self, cfg, *args, **kwargs):
            if is_offline_mode_enabled():
                patch_config_for_offline_mode(cfg)
            original_articulation_init(self, cfg, *args, **kwargs)

        Articulation.__init__ = patched_articulation_init
        print("[OfflineAssetResolver] Installed Articulation hook")
    except ImportError:
        pass

    try:
        from isaaclab.assets import RigidObject

        original_rigid_init = RigidObject.__init__

        def patched_rigid_init(self, cfg, *args, **kwargs):
            if is_offline_mode_enabled():
                patch_config_for_offline_mode(cfg)
            original_rigid_init(self, cfg, *args, **kwargs)

        RigidObject.__init__ = patched_rigid_init
        print("[OfflineAssetResolver] Installed RigidObject hook")
    except ImportError:
        pass


def install_hooks():
    """Install ALL path resolution hooks at multiple levels."""
    if _resolver.are_hooks_installed():
        return

    _install_spawn_hooks()
    _install_file_hooks()
    _install_config_hooks()
    _install_env_hooks()
    _install_asset_hooks()

    _resolver.set_hooks_installed(True)


def _is_nucleus_path(path: str) -> bool:
    """Check if a path is a Nucleus/S3 URL."""
    if not isinstance(path, str):
        return False
    return (
        "omniverse-content-production" in path
        or "nucleus" in path.lower()
        or path.startswith("omniverse://")
        or "/Assets/Isaac/" in path
    )


def _patch_object_recursive(obj, visited: set, depth: int = 0, max_depth: int = 15) -> int:
    """Recursively patch asset paths in an object."""
    if depth > max_depth:
        return 0

    obj_id = id(obj)
    if obj_id in visited:
        return 0
    visited.add(obj_id)

    patches = 0

    if obj is None or isinstance(obj, (str, int, float, bool, bytes)):
        return 0

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

    if hasattr(obj, "__dict__"):
        asset_attrs = [
            "usd_path",
            "texture_file",
            "asset_path",
            "file_path",
            "mesh_file",
            "network_file",
            "policy_path",
            "checkpoint_path",
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
                            pass

        try:
            for attr_name in dir(obj):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr_value = getattr(obj, attr_name)
                    if callable(attr_value) and not hasattr(attr_value, "__dict__"):
                        continue
                    patches += _patch_object_recursive(attr_value, visited, depth + 1, max_depth)
                except (AttributeError, TypeError, RuntimeError):
                    continue
        except (TypeError, RuntimeError):
            pass

    return patches


def patch_config_for_offline_mode(cfg):
    """Patch a configuration object to use offline assets."""
    if not is_offline_mode_enabled():
        return

    visited = set()
    patches = _patch_object_recursive(cfg, visited)

    if patches > 0:
        print(f"[OfflineAssetResolver] Patched {patches} config paths")


def setup_offline_mode(strict: bool = True):
    """
    Set up offline mode with all hooks.

    Args:
        strict: If True (default), raise an error when an asset is not found locally.
               If False, fall back to Nucleus (may cause timeouts if offline).
    """
    enable_offline_mode(strict=strict)
    install_hooks()
    print("[OfflineAssetResolver] Offline mode fully configured")


def print_summary():
    """Print summary of any failed asset resolutions."""
    failed = get_failed_assets()
    if failed:
        print("\n" + "=" * 70)
        print("[OfflineAssetResolver] ⚠️  MISSING ASSETS SUMMARY")
        print("=" * 70)
        print(f"The following {len(failed)} asset(s) were not found locally:\n")

        categories = {}
        for original, relative in failed:
            parts = relative.split("/")
            if len(parts) >= 2 and parts[0] == "IsaacLab":
                cat = f"IsaacLab/{parts[1]}"
            else:
                cat = parts[0] if parts else "Unknown"

            if cat not in categories:
                categories[cat] = []
            categories[cat].append(relative)

        for cat, assets in sorted(categories.items()):
            print(f"  {cat}:")
            for asset in assets:
                print(f"    - {asset}")

        print("\nTo download missing assets, run:")
        for cat in sorted(categories.keys()):
            print(f"  ./isaaclab.sh -p scripts/offline_setup/download_assets.py --categories {cat}")
        print("=" * 70 + "\n")
    else:
        print("[OfflineAssetResolver] ✓ All assets resolved successfully")


# Legacy aliases
def install_spawn_hooks():
    install_hooks()


def install_file_hooks():
    install_hooks()


def install_env_hooks():
    install_hooks()


def install_path_hooks():
    install_hooks()


__all__ = [
    "enable_offline_mode",
    "disable_offline_mode",
    "is_offline_mode_enabled",
    "is_offline_mode_strict",
    "resolve_asset_path",
    "get_offline_assets_dir",
    "get_failed_assets",
    "patch_config_for_offline_mode",
    "install_hooks",
    "setup_offline_mode",
    "print_summary",
    "install_spawn_hooks",
    "install_file_hooks",
    "install_env_hooks",
    "install_path_hooks",
]
