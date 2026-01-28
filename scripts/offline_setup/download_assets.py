# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Download Isaac Lab assets from Nucleus server for offline training.

This script downloads assets from the Nucleus server and mirrors the directory structure
locally in the offline_assets folder. This enables offline training without requiring
internet connectivity.

Usage:
    # Download all assets
    ./isaaclab.sh -p scripts/offline_setup/download_assets.py --categories all

    # Download specific categories
    ./isaaclab.sh -p scripts/offline_setup/download_assets.py --categories Robots Props

    # Download specific robot subset
    ./isaaclab.sh -p scripts/offline_setup/download_assets.py --categories Robots --subset Unitree

Available Categories:
    Props, Robots, Environments, Materials, Controllers, ActuatorNets, Policies, Mimic
"""

import argparse
import os
from pathlib import Path

from tqdm import tqdm

from isaaclab.app import AppLauncher

# Initialize Isaac Sim environment
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import carb
import omni.client

# Get Isaac Lab paths
ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", os.getcwd())
OFFLINE_ASSETS_DIR = os.path.join(ISAACLAB_PATH, "offline_assets")

# Get the Nucleus directory from settings
settings = carb.settings.get_settings()
NUCLEUS_ASSET_ROOT = settings.get("/persistent/isaac/asset_root/default")
ISAAC_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT}/Isaac"  # General Isaac Sim assets
ISAACLAB_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT}/Isaac/IsaacLab"  # Isaac Lab specific assets

# Asset categories and their locations
ASSET_CATEGORIES = {
    # General Isaac Sim assets (Isaac/)
    "Props": {"desc": "Props, objects, markers, and mounts", "base": ISAAC_NUCLEUS_DIR},
    "Robots": {"desc": "Robot USD files and configurations", "base": ISAAC_NUCLEUS_DIR},
    "Environments": {"desc": "Environment assets and terrains", "base": ISAAC_NUCLEUS_DIR},
    "Materials": {"desc": "Materials and textures including sky HDRs", "base": ISAAC_NUCLEUS_DIR},
    # Isaac Lab specific assets (Isaac/IsaacLab/)
    "Controllers": {"desc": "IK controllers and kinematics assets", "base": ISAACLAB_NUCLEUS_DIR},
    "ActuatorNets": {"desc": "Actuator network models", "base": ISAACLAB_NUCLEUS_DIR},
    "Policies": {"desc": "Pre-trained policy checkpoints", "base": ISAACLAB_NUCLEUS_DIR},
    "Mimic": {"desc": "Demonstration and imitation learning assets", "base": ISAACLAB_NUCLEUS_DIR},
}


def format_size(bytes_size: int) -> str:
    """
    Format bytes into human-readable size.

    Args:
        bytes_size: Size in bytes

    Returns:
        Human-readable size string (e.g., "1.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def get_local_directory_size(path: str) -> int:
    """
    Calculate total size of a local directory.

    Args:
        path: Local directory path

    Returns:
        Total size in bytes
    """
    total_size = 0
    if os.path.exists(path):
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    return total_size


def get_remote_directory_info(remote_path: str) -> tuple[int, int]:
    """
    Get file count and total size of a remote Nucleus directory.

    Args:
        remote_path: Nucleus directory URL

    Returns:
        Tuple of (file_count, total_size_bytes)
    """
    file_count = 0
    total_size = 0

    result, entries = omni.client.list(remote_path)
    if result != omni.client.Result.OK:
        return 0, 0

    for entry in entries:
        is_dir = entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN
        remote_item = f"{remote_path}/{entry.relative_path}"

        if is_dir:
            # Recursively get info from subdirectory
            sub_count, sub_size = get_remote_directory_info(remote_item)
            file_count += sub_count
            total_size += sub_size
        else:
            # Get file size
            file_count += 1
            stat_result, stat_entry = omni.client.stat(remote_item)
            if stat_result == omni.client.Result.OK:
                total_size += stat_entry.size

    return file_count, total_size


def ensure_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def download_file(remote_path: str, local_path: str) -> bool:
    """
    Download a single file from Nucleus to local storage.

    Args:
        remote_path: Full Nucleus URL
        local_path: Local file system path

    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        result = omni.client.copy(remote_path, local_path)
        return result == omni.client.Result.OK
    except Exception as e:
        print(f"Error downloading {remote_path}: {e}")
        return False


def download_directory_recursive(remote_path: str, local_base: str, progress_bar) -> None:
    """
    Recursively download a directory from Nucleus to local storage.

    Args:
        remote_path: Nucleus directory URL
        local_base: Local directory to mirror structure
        progress_bar: tqdm progress bar instance
    """
    result, entries = omni.client.list(remote_path)
    if result != omni.client.Result.OK:
        return

    for entry in entries:
        is_dir = entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN
        remote_item = f"{remote_path}/{entry.relative_path}"
        local_item = os.path.join(local_base, entry.relative_path)

        if is_dir:
            ensure_directory(local_item)
            download_directory_recursive(remote_item, local_item, progress_bar)
        else:
            progress_bar.set_description(f"Downloading {entry.relative_path[:50]}")
            download_file(remote_item, local_item)
            progress_bar.update(1)


def download_asset_category(category: str, subset: str = None) -> None:
    """
    Download all assets in a specific category.

    Args:
        category: Asset category (e.g., "Robots", "Props")
        subset: Optional subset filter (e.g., specific robot name)
    """
    category_info = ASSET_CATEGORIES[category]
    base_path = category_info["base"]
    description = category_info["desc"]

    remote_dir = f"{base_path}/{category}"
    local_dir = os.path.join(OFFLINE_ASSETS_DIR, category)

    # Apply subset filter if specified
    if subset and category == "Robots":
        remote_dir = f"{remote_dir}/{subset}"
        local_dir = os.path.join(local_dir, subset)

    print(f"\n{'=' * 70}")
    print(f"📦 {category}: {description}")
    print(f"{'=' * 70}")
    print(f"Source: {remote_dir}")
    print(f"Target: {local_dir}")

    # Check if remote directory exists
    result, _ = omni.client.stat(remote_dir)
    if result != omni.client.Result.OK:
        print(f"⚠️  Directory not found: {remote_dir}")
        print("    This category may not be available or may be in a different location.")
        return

    # Count files and get size
    print("📊 Analyzing remote directory...")
    file_count, total_size = get_remote_directory_info(remote_dir)

    if file_count == 0:
        print("✓ No files to download")
        return

    print(f"   Files: {file_count:,}")
    print(f"   Size:  {format_size(total_size)}")

    # Download with progress bar
    ensure_directory(local_dir)
    with tqdm(total=file_count, unit="file", desc="Progress") as pbar:
        download_directory_recursive(remote_dir, local_dir, pbar)

    print(f"✓ Completed {category}")


def verify_downloads() -> None:
    """Display summary of downloaded assets."""
    print("\n" + "=" * 70)
    print("📊 Downloaded Assets Summary")
    print("=" * 70)

    total_size = 0
    total_files = 0

    for category in ASSET_CATEGORIES.keys():
        local_dir = os.path.join(OFFLINE_ASSETS_DIR, category)
        if os.path.exists(local_dir):
            size = get_local_directory_size(local_dir)
            files = sum(1 for _ in Path(local_dir).rglob("*") if _.is_file())
            total_size += size
            total_files += files
            print(f"✓ {category:<15} {files:>6,} files    {format_size(size):>10}")
        else:
            print(f"✗ {category:<15} Not found")

    print("=" * 70)
    print(f"{'TOTAL':<15} {total_files:>6,} files    {format_size(total_size):>10}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Download Isaac Lab assets from Nucleus to local storage for offline training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(ASSET_CATEGORIES.keys()) + ["all"],
        default=["all"],
        help="Asset categories to download (default: all)",
    )
    parser.add_argument(
        "--subset", type=str, help="Download only specific subset (e.g., 'ANYbotics' or 'Unitree' for robots)"
    )
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing downloads without downloading")

    args = parser.parse_args()

    try:
        print("\n" + "=" * 70)
        print("🚀 Isaac Lab Offline Asset Downloader")
        print("=" * 70)
        print(f"Isaac Sim Assets:  {ISAAC_NUCLEUS_DIR}")
        print(f"Isaac Lab Assets:  {ISAACLAB_NUCLEUS_DIR}")
        print(f"Local Target:      {OFFLINE_ASSETS_DIR}")
        print("=" * 70)

        if args.verify_only:
            verify_downloads()
            return

        # Determine which categories to download
        categories = list(ASSET_CATEGORIES.keys()) if "all" in args.categories else args.categories

        print(f"\n📋 Selected Categories: {', '.join(categories)}")
        if args.subset:
            print(f"🔍 Subset Filter: {args.subset}")

        # Calculate total download size
        print("\n📊 Calculating download size...")
        total_files = 0
        total_size = 0

        for category in categories:
            category_info = ASSET_CATEGORIES[category]
            base_path = category_info["base"]
            remote_dir = f"{base_path}/{category}"

            # Apply subset filter
            if args.subset and category == "Robots":
                remote_dir = f"{remote_dir}/{args.subset}"

            # Check if directory exists
            result, _ = omni.client.stat(remote_dir)
            if result == omni.client.Result.OK:
                files, size = get_remote_directory_info(remote_dir)
                total_files += files
                total_size += size
                print(f"   {category}: {files:,} files ({format_size(size)})")

        print("\n" + "=" * 70)
        print(f"📦 Total Download: {total_files:,} files ({format_size(total_size)})")
        print("=" * 70)

        # Confirm before proceeding
        response = input("\nProceed with download? [y/N]: ")
        if response.lower() not in ["y", "yes"]:
            print("❌ Download cancelled")
            return

        # Download each category
        print("\n🔽 Starting download...")
        for category in categories:
            try:
                download_asset_category(category, args.subset)
            except KeyboardInterrupt:
                print("\n\n⚠️  Download interrupted by user")
                raise
            except Exception as e:
                print(f"\n❌ Error downloading {category}: {e}")
                continue

        # Show final summary
        verify_downloads()

        print("\n" + "=" * 70)
        print("✅ Download Complete!")
        print("=" * 70)
        print(f"\nOffline assets are available in: {OFFLINE_ASSETS_DIR}")
        print("\n💡 Usage: Add --offline flag to your training commands")
        print("\nExample:")
        print("  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\")
        print("      --task Isaac-Velocity-Flat-Unitree-Go2-v0 \\")
        print("      --num_envs 128 \\")
        print("      --offline\n")

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
