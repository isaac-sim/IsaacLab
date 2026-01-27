# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script downloads all assets from Nucleus server (ISAACLAB_NUCLEUS_DIR) and mirrors 
the directory structure locally in the local_assets folder. This enables offline/local training
without requiring S3/Nucleus connectivity. Must be connected to the internet to use!

Usage:
    ./isaaclab.sh -p scripts/setup/download_assets.py --categories Props Robots Environments Materials Controllers ActuatorNets Policies Mimic
    ./isaaclab.sh -p scripts/setup/download_assets.py [--subset ROBOT_NAME]
"""

import argparse
import os
from pathlib import Path

# Initialize Isaac Sim app first to get access to omni modules
from isaaclab.app import AppLauncher

# Create minimal app launcher to initialize Isaac Sim environment
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import carb
import omni.client

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

# Get Isaac Lab paths
ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", os.getcwd())
LOCAL_ASSETS_DIR = os.path.join(ISAACLAB_PATH, "local_assets")

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


def ensure_local_directory(local_path: str) -> None:
    """Create local directory if it doesn't exist."""
    os.makedirs(local_path, exist_ok=True)


def download_file(remote_path: str, local_path: str) -> bool:
    """
    Download a single file from Nucleus to local storage.
    
    Args:
        remote_path: Full Nucleus URL (e.g., omniverse://...)
        local_path: Local file system path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create parent directory if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Copy file from Nucleus to local
        result = omni.client.copy(remote_path, local_path)
        return result == omni.client.Result.OK
    except Exception as e:
        print(f"Error downloading {remote_path}: {e}")
        return False


def list_nucleus_directory(remote_path: str) -> list[tuple[str, bool]]:
    """
    List all files and directories in a Nucleus path.
    
    Args:
        remote_path: Nucleus directory URL
        
    Returns:
        List of (item_name, is_directory) tuples
    """
    result, entries = omni.client.list(remote_path)
    if result != omni.client.Result.OK:
        return []
    
    items = []
    for entry in entries:
        is_dir = entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN
        items.append((entry.relative_path, bool(is_dir)))
    return items


def download_directory_recursive(remote_path: str, local_base: str, progress_bar=None) -> None:
    """
    Recursively download a directory from Nucleus to local storage.
    
    Args:
        remote_path: Nucleus directory URL
        local_base: Local directory to mirror structure
        progress_bar: Optional tqdm progress bar (or None)
    """
    items = list_nucleus_directory(remote_path)
    
    for item_name, is_directory in items:
        remote_item = f"{remote_path}/{item_name}"
        local_item = os.path.join(local_base, item_name)
        
        if is_directory:
            # Recursively download subdirectory
            ensure_local_directory(local_item)
            download_directory_recursive(remote_item, local_item, progress_bar)
        else:
            # Download file
            if progress_bar is not None:
                progress_bar.set_description(f"Downloading {item_name}")
            else:
                print(f"  Downloading: {item_name}")
            download_file(remote_item, local_item)
            if progress_bar is not None:
                progress_bar.update(1)


def count_files_recursive(remote_path: str) -> int:
    """Count total files in a directory tree for progress tracking."""
    count = 0
    items = list_nucleus_directory(remote_path)
    
    for item_name, is_directory in items:
        if is_directory:
            count += count_files_recursive(f"{remote_path}/{item_name}")
        else:
            count += 1
    return count


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
    local_dir = os.path.join(LOCAL_ASSETS_DIR, category)
    
    print(f"\n{'='*60}")
    print(f"Downloading {category}: {description}")
    print(f"From: {remote_dir}")
    print(f"To:   {local_dir}")
    print(f"{'='*60}")
    
    # If subset is specified, only download that subset
    if subset and category == "Robots":
        remote_dir = f"{remote_dir}/{subset}"
        local_dir = os.path.join(local_dir, subset)
        print(f"Filtering to subset: {subset}")
    
    # Check if remote directory exists
    result, _ = omni.client.stat(remote_dir)
    if result != omni.client.Result.OK:
        print(f"⚠️  Directory not found: {remote_dir}")
        print(f"    This category may not be available or may be in a different location.")
        return
    
    # Count files for progress bar
    print("Counting files...")
    total_files = count_files_recursive(remote_dir)
    print(f"Found {total_files} files to download")
    
    if total_files == 0:
        print("No files to download")
        return
    
    # Download with progress bar
    ensure_local_directory(local_dir)
    if HAS_TQDM:
        with tqdm(total=total_files, unit="file") as pbar:
            download_directory_recursive(remote_dir, local_dir, pbar)
    else:
        print(f"Downloading {total_files} files (install tqdm for progress bars)...")
        download_directory_recursive(remote_dir, local_dir, None)
    
    print(f"✓ Completed {category}")


def verify_downloads(category: str = None) -> None:
    """Verify that local assets directory has expected structure."""
    print("\n" + "="*60)
    print("Verifying local assets...")
    print("="*60)
    
    categories_to_check = [category] if category else ASSET_CATEGORIES.keys()
    
    for cat in categories_to_check:
        local_dir = os.path.join(LOCAL_ASSETS_DIR, cat)
        if os.path.exists(local_dir):
            file_count = sum(1 for _ in Path(local_dir).rglob("*") if _.is_file())
            print(f"✓ {cat}: {file_count} files")
        else:
            print(f"✗ {cat}: Not found")


def main():
    parser = argparse.ArgumentParser(description="Download Isaac Lab assets from Nucleus to local storage")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(ASSET_CATEGORIES.keys()) + ["all"],
        default=["all"],
        help="Asset categories to download (default: all)"
    )
    parser.add_argument(
        "--subset",
        type=str,
        help="Download only specific subset (e.g., 'ANYbotics' or 'Unitree' for robots)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing downloads without downloading"
    )
    
    args = parser.parse_args()
    
    try:
        print("\n" + "="*60)
        print("Isaac Lab Asset Downloader")
        print("="*60)
        print(f"Isaac Sim assets:  {ISAAC_NUCLEUS_DIR}")
        print(f"Isaac Lab assets:  {ISAACLAB_NUCLEUS_DIR}")
        print(f"Local target:      {LOCAL_ASSETS_DIR}")
        print("="*60)
        
        if args.verify_only:
            verify_downloads()
            return
        
        # Determine which categories to download
        categories = (
            list(ASSET_CATEGORIES.keys()) 
            if "all" in args.categories 
            else args.categories
        )
        
        print(f"\nWill download: {', '.join(categories)}")
        if args.subset:
            print(f"Subset filter: {args.subset}")
        
        # Confirm before proceeding
        response = input("\nProceed with download? [y/N]: ")
        if response.lower() not in ["y", "yes"]:
            print("Download cancelled")
            return
        
        # Download each category
        for category in categories:
            try:
                download_asset_category(category, args.subset)
            except KeyboardInterrupt:
                print("\n\nDownload interrupted by user")
                raise
            except Exception as e:
                print(f"\n❌ Error downloading {category}: {e}")
                continue
        
        # Verify downloads
        verify_downloads()
        
        print("\n" + "="*60)
        print("✓ Download complete!")
        print("="*60)
        print(f"\nLocal assets are now available in: {LOCAL_ASSETS_DIR}")
        print("\nYou can now use the --local flag in training:")
        print("  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\")
        print("      --task Isaac-Velocity-Flat-Unitree-Go2-v0 \\")
        print("      --num_envs 128 \\")
        print("      --local")
    
    finally:
        # Always clean up simulation app
        simulation_app.close()


if __name__ == "__main__":
    main()