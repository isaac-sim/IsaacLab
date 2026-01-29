# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Download Isaac assets from Nucleus server for offline use.

This script mirrors the Nucleus Isaac/ directory structure locally:

    Nucleus: .../Assets/Isaac/5.1/Isaac/
    Local:   offline_assets/

Usage:
    # Download Isaac Lab essentials (default)
    ./isaaclab.sh -p scripts/offline_setup/download_assets.py

    # Download everything from Isaac/
    ./isaaclab.sh -p scripts/offline_setup/download_assets.py --categories all

    # Download specific directories
    ./isaaclab.sh -p scripts/offline_setup/download_assets.py --categories IsaacLab Props

    # Download specific subdirectory
    ./isaaclab.sh -p scripts/offline_setup/download_assets.py --categories IsaacLab/Robots --subset Unitree
"""

import argparse
import os
from pathlib import Path

from tqdm import tqdm

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import carb
import omni.client

ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", os.getcwd())
OFFLINE_ASSETS_DIR = os.path.join(ISAACLAB_PATH, "offline_assets")

settings = carb.settings.get_settings()
NUCLEUS_ASSET_ROOT = settings.get("/persistent/isaac/asset_root/default")
ISAAC_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT}/Isaac"

KNOWN_CATEGORIES = {
    "IsaacLab": "Isaac Lab assets (Robots, ActuatorNets, Controllers, Policies, etc.)",
    "Props": "Props, markers, UI elements",
    "Environments": "Environment assets, ground planes",
    "Materials": "Materials, textures, HDRs",
    "Robots": "Isaac Sim robots (separate from IsaacLab/Robots)",
    "Sensors": "Sensor assets",
    "People": "Human assets",
    "Samples": "Sample scenes",
}


def format_size(bytes_size: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def get_local_directory_size(path: str) -> int:
    total_size = 0
    if os.path.exists(path):
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    return total_size


def get_remote_directory_info(remote_path: str) -> tuple[int, int]:
    file_count = 0
    total_size = 0

    result, entries = omni.client.list(remote_path)
    if result != omni.client.Result.OK:
        return 0, 0

    for entry in entries:
        is_dir = entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN
        remote_item = f"{remote_path}/{entry.relative_path}"

        if is_dir:
            sub_count, sub_size = get_remote_directory_info(remote_item)
            file_count += sub_count
            total_size += sub_size
        else:
            file_count += 1
            stat_result, stat_entry = omni.client.stat(remote_item)
            if stat_result == omni.client.Result.OK:
                total_size += stat_entry.size

    return file_count, total_size


def list_remote_directories(remote_path: str) -> list[str]:
    result, entries = omni.client.list(remote_path)
    if result != omni.client.Result.OK:
        return []
    return sorted([e.relative_path for e in entries if e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN])


def download_file(remote_path: str, local_path: str, overwrite: bool = False) -> bool:
    try:
        if os.path.exists(local_path) and not overwrite:
            return True
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        result = omni.client.copy(remote_path, local_path)
        return result == omni.client.Result.OK
    except Exception as e:
        print(f"Error downloading {remote_path}: {e}")
        return False


def download_directory_recursive(remote_path: str, local_base: str, progress_bar, overwrite: bool = False) -> int:
    downloaded = 0
    result, entries = omni.client.list(remote_path)
    if result != omni.client.Result.OK:
        return downloaded

    for entry in entries:
        is_dir = entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN
        remote_item = f"{remote_path}/{entry.relative_path}"
        local_item = os.path.join(local_base, entry.relative_path)

        if is_dir:
            os.makedirs(local_item, exist_ok=True)
            downloaded += download_directory_recursive(remote_item, local_item, progress_bar, overwrite)
        else:
            progress_bar.set_description(f"Downloading {entry.relative_path[:50]}")
            if download_file(remote_item, local_item, overwrite):
                downloaded += 1
            progress_bar.update(1)

    return downloaded


def download_category(category: str, subset: str = None, overwrite: bool = False) -> None:
    remote_dir = f"{ISAAC_NUCLEUS_DIR}/{category}"
    local_dir = os.path.join(OFFLINE_ASSETS_DIR, category)

    if subset:
        remote_dir = f"{remote_dir}/{subset}"
        local_dir = os.path.join(local_dir, subset)

    base_category = category.split("/")[0]
    desc = KNOWN_CATEGORIES.get(base_category, "Assets")

    print(f"\n{'=' * 70}")
    print(f"📦 {category}: {desc}")
    print(f"{'=' * 70}")
    print(f"Source: {remote_dir}")
    print(f"Target: {local_dir}")

    result, _ = omni.client.stat(remote_dir)
    if result != omni.client.Result.OK:
        print(f"⚠️  Directory not found: {remote_dir}")
        return

    print("📊 Analyzing...")
    file_count, total_size = get_remote_directory_info(remote_dir)

    if file_count == 0:
        print("✓ No files")
        return

    print(f"   Files: {file_count:,} | Size: {format_size(total_size)}")

    os.makedirs(local_dir, exist_ok=True)
    with tqdm(total=file_count, unit="file", desc="Progress") as pbar:
        download_directory_recursive(remote_dir, local_dir, pbar, overwrite)

    print(f"✓ Completed {category}")


def verify_downloads() -> None:
    print("\n" + "=" * 70)
    print("📊 Downloaded Assets Summary")
    print("=" * 70)
    print(f"Location: {OFFLINE_ASSETS_DIR}\n")

    total_size = 0
    total_files = 0

    if not os.path.exists(OFFLINE_ASSETS_DIR):
        print("❌ Offline assets directory not found")
        return

    for item in sorted(os.listdir(OFFLINE_ASSETS_DIR)):
        item_path = os.path.join(OFFLINE_ASSETS_DIR, item)
        if os.path.isdir(item_path):
            size = get_local_directory_size(item_path)
            files = sum(1 for _ in Path(item_path).rglob("*") if _.is_file())
            total_size += size
            total_files += files
            print(f"✓ {item:<20} {files:>8,} files    {format_size(size):>10}")

    print("=" * 70)
    print(f"{'TOTAL':<20} {total_files:>8,} files    {format_size(total_size):>10}")
    print("=" * 70)


def get_isaaclab_essential_categories() -> list[str]:
    return ["IsaacLab", "Props", "Environments", "Materials"]


def main():
    parser = argparse.ArgumentParser(
        description="Download Isaac assets for offline use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--categories", nargs="+", default=["isaaclab-essentials"])
    parser.add_argument("--subset", type=str, help="Subset within category")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list", action="store_true")

    args = parser.parse_args()

    try:
        print("\n" + "=" * 70)
        print("🚀 Isaac Lab Offline Asset Downloader")
        print("=" * 70)
        print(f"Nucleus: {ISAAC_NUCLEUS_DIR}")
        print(f"Local:   {OFFLINE_ASSETS_DIR}")
        print("=" * 70)

        if args.list:
            categories = list_remote_directories(ISAAC_NUCLEUS_DIR)
            print("\n📂 Available under Isaac/:")
            for cat in categories:
                desc = KNOWN_CATEGORIES.get(cat, "")
                print(f"   • {cat:<20} {desc}")
            return

        if args.verify_only:
            verify_downloads()
            return

        if "all" in args.categories:
            categories = list_remote_directories(ISAAC_NUCLEUS_DIR)
        elif "isaaclab-essentials" in args.categories:
            categories = get_isaaclab_essential_categories()
            print("\n📋 Isaac Lab essentials:")
            for cat in categories:
                print(f"   • {cat}")
        else:
            categories = args.categories

        if args.subset:
            print(f"🔍 Subset: {args.subset}")

        print("\n📊 Calculating size...")
        total_files = 0
        total_size = 0

        for category in categories:
            remote_dir = f"{ISAAC_NUCLEUS_DIR}/{category}"
            if args.subset:
                remote_dir = f"{remote_dir}/{args.subset}"

            result, _ = omni.client.stat(remote_dir)
            if result == omni.client.Result.OK:
                files, size = get_remote_directory_info(remote_dir)
                total_files += files
                total_size += size
                print(f"   {category}: {files:,} files ({format_size(size)})")

        if total_files == 0:
            print("\n❌ No files to download")
            return

        print(f"\n📦 Total: {total_files:,} files ({format_size(total_size)})")

        response = input("\nProceed? [y/N]: ")
        if response.lower() not in ["y", "yes"]:
            print("❌ Cancelled")
            return

        print("\n🔽 Downloading...")
        for category in categories:
            try:
                download_category(category, args.subset, args.overwrite)
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted")
                raise
            except Exception as e:
                print(f"❌ Error: {category}: {e}")

        verify_downloads()
        print("\n✅ Complete! Use --offline flag with Isaac Lab commands.\n")

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
