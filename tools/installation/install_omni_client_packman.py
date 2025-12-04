#!/usr/bin/env python3
"""
Install omni.client from a prebuilt 7z payload into the current Python environment.

- Downloads https://d4i3qtqj3r0z5.cloudfront.net/omni_client_library.linux-x86_64@<version>.7z
- Extracts to a cache dir (default: $TMPDIR/omni_client_cache; override with OMNI_CLIENT_CACHE)
- Copies the Python package (release/bindings-python) and native libs (release/*.so) into site-packages/_omni_client
- Drops a .pth for import visibility
- Creates a minimal dist-info so `pip uninstall omni-client-offline` works

# TODO: Once pip has been shipped, remove this script and use pip install omniverseclient==<version> instead.
"""

import os
import pathlib
import shutil
import site
import subprocess
import sys
import tempfile
import urllib.request
import logging

# Ensure py7zr is available
try:
    import py7zr  # type: ignore  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "py7zr"])
    import py7zr  # type: ignore


logger = logging.getLogger(__name__)

# Configuration
pkg_ver = os.environ.get("OMNI_CLIENT_VERSION", "2.68.1")
cache_root = pathlib.Path(os.environ.get("OMNI_CLIENT_CACHE", tempfile.gettempdir())) / "omni_client_cache"
payload_url = f"https://d4i3qtqj3r0z5.cloudfront.net/omni_client_library.linux-x86_64%40{pkg_ver}.7z"

# Paths
cache_root.mkdir(parents=True, exist_ok=True)
payload = cache_root / f"omni_client.{pkg_ver}.7z"
extract_root = cache_root / f"omni_client.{pkg_ver}.extracted"

# Download payload if missing
if not payload.exists():
    logger.info(f" Downloading omni.client payload from {payload_url} ...")
    urllib.request.urlretrieve(payload_url, payload)

# Extract payload
extract_root.mkdir(parents=True, exist_ok=True)
with py7zr.SevenZipFile(payload, mode="r") as z:
    z.extractall(path=extract_root)

# Locate python package and native libs
src_py = extract_root / "release" / "bindings-python"
if not (src_py / "omni" / "client").exists():
    raise RuntimeError(f"Could not locate omni.client python package at {src_py}")

src_lib = extract_root / "release"
if not any(src_lib.glob("libomni*.so*")):
    raise RuntimeError(f"Could not locate native libs under {src_lib}")

# Install into site-packages
site_pkgs = pathlib.Path(site.getsitepackages()[0] if hasattr(site, "getsitepackages") else site.getusersitepackages())
dest = site_pkgs / "_omni_client"
dest.mkdir(parents=True, exist_ok=True)
shutil.copytree(src_py, dest, dirs_exist_ok=True)
shutil.copytree(src_lib, dest / "lib", dirs_exist_ok=True)

# Ensure the extension can find its libs without env vars
client_dir = dest / "omni" / "client"
client_dir.mkdir(parents=True, exist_ok=True)
for libfile in (dest / "lib").glob("libomni*.so*"):
    target = client_dir / libfile.name
    if not target.exists():
        try:
            target.symlink_to(libfile)
        except Exception:
            shutil.copy2(libfile, target)

# Add .pth for import visibility
with open(site_pkgs / "omni_client.pth", "w", encoding="utf-8") as f:
    f.write(str(dest) + "\n")
    f.write(str(dest / "lib") + "\n")

# Minimal dist-info so pip can uninstall (pip uninstall omni-client)
dist_name = "omni-client"
dist_info = site_pkgs / f"{dist_name.replace('-', '_')}-{pkg_ver}.dist-info"
dist_info.mkdir(parents=True, exist_ok=True)
(dist_info / "INSTALLER").write_text("manual\n", encoding="utf-8")
metadata = "\n".join(
    [
        f"Name: {dist_name}",
        f"Version: {pkg_ver}",
        "Summary: Offline omni.client bundle",
        "",
    ]
)
(dist_info / "METADATA").write_text(metadata, encoding="utf-8")

records = []
for path in [
    site_pkgs / "omni_client.pth",
    dist_info / "INSTALLER",
    dist_info / "METADATA",
]:
    records.append(f"{path.relative_to(site_pkgs)},,")
for path in dest.rglob("*"):
    records.append(f"{path.relative_to(site_pkgs)},,")
for path in dist_info.rglob("*"):
    if path.name != "RECORD":
        records.append(f"{path.relative_to(site_pkgs)},,")
(dist_info / "RECORD").write_text("\n".join(records), encoding="utf-8")

logger.info(f"Installed omni.client to {dest} (dist: {dist_info.name})")
logger.info("Uninstall with: pip uninstall omni-client")
